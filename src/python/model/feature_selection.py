"""Feature selection using Recursive Feature Elimination (RFE)."""

import argparse
import logging
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

from src.python.evaluation.metrics import aggregate_cv_metrics
from src.python.evaluation.plots import plot_feature_importance
from src.python.model.models import get_models
from src.python.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_rfe_analysis(
    X: np.ndarray,
    y: np.ndarray,
    model,
    model_name: str,
    n_features_range: list = None,
    cv: int = 5,
    seed: int = 42,
) -> dict:
    """
    Run Recursive Feature Elimination analysis.

    Args:
        X: Feature matrix
        y: Labels
        model: Sklearn estimator with fit and predict methods
        model_name: Name of the model for logging
        n_features_range: List of feature counts to test (default: [5, 10, 15, 20, 25])
        cv: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary with RFE results
    """
    if n_features_range is None:
        n_features_range = [5, 10, 15, 20, 25]

    n_total_features = X.shape[1]
    results = {
        "model": model_name,
        "total_features": n_total_features,
        "n_features_range": n_features_range,
        "experiments": [],
    }

    logger.info("Running RFE analysis for %s with %d features...", model_name, n_total_features)

    for n_features in n_features_range:
        if n_features > n_total_features:
            continue

        logger.info("  Testing with %d features...", n_features)
        start = time.time()

        # Run RFE
        selector = RFE(
            estimator=model,
            n_features_to_select=n_features,
            step=1,
        )
        selector.fit(X, y)

        # Get selected feature indices
        selected_indices = np.where(selector.support_)[0]

        # Evaluate with cross-validation
        scores = cross_val_score(model, X[:, selected_indices], y, cv=cv, scoring="f1")

        elapsed = time.time() - start

        experiment = {
            "n_features": n_features,
            "selected_indices": selected_indices.tolist(),
            "cv_f1_mean": float(np.mean(scores)),
            "cv_f1_std": float(np.std(scores)),
            "cv_accuracy_mean": float(np.mean(
                cross_val_score(model, X[:, selected_indices], y, cv=cv, scoring="accuracy")
            )),
            "elapsed_sec": elapsed,
        }
        results["experiments"].append(experiment)

        logger.info(
            "    %d features: F1=%.4f±%.4f (%.1fs)",
            n_features, np.mean(scores), np.std(scores), elapsed
        )

    # Find optimal number of features (elbow point)
    f1_scores = [exp["cv_f1_mean"] for exp in results["experiments"]]
    n_features_list = [exp["n_features"] for exp in results["experiments"]]

    # Simple elbow detection: find where adding more features gives diminishing returns
    if len(f1_scores) > 1:
        diffs = [f1_scores[i+1] - f1_scores[i] for i in range(len(f1_scores)-1)]
        # Find first point where improvement is < 1%
        optimal_idx = 0
        for i, diff in enumerate(diffs):
            if diff < 0.01:  # Less than 1% improvement
                optimal_idx = i
                break
        else:
            optimal_idx = len(f1_scores) - 1

        results["optimal_n_features"] = n_features_list[optimal_idx]
        results["optimal_f1"] = f1_scores[optimal_idx]
    else:
        results["optimal_n_features"] = n_features_list[0]
        results["optimal_f1"] = f1_scores[0]

    return results


def plot_rfe_results(
    rfe_results: dict,
    save_path: Path,
) -> None:
    """
    Plot RFE results: F1 score vs number of features.

    Args:
        rfe_results: Dictionary with RFE results
        save_path: Path to save the plot
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    experiments = rfe_results["experiments"]
    n_features = [exp["n_features"] for exp in experiments]
    f1_means = [exp["cv_f1_mean"] for exp in experiments]
    f1_stds = [exp["cv_f1_std"] for exp in experiments]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        n_features, f1_means, yerr=f1_stds,
        marker='o', linewidth=2, markersize=8,
        color='#2196F3', ecolor='#90CAF9', capsize=5,
        label='F1 Score'
    )

    # Mark optimal point
    optimal_n = rfe_results.get("optimal_n_features")
    optimal_f1 = rfe_results.get("optimal_f1")
    if optimal_n is not None:
        ax.scatter(
            [optimal_n], [optimal_f1],
            color='#4CAF50', s=200, zorder=5,
            label=f'Optimal: {optimal_n} features (F1={optimal_f1:.4f})',
            marker='*'
        )

    ax.set_xlabel("Number of Features", fontsize=11)
    ax.set_ylabel("F1 Score (Cross-Validation)", fontsize=11)
    ax.set_title(f"RFE Results - {rfe_results['model']}", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_features)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compare_feature_subsets(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    rfe_results: dict,
    model,
    cv: int = 5,
) -> dict:
    """
    Compare different feature subsets based on RFE ranking.

    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        rfe_results: RFE results with selected indices
        model: Model to evaluate
        cv: Number of CV folds

    Returns:
        Comparison results
    """
    comparison = {
        "subsets": [],
    }

    # Fit model on all features to get importance ranking
    model.fit(X, y)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        ranked_indices = np.argsort(importance)[::-1]

        for n_features in [5, 10, 15, 20]:
            if n_features > X.shape[1]:
                continue

            # Top-N features by importance
            top_n_indices = ranked_indices[:n_features]

            # Evaluate
            scores = cross_val_score(model, X[:, top_n_indices], y, cv=cv, scoring="f1")

            subset_result = {
                "n_features": n_features,
                "feature_indices": top_n_indices.tolist(),
                "feature_names": [feature_names[i] for i in top_n_indices],
                "cv_f1_mean": float(np.mean(scores)),
                "cv_f1_std": float(np.std(scores)),
            }
            comparison["subsets"].append(subset_result)

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature selection with RFE")
    parser.add_argument("--features-dir", type=Path, default=Path("data/intermediate/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--model-name", type=str, default="xgboost",
                       choices=["xgboost", "random_forest", "logistic_regression", "linear_svm", "knn"])
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load features
    train_path = args.features_dir / "train_features.csv"
    if not train_path.exists():
        logger.error("Run feature engineering first")
        return

    train_df = pd.read_csv(train_path)
    feature_cols = [c for c in train_df.columns if c not in ("label", "id1", "id2")]
    X = train_df[feature_cols].values.astype(np.float64)
    y = train_df["label"].values.astype(np.int32)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    logger.info("Loaded %d samples with %d features", X.shape[0], X.shape[1])

    # Get model
    models = get_models()
    if args.model_name not in models:
        logger.error("Model %s not found", args.model_name)
        return
    model = models[args.model_name]

    # Run RFE analysis
    n_features_range = [3, 5, 8, 10, 13, 15, 18, 20]
    rfe_results = run_rfe_analysis(
        X, y, model, args.model_name,
        n_features_range=n_features_range,
        cv=args.cv,
        seed=args.seed,
    )

    # Compare feature subsets
    comparison = compare_feature_subsets(
        X, y, feature_cols, rfe_results, model, cv=args.cv,
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(rfe_results, args.output_dir / "rfe_results.json")
    save_json(comparison, args.output_dir / "feature_subset_comparison.json")

    # Plot results
    plots_dir = args.output_dir / "plots" / "rfe"
    plot_rfe_results(rfe_results, plots_dir / "rfe_curve.png")

    # Plot feature importance with optimal features
    optimal_n = rfe_results.get("optimal_n_features")
    if optimal_n and rfe_results["experiments"]:
        # Find experiment with optimal features
        for exp in rfe_results["experiments"]:
            if exp["n_features"] == optimal_n:
                # Get feature importance from model trained on selected features
                selected_indices = exp["selected_indices"]
                model.fit(X[:, selected_indices], y)

                if hasattr(model, "feature_importances_"):
                    importance = {}
                    for i, idx in enumerate(selected_indices):
                        importance[feature_cols[idx]] = float(model.feature_importances_[i])

                    plot_feature_importance(
                        importance,
                        plots_dir / "optimal_feature_importance.png",
                        title=f"Feature Importance (Top {optimal_n} Features)",
                        top_n=optimal_n,
                    )
                break

    logger.info("Results saved to %s", args.output_dir)
    logger.info("Plots saved to %s", plots_dir)

    print("\n=== RFE Summary ===")
    print(f"Model: {args.model_name}")
    print(f"Total features: {X.shape[1]}")
    print(f"Optimal features: {optimal_n}")
    print(f"Optimal F1: {rfe_results.get('optimal_f1', 0):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
