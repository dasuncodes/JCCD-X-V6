"""Cross-validation stability analysis for model evaluation."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from src.python.evaluation.metrics import compute_cv_stability, aggregate_cv_metrics
from src.python.evaluation.plots import plot_cv_variance
from src.python.model.models import get_models
from src.python.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_fold_correlation(
    fold_predictions: Dict[int, np.ndarray],
) -> Dict:
    """
    Compute correlation between fold predictions.

    Args:
        fold_predictions: Dictionary mapping fold index to predictions

    Returns:
        Dictionary with correlation matrix and statistics
    """
    n_folds = len(fold_predictions)
    if n_folds < 2:
        return {"correlation_matrix": [], "mean_correlation": 0}

    # Build prediction matrix
    pred_matrix = np.array([fold_predictions[i] for i in range(n_folds)])

    # Compute correlation matrix
    corr_matrix = np.corrcoef(pred_matrix)

    # Extract upper triangle (excluding diagonal)
    upper_tri = corr_matrix[np.triu_indices(n_folds, k=1)]
    mean_corr = float(np.mean(upper_tri))

    return {
        "correlation_matrix": corr_matrix.tolist(),
        "mean_correlation": mean_corr,
        "std_correlation": float(np.std(upper_tri)),
        "min_correlation": float(np.min(upper_tri)),
        "max_correlation": float(np.max(upper_tri)),
    }


def compute_feature_importance_stability(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Compute stability of feature importance across CV folds.

    Args:
        model: Sklearn estimator with feature_importances_
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        cv: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary with feature importance stability metrics
    """
    if not hasattr(model, "feature_importances_"):
        return {"error": "Model does not support feature importance"}

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    importances = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        model.fit(X_train, y_train)
        importances.append(model.feature_importances_)

    importances = np.array(importances)

    # Compute statistics per feature
    stability_metrics = {}
    for i, name in enumerate(feature_names):
        imp_values = importances[:, i]
        stability_metrics[name] = {
            "mean": float(np.mean(imp_values)),
            "std": float(np.std(imp_values)),
            "cv": float(np.std(imp_values) / np.mean(imp_values)) if np.mean(imp_values) > 0 else 0,
            "min": float(np.min(imp_values)),
            "max": float(np.max(imp_values)),
            "rank_mean": 0,  # Will be computed below
        }

    # Compute mean rank for each feature
    ranks = np.argsort(np.argsort(-importances, axis=1), axis=1) + 1
    for i, name in enumerate(feature_names):
        stability_metrics[name]["rank_mean"] = float(np.mean(ranks[:, i]))
        stability_metrics[name]["rank_std"] = float(np.std(ranks[:, i]))

    # Overall stability score (lower CV = more stable)
    all_cvs = [stability_metrics[name]["cv"] for name in feature_names]
    overall_stability = {
        "mean_cv": float(np.mean(all_cvs)),
        "std_cv": float(np.std(all_cvs)),
        "stable_features": sum(1 for cv in all_cvs if cv < 0.5),  # CV < 0.5 is stable
        "unstable_features": sum(1 for cv in all_cvs if cv > 1.0),  # CV > 1.0 is unstable
    }

    # Rank features by stability
    sorted_by_stability = sorted(
        stability_metrics.items(),
        key=lambda x: x[1]["cv"]
    )
    stability_metrics["most_stable"] = sorted_by_stability[0][0]
    stability_metrics["least_stable"] = sorted_by_stability[-1][0]

    return {
        "feature_stability": stability_metrics,
        "overall_stability": overall_stability,
        "importances": importances.tolist(),
    }


def compute_prediction_stability(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Compute stability of predictions across CV folds.

    Args:
        model: Sklearn estimator
        X: Feature matrix
        y: Labels
        cv: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary with prediction stability metrics
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    # Get predictions for each fold
    fold_predictions = {}
    fold_probabilities = {}
    fold_metrics = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        fold_predictions[fold_idx] = y_pred
        fold_metrics[fold_idx] = {
            "accuracy": float(np.mean(y_pred == y_val)),
            "n_samples": len(y_val),
        }

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
            fold_probabilities[fold_idx] = y_proba

    # Compute correlation between fold predictions
    pred_correlation = compute_fold_correlation(fold_predictions)

    # Compute prediction agreement
    # For each sample, how often do different folds agree?
    # (This requires overlapping validation sets, so we approximate)
    all_preds = np.concatenate([fold_predictions[i] for i in range(cv)])
    agreement_rate = float(np.mean(all_preds == np.round(np.mean(all_preds))))

    return {
        "fold_predictions": {k: v.tolist() for k, v in fold_predictions.items()},
        "fold_metrics": fold_metrics,
        "prediction_correlation": pred_correlation,
        "agreement_rate": agreement_rate,
    }


def compare_model_stability(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Compare stability across different models.

    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        cv: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary with stability comparison for all models
    """
    models = get_models()
    results = {}

    for name, model in models.items():
        logger.info("Analyzing stability for %s...", name)

        # Metric stability
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        metrics = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = np.mean(y_pred == y_val)
            metrics.append(acc)

        metric_stability = {
            "mean": float(np.mean(metrics)),
            "std": float(np.std(metrics)),
            "cv": float(np.std(metrics) / np.mean(metrics)) if np.mean(metrics) > 0 else 0,
        }

        # Feature importance stability (if applicable)
        model_copy = get_models()[name]  # Fresh copy
        if hasattr(model_copy, "feature_importances_"):
            fi_stability = compute_feature_importance_stability(
                model_copy, X, y, feature_names, cv, seed
            )
        else:
            fi_stability = {"available": False}

        results[name] = {
            "metric_stability": metric_stability,
            "feature_importance_stability": fi_stability,
        }

    # Rank models by stability
    stability_scores = [
        (name, results[name]["metric_stability"]["cv"])
        for name in results
    ]
    stability_scores.sort(key=lambda x: x[1])  # Lower CV = more stable

    results["ranking"] = [
        {"rank": i + 1, "model": name, "cv": cv}
        for i, (name, cv) in enumerate(stability_scores)
    ]
    results["most_stable_model"] = stability_scores[0][0]
    results["least_stable_model"] = stability_scores[-1][0]

    return results


def plot_stability_heatmap(
    correlation_matrix: List[List[float]],
    model_names: List[str],
    save_path: Path,
) -> None:
    """
    Plot heatmap of fold-to-fold correlation.

    Args:
        correlation_matrix: Correlation matrix
        model_names: List of model names
        save_path: Path to save the plot
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    corr = np.array(correlation_matrix)
    n_folds = corr.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(n_folds))
    ax.set_yticks(range(n_folds))
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)], fontsize=9)
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_folds)], fontsize=9)
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("Fold", fontsize=11)
    ax.set_title("Fold-to-Fold Prediction Correlation", fontsize=13)

    # Add value annotations
    for i in range(n_folds):
        for j in range(n_folds):
            val = corr[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_stability(
    feature_stability: Dict,
    save_path: Path,
    top_n: int = 15,
) -> None:
    """
    Plot feature importance stability.

    Args:
        feature_stability: Dictionary with feature stability metrics
        save_path: Path to save the plot
        top_n: Number of top features to show
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter out metadata keys
    features = {k: v for k, v in feature_stability.items()
                if k not in ["most_stable", "least_stable"]}

    # Sort by mean importance
    sorted_features = sorted(
        features.items(),
        key=lambda x: x[1].get("mean", 0),
        reverse=True
    )[:top_n]

    names = [f[0] for f in sorted_features]
    means = [f[1]["mean"] for f in sorted_features]
    stds = [f[1]["std"] for f in sorted_features]

    fig, ax = plt.subplots(figsize=(10, 8))

    x_pos = range(len(names))
    ax.barh(x_pos, means, xerr=stds, color="#2196F3", edgecolor="black", alpha=0.7)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean Importance ± Std Dev", fontsize=11)
    ax.set_title(f"Feature Importance Stability (Top {top_n})", fontsize=13)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CV stability analysis")
    parser.add_argument(
        "--features-dir", type=Path,
        default=Path("data/intermediate/features")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/evaluation")
    )
    parser.add_argument(
        "--model-name", type=str,
        default="xgboost",
        choices=["xgboost", "random_forest", "logistic_regression", "linear_svm", "knn"]
    )
    parser.add_argument("--cv", type=int, default=5)
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
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    logger.info("Loaded %d samples with %d features", X.shape[0], X.shape[1])

    # Get model
    models = get_models()
    model = models[args.model_name]

    # 1. Metric stability analysis
    logger.info("\n=== Metric Stability Analysis ===")
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Get probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]

        from src.python.evaluation.metrics import compute_fold_metrics
        fold_metrics = compute_fold_metrics(y_val, y_pred, y_proba, fold_idx)
        fold_results.append(fold_metrics)

    stability = compute_cv_stability(fold_results)
    aggregate = aggregate_cv_metrics(fold_results)

    logger.info("Stability metrics:")
    for key, value in stability.items():
        if "stability" in key:
            logger.info("  %s: %.4f", key, value)

    # 2. Feature importance stability
    logger.info("\n=== Feature Importance Stability ===")
    model_fi = get_models()[args.model_name]
    fi_stability = compute_feature_importance_stability(
        model_fi, X, y, feature_cols, args.cv, args.seed
    )

    if "overall_stability" in fi_stability:
        logger.info("Overall stability:")
        logger.info("  Mean CV: %.4f", fi_stability["overall_stability"]["mean_cv"])
        logger.info("  Stable features: %d", fi_stability["overall_stability"]["stable_features"])
        logger.info("  Most stable: %s", fi_stability.get("most_stable", "N/A"))

    # 3. Prediction stability
    logger.info("\n=== Prediction Stability ===")
    model_pred = get_models()[args.model_name]
    pred_stability = compute_prediction_stability(
        model_pred, X, y, args.cv, args.seed
    )
    logger.info("  Mean correlation: %.4f", pred_stability["prediction_correlation"]["mean_correlation"])
    logger.info("  Agreement rate: %.4f", pred_stability["agreement_rate"])

    # 4. Compare all models
    logger.info("\n=== Model Comparison ===")
    comparison = compare_model_stability(X, y, feature_cols, args.cv, args.seed)
    logger.info("Most stable model: %s", comparison["most_stable_model"])
    logger.info("Least stable model: %s", comparison["least_stable_model"])

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.model_name,
        "cv_folds": args.cv,
        "metric_stability": stability,
        "aggregate_metrics": aggregate,
        "feature_importance_stability": fi_stability,
        "prediction_stability": pred_stability,
        "model_comparison": comparison,
    }
    save_json(results, args.output_dir / "stability_results.json")

    # Generate plots
    plots_dir = args.output_dir / "plots" / "stability"

    # CV variance plot
    plot_cv_variance(
        fold_results,
        plots_dir / "cv_variance.png",
        metrics=["f1", "accuracy", "roc_auc"],
        title="Cross-Validation Metric Variance",
    )

    # Feature stability plot
    if "feature_stability" in fi_stability:
        plot_feature_stability(
            fi_stability["feature_stability"],
            plots_dir / "feature_stability.png",
            top_n=15,
        )

    # Fold correlation heatmap
    if "correlation_matrix" in pred_stability["prediction_correlation"]:
        plot_stability_heatmap(
            pred_stability["prediction_correlation"]["correlation_matrix"],
            [args.model_name],
            plots_dir / "fold_correlation_heatmap.png",
        )

    logger.info("Results saved to %s", args.output_dir)
    logger.info("Plots saved to %s", plots_dir)

    print("\n=== Stability Analysis Summary ===")
    print(f"Model: {args.model_name}")
    print(f"CV Folds: {args.cv}")
    print(f"\nMetric Stability:")
    for key, value in stability.items():
        if "stability" in key:
            print(f"  {key}: {value:.4f}")

    print(f"\nMost stable model: {comparison['most_stable_model']}")
    print(f"Least stable model: {comparison['least_stable_model']}")
    print("Done.")


if __name__ == "__main__":
    main()
