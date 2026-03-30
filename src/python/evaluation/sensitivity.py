"""Sensitivity analysis for model robustness evaluation."""

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
from sklearn.model_selection import cross_val_score

from src.python.evaluation.metrics import compute_classification_metrics
from src.python.evaluation.plots import plot_ablation_results
from src.python.model.models import get_models
from src.python.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def add_gaussian_noise(
    X: np.ndarray,
    noise_level: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Add Gaussian noise to features.

    Args:
        X: Feature matrix
        noise_level: Standard deviation of noise as fraction of feature std
        seed: Random seed

    Returns:
        Noisy feature matrix
    """
    np.random.seed(seed)
    noise = np.random.normal(0, noise_level, X.shape)
    # Scale noise by feature standard deviation
    feature_std = np.std(X, axis=0, keepdims=True)
    feature_std = np.where(feature_std == 0, 1, feature_std)  # Avoid division by zero
    X_noisy = X + noise * feature_std
    return X_noisy


def mask_features(
    X: np.ndarray,
    mask_ratio: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Randomly mask (zero out) features.

    Args:
        X: Feature matrix
        mask_ratio: Fraction of features to mask
        seed: Random seed

    Returns:
        Feature matrix with masked features
    """
    np.random.seed(seed)
    X_masked = X.copy()
    n_features = X.shape[1]
    n_mask = int(n_features * mask_ratio)

    # Randomly select features to mask
    mask_indices = np.random.choice(n_features, n_mask, replace=False)
    X_masked[:, mask_indices] = 0

    return X_masked


def shift_feature_values(
    X: np.ndarray,
    shift_ratio: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Shift feature values by a percentage.

    Args:
        X: Feature matrix
        shift_ratio: Fraction to shift values
        seed: Random seed

    Returns:
        Shifted feature matrix
    """
    np.random.seed(seed)
    X_shifted = X.copy()

    # Random shift direction per feature
    directions = np.random.choice([-1, 1], X.shape[1])
    shift = directions * shift_ratio * np.abs(X)

    X_shifted = X + shift
    return X_shifted


def evaluate_robustness(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    perturbation_type: str,
    perturbation_levels: List[float],
    cv: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Evaluate model robustness to perturbations.

    Args:
        model: Sklearn estimator
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        perturbation_type: Type of perturbation ("noise", "mask", "shift")
        perturbation_levels: List of perturbation levels to test
        cv: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary with robustness metrics
    """
    results = {
        "perturbation_type": perturbation_type,
        "baseline": {},
        "perturbations": [],
    }

    # Baseline (no perturbation)
    logger.info("Evaluating baseline (no perturbation)...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    baseline_metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    results["baseline"] = {
        "f1": baseline_metrics["f1"],
        "accuracy": baseline_metrics["accuracy"],
        "precision": baseline_metrics["precision"],
        "recall": baseline_metrics["recall"],
    }

    # CV scores for baseline
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    results["baseline"]["cv_f1_mean"] = float(np.mean(cv_scores))
    results["baseline"]["cv_f1_std"] = float(np.std(cv_scores))

    # Evaluate at each perturbation level
    for level in perturbation_levels:
        logger.info("Evaluating %s at level %.2f...", perturbation_type, level)

        # Apply perturbation
        if perturbation_type == "noise":
            X_train_pert = add_gaussian_noise(X_train, level, seed)
            X_test_pert = add_gaussian_noise(X_test, level, seed)
        elif perturbation_type == "mask":
            X_train_pert = mask_features(X_train, level, seed)
            X_test_pert = mask_features(X_test, level, seed)
        elif perturbation_type == "shift":
            X_train_pert = shift_feature_values(X_train, level, seed)
            X_test_pert = shift_feature_values(X_test, level, seed)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        # Evaluate
        model_pert = get_models()[type(model).__name__.lower().replace("classifier", "")]
        model_pert.fit(X_train_pert, y_train)
        y_pred_pert = model_pert.predict(X_test_pert)

        if hasattr(model_pert, "predict_proba"):
            y_proba_pert = model_pert.predict_proba(X_test_pert)[:, 1]
        else:
            y_proba_pert = None

        metrics = compute_classification_metrics(y_test, y_pred_pert, y_proba_pert)

        # Compute degradation
        f1_degradation = baseline_metrics["f1"] - metrics["f1"]
        acc_degradation = baseline_metrics["accuracy"] - metrics["accuracy"]

        perturbation_result = {
            "level": level,
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_degradation": f1_degradation,
            "f1_degradation_pct": f1_degradation / max(baseline_metrics["f1"], 0.001) * 100,
            "accuracy_degradation": acc_degradation,
        }
        results["perturbations"].append(perturbation_result)

    # Compute robustness score (area under degradation curve)
    levels = [p["level"] for p in results["perturbations"]]
    f1_scores = [p["f1"] for p in results["perturbations"]]
    robustness_score = np.trapz(f1_scores, levels) / (max(levels) - min(levels))
    results["robustness_score"] = float(robustness_score)

    return results


def compare_model_robustness(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    perturbation_type: str = "noise",
    perturbation_levels: List[float] = None,
    cv: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Compare robustness across different models.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        perturbation_type: Type of perturbation
        perturbation_levels: List of perturbation levels
        cv: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary with robustness comparison
    """
    if perturbation_levels is None:
        perturbation_levels = [0.05, 0.10, 0.15, 0.20, 0.25]

    models = get_models()
    results = {}

    for name, model in models.items():
        logger.info("Evaluating robustness for %s...", name)
        model_robustness = evaluate_robustness(
            model, X_train, y_train, X_test, y_test,
            perturbation_type, perturbation_levels, cv, seed
        )
        results[name] = model_robustness

    # Rank models by robustness
    robustness_scores = [
        (name, results[name]["robustness_score"])
        for name in results
    ]
    robustness_scores.sort(key=lambda x: x[1], reverse=True)  # Higher = more robust

    results["ranking"] = [
        {"rank": i + 1, "model": name, "robustness_score": score}
        for i, (name, score) in enumerate(robustness_scores)
    ]
    results["most_robust_model"] = robustness_scores[0][0]
    results["least_robust_model"] = robustness_scores[-1][0]

    return results


def plot_sensitivity_results(
    results: Dict,
    save_dir: Path,
) -> None:
    """
    Generate sensitivity analysis plots.

    Args:
        results: Dictionary with sensitivity results
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Degradation curve
    fig, ax = plt.subplots(figsize=(10, 6))

    for pert in results["perturbations"]:
        pert_type = results["perturbation_type"]
        ax.plot(
            [pert["level"]], [pert["f1"]],
            marker="o", markersize=10,
            label=f"{pert_type} level={pert['level']:.2f}: F1={pert['f1']:.3f}"
        )

    # Baseline
    ax.axhline(
        results["baseline"]["f1"],
        color="green", linestyle="--", linewidth=2,
        label=f"Baseline F1={results['baseline']['f1']:.3f}"
    )

    ax.set_xlabel("Perturbation Level", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title(f"Sensitivity to {results['perturbation_type']}", fontsize=13)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "degradation_curve.png", dpi=150)
    plt.close(fig)

    # 2. F1 degradation percentage
    fig, ax = plt.subplots(figsize=(10, 6))

    levels = [p["level"] for p in results["perturbations"]]
    degradation = [p["f1_degradation_pct"] for p in results["perturbations"]]

    ax.bar(levels, degradation, color="#F44336", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Perturbation Level", fontsize=11)
    ax.set_ylabel("F1 Degradation (%)", fontsize=11)
    ax.set_title(f"F1 Score Degradation by {results['perturbation_type']} Level", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(save_dir / "degradation_percentage.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(
    comparison_results: Dict,
    save_dir: Path,
) -> None:
    """
    Plot model robustness comparison.

    Args:
        comparison_results: Dictionary with model comparison results
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Robustness scores bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(comparison_results.keys())
    if "ranking" in comparison_results:
        models = [m for m in models if m != "ranking" and m not in ["most_robust_model", "least_robust_model"]]

    scores = [comparison_results[m]["robustness_score"] for m in models]

    colors = ["#4CAF50" if m == comparison_results.get("most_robust_model") else "#2196F3"
              for m in models]

    ax.bar(models, scores, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Robustness Score", fontsize=11)
    ax.set_title("Model Robustness Comparison", fontsize=13)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (model, score) in enumerate(zip(models, scores)):
        ax.text(i, score + 0.01, f"{score:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_dir / "model_robustness_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity analysis")
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
    parser.add_argument(
        "--perturbation-types", type=str, nargs="+",
        default=["noise", "mask", "shift"],
        help="Types of perturbations to test"
    )
    parser.add_argument(
        "--perturbation-levels", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25],
        help="Perturbation levels to test"
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load features
    train_path = args.features_dir / "train_features.csv"
    test_path = args.features_dir / "test_features.csv"

    if not train_path.exists():
        logger.error("Run feature engineering first")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in ("label", "id1", "id2")]
    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df["label"].values.astype(np.int32)
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df["label"].values.astype(np.int32)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=0.0)

    logger.info("Loaded %d training samples, %d test samples", len(y_train), len(y_test))

    # Get model
    models = get_models()
    model = models[args.model_name]

    # 1. Evaluate robustness for each perturbation type
    logger.info("\n=== Sensitivity Analysis ===")
    results = {}

    for pert_type in args.perturbation_types:
        logger.info("\nEvaluating sensitivity to %s...", pert_type)
        pert_results = evaluate_robustness(
            model, X_train, y_train, X_test, y_test,
            pert_type, args.perturbation_levels, args.cv, args.seed
        )
        results[pert_type] = pert_results

        logger.info("Robustness score: %.4f", pert_results["robustness_score"])
        logger.info("F1 degradation at max perturbation: %.1f%%",
                    pert_results["perturbations"][-1]["f1_degradation_pct"])

    # 2. Compare all models
    logger.info("\n=== Model Robustness Comparison ===")
    comparison = compare_model_robustness(
        X_train, y_train, X_test, y_test,
        perturbation_type="noise",
        perturbation_levels=args.perturbation_levels,
        cv=args.cv,
        seed=args.seed,
    )

    logger.info("Most robust model: %s", comparison["most_robust_model"])
    logger.info("Least robust model: %s", comparison["least_robust_model"])

    # 3. Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "model": args.model_name,
        "sensitivity": results,
        "model_comparison": comparison,
    }
    save_json(all_results, args.output_dir / "sensitivity_results.json")

    # 4. Generate plots
    plots_dir = args.output_dir / "plots" / "sensitivity"

    for pert_type, pert_results in results.items():
        pert_dir = plots_dir / pert_type
        plot_sensitivity_results(pert_results, pert_dir)

    # Model comparison plot
    plot_model_comparison(comparison, plots_dir)

    logger.info("Results saved to %s", args.output_dir)
    logger.info("Plots saved to %s", plots_dir)

    print("\n=== Sensitivity Analysis Summary ===")
    print(f"Model: {args.model_name}")
    print(f"\nRobustness scores by perturbation type:")
    for pert_type, pert_results in results.items():
        print(f"  {pert_type}: {pert_results['robustness_score']:.4f}")

    print(f"\nMost robust model: {comparison['most_robust_model']}")
    print(f"Least robust model: {comparison['least_robust_model']}")
    print("Done.")


if __name__ == "__main__":
    main()
