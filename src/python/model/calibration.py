"""Probability calibration for improved model predictions."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from src.python.evaluation.metrics import compute_classification_metrics
from src.python.model.models import get_models
from src.python.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calibrate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    method: str = "sigmoid",
    cv: int = 5,
    seed: int = 42,
) -> Tuple:
    """
    Calibrate a model using Platt scaling or isotonic regression.

    Args:
        model: Sklearn estimator to calibrate
        X: Feature matrix
        y: Labels
        method: Calibration method ("sigmoid" for Platt, "isotonic" for isotonic)
        cv: Number of CV folds
        seed: Random seed

    Returns:
        Tuple of (calibrated_model, calibration_results)
    """
    logger.info("Calibrating model with %s method...", method)

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv,
    )
    calibrated.fit(X, y)

    # Get calibrated probabilities
    y_proba = calibrated.predict_proba(X)[:, 1]

    # Compute calibration metrics
    brier = brier_score_loss(y, y_proba)
    logloss = log_loss(y, y_proba)

    # Compute calibration curve data
    prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)

    results = {
        "method": method,
        "brier_score": float(brier),
        "log_loss": float(logloss),
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
    }

    return calibrated, results


def compare_calibration_methods(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Compare different calibration methods.

    Args:
        model: Sklearn estimator
        X: Feature matrix
        y: Labels
        cv: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary with comparison results
    """
    results = {
        "uncalibrated": {},
        "sigmoid": {},
        "isotonic": {},
    }

    # Uncalibrated (baseline)
    logger.info("Evaluating uncalibrated model...")
    model.fit(X, y)
    y_proba_uncal = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)

    results["uncalibrated"] = {
        "brier_score": float(brier_score_loss(y, y_proba_uncal)),
        "log_loss": float(log_loss(y, y_proba_uncal)),
    }

    # Sigmoid calibration (Platt scaling)
    logger.info("Calibrating with sigmoid (Platt scaling)...")
    model_sigmoid = get_models()[type(model).__name__.lower().replace("classifier", "")]
    _, sigmoid_results = calibrate_model(model_sigmoid, X, y, method="sigmoid", cv=cv, seed=seed)
    results["sigmoid"] = sigmoid_results

    # Isotonic calibration
    logger.info("Calibrating with isotonic regression...")
    model_isotonic = get_models()[type(model).__name__.lower().replace("classifier", "")]
    _, isotonic_results = calibrate_model(model_isotonic, X, y, method="isotonic", cv=cv, seed=seed)
    results["isotonic"] = isotonic_results

    # Determine best method
    methods = ["uncalibrated", "sigmoid", "isotonic"]
    best_method = min(methods, key=lambda m: results[m]["brier_score"])
    results["best_method"] = best_method
    results["best_brier_score"] = results[best_method]["brier_score"]

    return results


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> Dict:
    """
    Find optimal probability threshold for classification.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ("f1", "accuracy", "precision", "recall")

    Returns:
        Dictionary with optimal threshold and metrics
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    all_scores = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        score = metrics.get(metric, 0)
        all_scores.append({"threshold": thresh, metric: score})

        if score > best_score:
            best_score = score
            best_threshold = thresh

    # Get metrics at optimal threshold
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    optimal_metrics = compute_classification_metrics(y_true, y_pred_optimal, y_proba)

    # Get metrics at default threshold (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)
    default_metrics = compute_classification_metrics(y_true, y_pred_default, y_proba)

    return {
        "optimal_threshold": float(best_threshold),
        f"optimal_{metric}": float(best_score),
        "default_threshold": 0.5,
        f"default_{metric}": default_metrics.get(metric, 0),
        "improvement": float(best_score - default_metrics.get(metric, 0)),
        "improvement_pct": float((best_score - default_metrics.get(metric, 0)) /
                                  max(default_metrics.get(metric, 0), 0.001) * 100),
        "all_scores": all_scores,
        "optimal_metrics": optimal_metrics,
        "default_metrics": default_metrics,
    }


def plot_calibration_curves(
    y_true: np.ndarray,
    y_proba_dict: Dict[str, np.ndarray],
    save_path: Path,
    title: str = "Calibration Curves",
) -> None:
    """
    Plot calibration curves for different methods.

    Args:
        y_true: Ground truth labels
        y_proba_dict: Dictionary mapping method name to probabilities
        save_path: Path to save the plot
        title: Plot title
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration curve
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)

    colors = {"uncalibrated": "gray", "sigmoid": "blue", "isotonic": "green"}

    for method, y_proba in y_proba_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        color = colors.get(method, "black")
        ax1.plot(
            prob_pred, prob_true,
            marker="o", linewidth=2, markersize=8,
            color=color, label=method
        )

    ax1.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax1.set_ylabel("Fraction of Positives", fontsize=11)
    ax1.set_title("Calibration Curves", fontsize=13)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Histogram of probabilities
    ax2 = axes[1]
    for method, y_proba in y_proba_dict.items():
        color = colors.get(method, "black")
        ax2.hist(
            y_proba, bins=20, alpha=0.5,
            label=method, color=color, edgecolor="black"
        )

    ax2.set_xlabel("Predicted Probability", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Distribution of Predicted Probabilities", fontsize=13)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_threshold_analysis(
    threshold_results: Dict,
    save_path: Path,
) -> None:
    """
    Plot threshold analysis results.

    Args:
        threshold_results: Dictionary with threshold analysis results
        save_path: Path to save the plot
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    all_scores = threshold_results["all_scores"]
    metric = list(all_scores[0].keys())[1]  # Get metric name

    thresholds = [s["threshold"] for s in all_scores]
    scores = [s[metric] for s in all_scores]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Threshold vs metric
    ax1 = axes[0]
    ax1.plot(thresholds, scores, linewidth=2, marker="o", color="#2196F3")
    ax1.axvline(
        threshold_results["optimal_threshold"],
        color="#4CAF50", linestyle="--", linewidth=2,
        label=f"Optimal: {threshold_results['optimal_threshold']:.2f}"
    )
    ax1.axvline(
        0.5, color="#F44336", linestyle="--", linewidth=2,
        label="Default: 0.50"
    )
    ax1.set_xlabel("Probability Threshold", fontsize=11)
    ax1.set_ylabel(metric.upper(), fontsize=11)
    ax1.set_title(f"Threshold vs {metric.upper()}", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Metric comparison
    ax2 = axes[1]
    metrics_to_show = ["f1", "accuracy", "precision", "recall"]
    optimal_vals = [threshold_results["optimal_metrics"].get(m, 0) for m in metrics_to_show]
    default_vals = [threshold_results["default_metrics"].get(m, 0) for m in metrics_to_show]

    x = np.arange(len(metrics_to_show))
    width = 0.35

    ax2.bar(x - width/2, optimal_vals, width, label="Optimal Threshold", color="#4CAF50")
    ax2.bar(x + width/2, default_vals, width, label="Default (0.5)", color="#F44336")

    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Metric Comparison: Optimal vs Default Threshold", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_to_show, fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Probability calibration")
    parser.add_argument(
        "--features-dir", type=Path,
        default=Path("data/intermediate/features")
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("artifacts/models")
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

    # 1. Compare calibration methods
    logger.info("\n=== Comparing Calibration Methods ===")
    calibration_results = compare_calibration_methods(
        model, X_train, y_train, cv=args.cv, seed=args.seed
    )

    logger.info("Brier scores:")
    for method in ["uncalibrated", "sigmoid", "isotonic"]:
        logger.info("  %s: %.4f", method, calibration_results[method]["brier_score"])
    logger.info("Best method: %s", calibration_results["best_method"])

    # 2. Train calibrated model with best method
    logger.info("\n=== Training Calibrated Model ===")
    best_method = calibration_results["best_method"]
    if best_method != "uncalibrated":
        calibrated_model, cal_results = calibrate_model(
            model, X_train, y_train,
            method=best_method,
            cv=args.cv,
            seed=args.seed,
        )

        # Evaluate on test set
        y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
        y_pred_cal = calibrated_model.predict(X_test)

        test_metrics_cal = compute_classification_metrics(y_test, y_pred_cal, y_proba_cal)
        logger.info("Calibrated model test F1: %.4f", test_metrics_cal["f1"])
    else:
        model.fit(X_train, y_train)
        y_proba_cal = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        test_metrics_cal = None

    # 3. Find optimal threshold
    logger.info("\n=== Finding Optimal Threshold ===")
    threshold_results = find_optimal_threshold(y_test, y_proba_cal, metric="f1")

    logger.info("Optimal threshold: %.2f (default: 0.50)", threshold_results["optimal_threshold"])
    logger.info("F1 improvement: %.4f (%.1f%%)",
                threshold_results["improvement"],
                threshold_results["improvement_pct"])

    # 4. Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model_name,
        "calibration": calibration_results,
        "threshold_analysis": threshold_results,
        "test_metrics_calibrated": test_metrics_cal,
    }
    save_json(results, args.output_dir / "calibration_results.json")

    # Save calibrated model
    if best_method != "uncalibrated":
        args.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated_model, args.model_dir / "calibrated_model.joblib")
        logger.info("Calibrated model saved to %s", args.model_dir)

    # 5. Generate plots
    plots_dir = args.output_dir / "plots" / "calibration"

    # Calibration curves
    y_proba_dict = {"uncalibrated": model.predict_proba(X_test)[:, 1]}
    if best_method != "uncalibrated":
        y_proba_dict[best_method] = y_proba_cal

    plot_calibration_curves(
        y_test, y_proba_dict,
        plots_dir / "calibration_curves.png",
        title=f"Calibration Curves - {args.model_name}",
    )

    # Threshold analysis
    plot_threshold_analysis(
        threshold_results,
        plots_dir / "threshold_analysis.png",
    )

    logger.info("Results saved to %s", args.output_dir)
    logger.info("Plots saved to %s", plots_dir)

    print("\n=== Calibration Summary ===")
    print(f"Model: {args.model_name}")
    print(f"Best calibration method: {best_method}")
    print(f"Best Brier score: {calibration_results['best_brier_score']:.4f}")
    print(f"\nOptimal threshold: {threshold_results['optimal_threshold']:.2f}")
    print(f"F1 at optimal: {threshold_results['optimal_f1']:.4f}")
    print(f"F1 at default: {threshold_results['default_f1']:.4f}")
    print(f"Improvement: {threshold_results['improvement']:.4f} ({threshold_results['improvement_pct']:.1f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
