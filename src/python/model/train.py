"""ML model training with 5-fold stratified cross-validation."""

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.python.model.models import get_models
from src.python.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(y_true, y_pred, y_proba=None) -> dict:
    """Compute metrics for a single fold."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = 0.0
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            metrics["pr_auc"] = 0.0
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["tp"] = int(tp)
        metrics["fp"] = int(fp)
        metrics["tn"] = int(tn)
        metrics["fn"] = int(fn)
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    return metrics


def cross_validate_model(model, X, y, cv=5, seed=42) -> dict:
    """Run stratified k-fold CV and return per-fold + aggregate metrics."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Get probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_val)

        fold_metrics = evaluate_fold(y_val, y_pred, y_proba)
        fold_metrics["fold"] = fold_idx
        fold_results.append(fold_metrics)

    # Aggregate
    metric_keys = [k for k in fold_results[0] if k != "fold"]
    aggregate = {}
    for key in metric_keys:
        values = [f[key] for f in fold_results]
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))

    return {"folds": fold_results, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: np.ndarray, y: np.ndarray, feature_names: list[str],
    output_dir: Path, eval_dir: Path, seed: int = 42,
) -> tuple:
    """Train all models, evaluate with CV, select best."""
    models = get_models()
    all_results = {}
    best_f1 = -1
    best_model_name = None
    best_model = None

    for name, model in models.items():
        logger.info("Training %s...", name)
        start = time.time()
        result = cross_validate_model(model, X, y, cv=5, seed=seed)
        elapsed = time.time() - start

        agg = result["aggregate"]
        logger.info(
            "  %s: F1=%.4f±%.4f  Acc=%.4f±%.4f  AUC=%.4f±%.4f  (%.1fs)",
            name,
            agg.get("f1_mean", 0), agg.get("f1_std", 0),
            agg.get("accuracy_mean", 0), agg.get("accuracy_std", 0),
            agg.get("roc_auc_mean", 0), agg.get("roc_auc_std", 0),
            elapsed,
        )

        result["elapsed_sec"] = elapsed
        all_results[name] = result

        # Select best by F1 mean (with preference for lower variance)
        f1_mean = agg.get("f1_mean", 0)
        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_model_name = name
            best_model = model

    logger.info("Best model: %s (F1=%.4f)", best_model_name, best_f1)

    # Retrain best model on full training data
    best_model.fit(X, y)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_dir / "best_model.joblib")
    joblib.dump(best_model_name, output_dir / "best_model_name.joblib")
    logger.info("Saved best model to %s", output_dir)

    # Feature importance (if available)
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance = dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True,
        ))
        logger.info("Top 5 features: %s",
                     list(feature_importance.items())[:5])

    # Save results
    eval_dir.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for name, res in all_results.items():
        serializable[name] = {
            "folds": res["folds"],
            "aggregate": res["aggregate"],
            "elapsed_sec": res["elapsed_sec"],
        }
    serializable["best_model"] = best_model_name
    serializable["best_f1"] = best_f1
    serializable["feature_importance"] = {k: float(v) for k, v in feature_importance.items()}
    save_json(serializable, eval_dir / "cv_results.json")

    return all_results, best_model_name, best_model, feature_importance


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_training_plots(
    all_results: dict, feature_importance: dict,
    best_model, best_model_name: str,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: list[str],
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ROC curves per model
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in all_results.items():
        aucs = [f.get("roc_auc", 0) for f in res["folds"]]
        ax.plot(range(1, len(aucs) + 1), aucs, "o-", label=f"{name} (mean={np.mean(aucs):.3f})")
    ax.set_xlabel("Fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC per Fold by Model")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    fig.savefig(plots_dir / "roc_auc_per_fold.png", dpi=150)
    plt.close(fig)

    # CV metric variance box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["f1", "accuracy", "roc_auc"]):
        data = []
        labels = []
        for name, res in all_results.items():
            vals = [f.get(metric, 0) for f in res["folds"]]
            data.append(vals)
            labels.append(name)
        ax.boxplot(data, tick_labels=labels)
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(plots_dir / "cv_metric_variance.png", dpi=150)
    plt.close(fig)

    # Feature importance bar chart
    if feature_importance:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(feature_importance.keys())[:15]
        values = [feature_importance[n] for n in names]
        ax.barh(range(len(names)), values)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance ({best_model_name})")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(plots_dir / "feature_importance.png", dpi=150)
        plt.close(fig)

    # Confusion matrix on test set
    if best_model is not None and len(X_test) > 0:
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["NON", "Type-3"])
        ax.set_yticklabels(["NON", "Type-3"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix ({best_model_name})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(plots_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

        # Predicted probability histogram
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_test)[:, 1]
            fig, ax = plt.subplots(figsize=(8, 5))
            d0 = y_proba[y_test == 0]
            d1 = y_proba[y_test == 1]
            try:
                ax.hist(d0[np.isfinite(d0)], bins=50, alpha=0.6, label="NON", edgecolor="black")
                ax.hist(d1[np.isfinite(d1)], bins=50, alpha=0.6, label="Type-3", edgecolor="black")
            except (ValueError, TypeError):
                pass
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Frequency")
            ax.set_title("Predicted Probabilities by Class")
            ax.legend()
            plt.tight_layout()
            fig.savefig(plots_dir / "probability_histogram.png", dpi=150)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ML model training")
    parser.add_argument("--features-dir", type=Path, default=Path("data/intermediate/features"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load features
    train_path = args.features_dir / "train_features.csv"
    test_path = args.features_dir / "test_features.csv"
    if not train_path.exists():
        logger.error("Run feature engineering first")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in ("label", "id1", "id2")]
    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df["label"].values.astype(np.int32)
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df["label"].values.astype(np.int32)

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=0.0)

    logger.info("Training data: %d samples, %d features", X_train.shape[0], X_train.shape[1])
    logger.info("Testing data: %d samples", X_test.shape[0])

    # Train and evaluate
    all_results, best_name, best_model, feature_importance = train_and_evaluate(
        X_train, y_train, feature_cols, args.model_dir, args.eval_dir, args.seed,
    )

    # Test set evaluation (only on binary-labeled samples)
    logger.info("Evaluating on test set...")
    binary_mask = (y_test >= 0)
    X_test_bin = X_test[binary_mask]
    y_test_bin = y_test[binary_mask]
    y_pred = best_model.predict(X_test_bin)
    y_proba = None
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test_bin)[:, 1]
    test_metrics = evaluate_fold(y_test_bin, y_pred, y_proba)
    test_metrics["total_test_samples"] = int(len(y_test))
    test_metrics["binary_test_samples"] = int(binary_mask.sum())
    save_json(test_metrics, args.eval_dir / "test_metrics.json")
    logger.info("Test metrics: %s", {k: f"{v:.4f}" if isinstance(v, float) else v
                                       for k, v in test_metrics.items()})

    # Plots
    plots_dir = args.eval_dir / "plots" / "training"
    generate_training_plots(
        all_results, feature_importance, best_model, best_name,
        X_test_bin, y_test_bin, feature_cols, plots_dir,
    )
    logger.info("Plots saved to %s", plots_dir)

    print("\n=== Training Summary ===")
    print(f"Best model: {best_name}")
    print(f"CV F1: {all_results[best_name]['aggregate']['f1_mean']:.4f} "
          f"± {all_results[best_name]['aggregate']['f1_std']:.4f}")
    print(f"Test F1: {test_metrics.get('f1', 0):.4f}")
    print(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
