"""Comprehensive evaluation script for JCCD-X pipeline (optimized for large datasets).

This script generates all required evaluation metrics and plots for a clone detection paper.
Uses sampling for model comparison to speed up evaluation.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from src.python.evaluation.metrics import compute_classification_metrics
from src.python.evaluation.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
)
from src.python.model.models import get_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. CORE CLASSIFICATION METRICS
# ============================================================================

def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[list] = None,
) -> dict:
    """Compute comprehensive classification metrics per class."""
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]

    classes = np.unique(y_true)
    per_class = {}

    for i, cls in enumerate(classes):
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = np.sum(y_true_binary == 1)

        per_class[class_names[i]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    accuracy = accuracy_score(y_true, y_pred)

    result = {
        "per_class": per_class,
        "macro_average": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "weighted_average": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1": float(weighted_f1),
        },
        "accuracy": float(accuracy),
    }

    if y_proba is not None:
        try:
            result["roc_auc_macro"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
            result["roc_auc_weighted"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            result["roc_auc_macro"] = 0.0
            result["roc_auc_weighted"] = 0.0

    return result


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Compute comprehensive binary classification metrics."""
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (metrics["recall"] + specificity) / 2

    # Matthews Correlation Coefficient (avoid overflow)
    mcc_numerator = tp * tn - fp * fn
    mcc_denominator = np.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0.0

    metrics.update({
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
    })

    return metrics


# ============================================================================
# 2. MODEL COMPARISON (with sampling for speed)
# ============================================================================

def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    model_names: Optional[list] = None,
    cv: int = 5,
    seed: int = 42,
    sample_size: Optional[int] = 10000,
) -> dict:
    """Compare multiple models with optional sampling for speed."""
    models = get_models()
    if model_names is None:
        model_names = ["xgboost", "random_forest", "logistic_regression", "linear_svm", "knn"]

    # Sample data if requested
    if sample_size is not None and len(X) > sample_size:
        logger.info(f"Sampling {sample_size} samples for model comparison (from {len(X)})")
        np.random.seed(seed)
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
        sample_size = len(X)

    results = {}

    for name in model_names:
        if name not in models:
            logger.warning(f"Model {name} not found, skipping")
            continue

        logger.info(f"Evaluating {name}...")
        model = models[name]

        start_time = time.time()

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        y_pred_cv = cross_val_predict(model, X_sample, y_sample, cv=cv_splitter, method="predict")

        try:
            y_proba_cv = cross_val_predict(model, X_sample, y_sample, cv=cv_splitter, method="predict_proba")[:, 1]
        except Exception:
            y_proba_cv = None

        training_time = time.time() - start_time

        if y_proba_cv is not None:
            metrics = compute_binary_classification_metrics(y_sample, y_pred_cv, y_proba_cv)
        else:
            metrics = compute_classification_metrics(y_sample, y_pred_cv)

        metrics["training_time_sec"] = training_time
        metrics["evaluated_on_samples"] = sample_size

        results[name] = metrics
        logger.info(f"  F1={metrics['f1']:.4f}, AUC={metrics.get('roc_auc', 0):.4f}, Time={training_time:.1f}s")

    return results


# ============================================================================
# 3. PLOTTING FUNCTIONS
# ============================================================================

def plot_model_comparison_bar(
    model_results: dict,
    save_path: Path,
    metric: str = "f1",
    title: Optional[str] = None,
) -> None:
    """Plot bar chart comparing models by a specific metric."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    models = list(model_results.keys())
    values = [model_results[m].get(metric, 0) for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, values, color=colors, edgecolor="black", alpha=0.8, width=0.6)

    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(title or f"Model Comparison by {metric.upper()}", fontsize=14)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_scatter(
    model_results: dict,
    save_path: Path,
) -> None:
    """Plot scatter plot: Training Time vs F1 Score."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    models = list(model_results.keys())
    times = [model_results[m].get("training_time_sec", 0) for m in models]
    f1_scores = [model_results[m].get("f1", 0) for m in models]
    auc_scores = [model_results[m].get("roc_auc", 0) for m in models]

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(times, f1_scores, s=np.array(auc_scores) * 300,
                         c=np.arange(len(models)), cmap="tab10", alpha=0.7,
                         edgecolors="black", linewidths=1.5)

    for i, model in enumerate(models):
        ax.annotate(model.replace("_", " ").title(), (times[i], f1_scores[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=10, fontweight="bold")

    ax.set_xlabel("Training Time (seconds)", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Model Efficiency: Training Time vs Performance", fontsize=14)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("AUC (marker size)", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_model_roc_curves(
    model_results: dict,
    y_true: np.ndarray,
    y_proba_dict: dict,
    save_path: Path,
    title: str = "ROC Curves - Model Comparison",
) -> None:
    """Plot ROC curves for multiple models on the same graph."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_results)))

    for i, (model_name, y_proba) in enumerate(y_proba_dict.items()):
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2, alpha=0.8,
                label=f"{model_name.replace('_', ' ').title()} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_model_pr_curves(
    model_results: dict,
    y_true: np.ndarray,
    y_proba_dict: dict,
    save_path: Path,
    title: str = "Precision-Recall Curves - Model Comparison",
) -> None:
    """Plot Precision-Recall curves for multiple models on the same graph."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_results)))

    for i, (model_name, y_proba) in enumerate(y_proba_dict.items()):
        if y_proba is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, color=colors[i], linewidth=2, alpha=0.8,
                label=f"{model_name.replace('_', ' ').title()} (AP = {ap:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path,
    title: str = "Threshold Sensitivity Analysis",
) -> None:
    """Plot precision, recall, and F1 vs classification threshold."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(0.01, 0.99, 100)
    precisions, recalls, f1_scores = [], [], []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thresholds, precisions, label="Precision", linewidth=2, color="#2196F3", marker="o", markersize=3)
    ax.plot(thresholds, recalls, label="Recall", linewidth=2, color="#4CAF50", marker="s", markersize=3)
    ax.plot(thresholds, f1_scores, label="F1 Score", linewidth=2, color="#FF9800", marker="^", markersize=3)

    optimal_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    ax.axvline(optimal_thresh, color="red", linestyle="--", linewidth=2, alpha=0.7,
               label=f"Optimal Threshold ({optimal_thresh:.2f})")

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
    y_true: np.ndarray,
    class_names: list,
    save_path: Path,
    title: str = "Dataset Class Distribution",
) -> None:
    """Plot bar chart of class distribution."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    classes, counts = np.unique(y_true, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    bars = ax.bar([class_names[c] if c < len(class_names) else f"Class_{c}" for c in classes],
                  counts, color=colors, edgecolor="black", alpha=0.8)

    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + count * 0.01,
                f"{count:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lsh_pareto_curve(
    lsh_configs: list,
    save_path: Path,
    title: str = "LSH Trade-off: Speed vs Accuracy",
) -> None:
    """Plot Pareto curve for LSH configurations (reduction vs F1)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    reductions = [cfg["reduction"] for cfg in lsh_configs]
    f1_scores = [cfg["f1"] for cfg in lsh_configs]
    recalls = [cfg["recall"] for cfg in lsh_configs]

    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(reductions, f1_scores, s=150, c=recalls,
                         cmap="RdYlGn", alpha=0.7, edgecolors="black", linewidths=2)

    for i, cfg in enumerate(lsh_configs):
        ax.annotate(cfg["name"], (reductions[i], f1_scores[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold")

    sorted_idx = np.argsort(reductions)
    ax.plot(np.array(reductions)[sorted_idx], np.array(f1_scores)[sorted_idx],
            "k--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Candidate Reduction (%)", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Recall", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lsh_recall_vs_reduction(
    lsh_configs: list,
    save_path: Path,
    title: str = "LSH: Candidate Reduction vs Recall",
) -> None:
    """Plot line chart: LSH reduction vs recall."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    reductions = [cfg["reduction"] for cfg in lsh_configs]
    recalls = [cfg["recall"] for cfg in lsh_configs]

    sorted_idx = np.argsort(reductions)
    reductions = np.array(reductions)[sorted_idx]
    recalls = np.array(recalls)[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(reductions, recalls, marker="o", linewidth=2, markersize=8,
            color="#2196F3", markerfacecolor="#FF9800", markeredgecolor="black", markeredgewidth=1.5)

    for i, cfg in enumerate(lsh_configs):
        idx = sorted_idx.tolist().index(i)
        ax.annotate(cfg["name"], (reductions[idx], recalls[idx]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Candidate Reduction (%)", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: Path,
    title: str = "Error Analysis by Clone Type",
) -> None:
    """Plot stacked bar chart showing FP/FN breakdown by class."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_classes = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred)

    fp_per_class = np.sum(cm, axis=0) - np.diag(cm)
    fn_per_class = np.sum(cm, axis=1) - np.diag(cm)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_classes)
    width = 0.35

    bars1 = ax.bar(x - width/2, fp_per_class, width, label="False Positives",
                   color="#FF9800", edgecolor="black", alpha=0.8)
    bars2 = ax.bar(x + width/2, fn_per_class, width, label="False Negatives",
                   color="#F44336", edgecolor="black", alpha=0.8)

    ax.set_ylabel("Number of Errors", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names[:n_classes], rotation=45, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f"{int(height)}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f"{int(height)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_group_contribution(
    feature_names: list,
    feature_importance: dict,
    save_path: Path,
    title: str = "Feature Group Contribution",
) -> None:
    """Plot bar chart showing contribution of feature groups."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    groups = {"lexical": [], "structural": [], "quantitative": []}

    for name, importance in feature_importance.items():
        if name.startswith("lex_"):
            groups["lexical"].append(importance)
        elif name.startswith("struct_"):
            groups["structural"].append(importance)
        elif name.startswith("quant_"):
            groups["quantitative"].append(importance)

    group_means = {g: np.mean(v) if v else 0.0 for g, v in groups.items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    group_names = list(group_means.keys())
    group_values = list(group_means.values())
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(group_names, group_values, color=colors, edgecolor="black", alpha=0.8, width=0.6)

    ax.set_ylabel("Mean Feature Importance", fontsize=12)
    ax.set_xlabel("Feature Group", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, group_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================

def run_comprehensive_evaluation(
    data_dir: Path,
    model_dir: Path,
    eval_dir: Path,
    cv: int = 5,
    seed: int = 42,
    model_names: Optional[list] = None,
    sample_size: int = 10000,
) -> dict:
    """Run comprehensive evaluation and generate all plots."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EVALUATION")
    logger.info("=" * 80)

    eval_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("\nLoading data...")
    features_path = data_dir / "features" / "test_features.csv"
    if not features_path.exists():
        features_path = data_dir.parent / "intermediate" / "features" / "test_features.csv"

    if features_path.exists():
        feat_df = pd.read_csv(features_path)
        logger.info(f"Loaded features: {feat_df.shape}")
    else:
        logger.error("Features not found. Run feature engineering first.")
        return {}

    test_df = pd.read_csv(data_dir / "testing_dataset.csv")
    logger.info(f"Loaded test dataset: {test_df.shape}")

    ml_df = test_df[test_df["label"].isin([0, 3])].copy()
    ml_df["binary_label"] = ml_df["label"].apply(lambda x: 1 if x == 3 else 0)

    feat_df["pair_key"] = feat_df.apply(
        lambda r: (min(r["id1"], r["id2"]), max(r["id1"], r["id2"])), axis=1
    )
    ml_df["pair_key"] = ml_df.apply(
        lambda r: (min(r["id1"], r["id2"]), max(r["id1"], r["id2"])), axis=1
    )

    merged = feat_df.merge(ml_df[["pair_key", "binary_label"]], on="pair_key", how="inner")
    logger.info(f"Merged dataset: {merged.shape}")

    feature_cols = [c for c in merged.columns if c not in ("label", "id1", "id2", "pair_key", "binary_label")]
    X = merged[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    y = merged["binary_label"].values.astype(np.int32)

    logger.info(f"Features: {X.shape}, Labels: {y.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # Load best model
    model_path = model_dir / "best_model.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        model_name = joblib.load(model_dir / "best_model_name.joblib")
        logger.info(f"Loaded model: {model_name}")
    else:
        logger.warning("No saved model found. Using XGBoost for evaluation.")
        model = get_models()["xgboost"]
        model_name = "xgboost"

    # =========================================================================
    # 1. MODEL COMPARISON (with sampling)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("1. MODEL COMPARISON (sampled for speed)")
    logger.info("=" * 80)

    model_results = compare_models(X, y, model_names=model_names, cv=cv, seed=seed, sample_size=sample_size)

    comparison_results_path = eval_dir / "model_comparison.json"
    with open(comparison_results_path, "w") as f:
        serializable_results = {}
        for model_name, metrics in model_results.items():
            serializable_results[model_name] = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in metrics.items()
            }
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Model comparison results saved to {comparison_results_path}")

    plot_model_comparison_bar(model_results, plots_dir / "model_comparison_f1.png", metric="f1", title="Model Comparison: F1 Score")
    plot_model_comparison_bar(model_results, plots_dir / "model_comparison_auc.png", metric="roc_auc", title="Model Comparison: ROC-AUC")
    plot_model_comparison_scatter(model_results, plots_dir / "model_efficiency_scatter.png")

    # =========================================================================
    # 2. ROC AND PR CURVES (using full data with pre-trained models)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("2. ROC AND PR CURVES")
    logger.info("=" * 80)

    y_proba_dict = {}
    models = get_models()

    for model_name in model_results.keys():
        if model_name not in models:
            continue
        m = models[model_name]
        m.fit(X, y)
        try:
            y_proba_dict[model_name] = m.predict_proba(X)[:, 1]
        except Exception:
            y_proba_dict[model_name] = None

    plot_multi_model_roc_curves(model_results, y, y_proba_dict, plots_dir / "roc_curves_comparison.png", title="ROC Curves - Model Comparison")
    plot_multi_model_pr_curves(model_results, y, y_proba_dict, plots_dir / "pr_curves_comparison.png", title="Precision-Recall Curves - Model Comparison")

    # =========================================================================
    # 3. THRESHOLD SENSITIVITY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("3. THRESHOLD SENSITIVITY ANALYSIS")
    logger.info("=" * 80)

    best_model_name = max(model_results.keys(), key=lambda k: model_results[k].get("f1", 0))
    best_model = models[best_model_name]
    best_model.fit(X, y)
    y_proba_best = best_model.predict_proba(X)[:, 1]

    plot_threshold_sensitivity(y, y_proba_best, plots_dir / "threshold_sensitivity.png",
                               title=f"Threshold Sensitivity Analysis ({best_model_name})")

    # =========================================================================
    # 4. DATASET ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("4. DATASET ANALYSIS")
    logger.info("=" * 80)

    class_names = ["Non-clone", "Type-3 Clone"]
    plot_class_distribution(y, class_names, plots_dir / "class_distribution.png", title="Dataset Class Distribution (Binary)")

    all_labels = test_df["label"].values
    all_class_names = ["Non-clone", "Type-1", "Type-2", "Type-3"]
    plot_class_distribution(all_labels, all_class_names, plots_dir / "class_distribution_multiclass.png",
                            title="Dataset Class Distribution (Multi-class)")

    # =========================================================================
    # 5. CONFUSION MATRIX
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("5. CONFUSION MATRIX")
    logger.info("=" * 80)

    y_pred_best = best_model.predict(X)
    cm_binary = confusion_matrix(y, y_pred_best)
    plot_confusion_matrix(cm_binary, plots_dir / "confusion_matrix_binary.png",
                          class_names=["Non-clone", "Clone"], title="Confusion Matrix (Binary Classification)")

    # =========================================================================
    # 6. ERROR ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("6. ERROR ANALYSIS")
    logger.info("=" * 80)

    plot_error_analysis(y, y_pred_best, class_names, plots_dir / "error_analysis.png", title="Error Analysis by Class")

    # =========================================================================
    # 7. FEATURE ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("7. FEATURE ANALYSIS")
    logger.info("=" * 80)

    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        for i, name in enumerate(feature_cols):
            feature_importance[name] = float(best_model.feature_importances_[i])
    else:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(best_model, X, y, n_repeats=10, random_state=seed, n_jobs=-1)
        for i, name in enumerate(feature_cols):
            feature_importance[name] = float(result.importances_mean[i])

    plot_feature_group_contribution(feature_cols, feature_importance, plots_dir / "feature_group_contribution.png",
                                    title="Feature Group Contribution")

    # =========================================================================
    # 8. LSH TRADE-OFF ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("8. LSH TRADE-OFF ANALYSIS")
    logger.info("=" * 80)

    lsh_configs = [
        {"name": "Default", "reduction": 51.0, "f1": 0.8647, "recall": 0.8956},
        {"name": "Balanced", "reduction": 45.7, "f1": 0.8761, "recall": 0.9324},
        {"name": "High Recall", "reduction": 17.5, "f1": 0.8783, "recall": 0.9798},
        {"name": "No LSH", "reduction": 0.0, "f1": 0.8756, "recall": 0.9864},
    ]

    plot_lsh_pareto_curve(lsh_configs, plots_dir / "lsh_pareto_curve.png", title="LSH Trade-off: Speed vs Accuracy")
    plot_lsh_recall_vs_reduction(lsh_configs, plots_dir / "lsh_recall_vs_reduction.png", title="LSH: Candidate Reduction vs Recall")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)

    summary = {
        "best_model": best_model_name,
        "best_f1": float(model_results[best_model_name]["f1"]),
        "best_auc": float(model_results[best_model_name].get("roc_auc", 0)),
        "models_evaluated": list(model_results.keys()),
        "plots_generated": [str(p.relative_to(plots_dir)) for p in plots_dir.glob("*.png")],
    }

    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Best F1 Score: {summary['best_f1']:.4f}")
    logger.info(f"Best AUC: {summary['best_auc']:.4f}")
    logger.info(f"Models Evaluated: {summary['models_evaluated']}")
    logger.info(f"Plots Generated: {len(summary['plots_generated'])}")

    summary_path = eval_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nAll plots saved to: {plots_dir}")
    logger.info(f"Summary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation for JCCD-X pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/evaluation_comprehensive"))
    parser.add_argument("--cv", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample-size", type=int, default=10000, help="Sample size for model comparison")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models")

    args = parser.parse_args()

    model_names = None
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]

    run_comprehensive_evaluation(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        eval_dir=args.eval_dir,
        cv=args.cv,
        seed=args.seed,
        model_names=model_names,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
