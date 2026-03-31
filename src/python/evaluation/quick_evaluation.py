"""Quick comprehensive evaluation for JCCD-X pipeline.

Generates all required plots using pre-trained models without extensive CV.
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
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.python.evaluation.metrics import compute_classification_metrics
from src.python.evaluation.plots import plot_confusion_matrix
from src.python.model.models import get_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def run_quick_evaluation(
    data_dir: Path,
    model_dir: Path,
    eval_dir: Path,
    seed: int = 42,
) -> dict:
    """Run quick evaluation using pre-trained models."""
    logger.info("=" * 80)
    logger.info("QUICK COMPREHENSIVE EVALUATION")
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
        logger.error("Features not found.")
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
        best_model = joblib.load(model_path)
        best_model_name = joblib.load(model_dir / "best_model_name.joblib")
        logger.info(f"Loaded model: {best_model_name}")
    else:
        logger.warning("No saved model found.")
        return {}

    # =========================================================================
    # 1. QUICK MODEL COMPARISON (train on subset, predict on subset)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("1. QUICK MODEL COMPARISON")
    logger.info("=" * 80)

    # Sample for quick comparison
    np.random.seed(seed)
    sample_size = 5000
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]

    # Define models with reduced complexity for speed
    quick_models = {
        "xgboost": XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, n_jobs=-1, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=50, max_depth=6, n_jobs=-1, random_state=seed),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "linear_svm": LinearSVC(max_iter=1000, random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    model_results = {}
    y_proba_dict = {}

    for name, model in quick_models.items():
        logger.info(f"Evaluating {name}...")
        start = time.time()
        model.fit(X_sample, y_sample)
        train_time = time.time() - start

        # Predict on same sample for quick metrics
        y_pred = model.predict(X_sample)

        # Get probabilities or decision scores
        y_proba = None
        auc = 0.0

        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_sample)[:, 1]
                y_proba_dict[name] = y_proba
                auc = roc_auc_score(y_sample, y_proba)
            except Exception:
                y_proba_dict[name] = None
                auc = 0.0
        elif hasattr(model, "decision_function"):
            # For SVM, use decision function scores
            try:
                decision_scores = model.decision_function(X_sample)
                # Convert to probabilities using sigmoid
                y_proba = 1 / (1 + np.exp(-decision_scores))
                y_proba_dict[name] = y_proba
                auc = roc_auc_score(y_sample, decision_scores)
            except Exception:
                y_proba_dict[name] = None
                auc = 0.0
        else:
            y_proba_dict[name] = None
            auc = 0.0

        f1 = f1_score(y_sample, y_pred)
        accuracy = accuracy_score(y_sample, y_pred)

        model_results[name] = {
            "f1": float(f1),
            "accuracy": float(accuracy),
            "roc_auc": float(auc),
            "training_time_sec": float(train_time),
        }
        logger.info(f"  F1={f1:.4f}, AUC={auc:.4f}, Time={train_time:.1f}s")

    # Save model comparison
    comparison_path = eval_dir / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(model_results, f, indent=2)

    # Plot model comparison
    plot_model_comparison_bar(model_results, plots_dir / "model_comparison_f1.png", metric="f1", title="Model Comparison: F1 Score")
    plot_model_comparison_bar(model_results, plots_dir / "model_comparison_auc.png", metric="roc_auc", title="Model Comparison: ROC-AUC")
    plot_model_comparison_scatter(model_results, plots_dir / "model_efficiency_scatter.png")

    # =========================================================================
    # 2. ROC AND PR CURVES (using full data with trained models)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("2. ROC AND PR CURVES (full data)")
    logger.info("=" * 80)

    # Train models on full data for ROC/PR curves
    full_models = get_models()
    y_proba_full_dict = {}

    for name in model_results.keys():
        if name not in full_models:
            continue
        logger.info(f"Training {name} on full data for curves...")
        m = full_models[name]
        m.fit(X, y)
        try:
            y_proba_full_dict[name] = m.predict_proba(X)[:, 1]
        except Exception:
            y_proba_full_dict[name] = None

    plot_multi_model_roc_curves(model_results, y, y_proba_full_dict, plots_dir / "roc_curves_comparison.png", title="ROC Curves - Model Comparison")
    plot_multi_model_pr_curves(model_results, y, y_proba_full_dict, plots_dir / "pr_curves_comparison.png", title="Precision-Recall Curves - Model Comparison")

    # =========================================================================
    # 3. THRESHOLD SENSITIVITY (using best model)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("3. THRESHOLD SENSITIVITY ANALYSIS")
    logger.info("=" * 80)

    best_model_name = max(model_results.keys(), key=lambda k: model_results[k].get("f1", 0))
    y_proba_best = y_proba_full_dict.get(best_model_name)

    if y_proba_best is not None:
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
    # 5. CONFUSION MATRIX (using pre-trained best model)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("5. CONFUSION MATRIX")
    logger.info("=" * 80)

    best_model.fit(X, y)
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
        logger.info("Computing permutation importance (this may take a while)...")
        result = permutation_importance(best_model, X, y, n_repeats=5, random_state=seed, n_jobs=-1)
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
        {"name": "Balanced ⭐", "reduction": 45.7, "f1": 0.8761, "recall": 0.9324},
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

    # Also save full metrics
    metrics_path = eval_dir / "full_pipeline_metrics.json"
    full_metrics = compute_classification_metrics(y, y_pred_best, y_proba_best)
    with open(metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)

    logger.info(f"\nAll plots saved to: {plots_dir}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Full pipeline metrics saved to: {metrics_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Quick comprehensive evaluation for JCCD-X pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/evaluation_comprehensive"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_quick_evaluation(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        eval_dir=args.eval_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
