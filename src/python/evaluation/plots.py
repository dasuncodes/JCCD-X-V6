"""Centralized plotting functions for JCCD-X pipeline."""

from pathlib import Path
from typing import Optional, List, Dict, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
)


def plot_roc_curves(
    fold_results: list,
    save_path: Path,
    title: str = "ROC Curves - Cross-Validation",
    show_mean: bool = True,
) -> None:
    """
    Plot ROC curves for each fold with optional mean curve.

    Args:
        fold_results: List of dicts with 'tpr', 'fpr', 'roc_auc' per fold
        save_path: Path to save the plot
        title: Plot title
        show_mean: Whether to plot mean ROC curve
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if any fold has ROC curve data
    has_curve_data = any("fpr" in fold and "tpr" in fold for fold in fold_results)
    if not has_curve_data:
        # No ROC curve data available, skip plotting
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot individual folds
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(fold_results))))
    for i, fold in enumerate(fold_results):
        if "fpr" in fold and "tpr" in fold:
            auc = fold.get("roc_auc", 0)
            ax.plot(
                fold["fpr"],
                fold["tpr"],
                color=colors[i],
                linewidth=1.5,
                alpha=0.6,
                label=f"Fold {i+1} (AUC = {auc:.3f})" if auc > 0 else f"Fold {i+1}",
            )

    # Plot mean ROC curve if data is available
    if show_mean and fold_results:
        # Interpolate TPR values at common FPR points
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = []

        for fold in fold_results:
            if "fpr" in fold and "tpr" in fold:
                tpr_interp = np.interp(mean_fpr, fold["fpr"], fold["tpr"])
                tpr_interp[0] = 0.0  # Ensure starts at (0,0)
                mean_tpr.append(tpr_interp)

        if mean_tpr:
            mean_tpr = np.mean(mean_tpr, axis=0)
            mean_auc = fold_results[0].get("roc_auc_mean", 0)
            ax.plot(
                mean_fpr,
                mean_tpr,
                color="navy",
                linewidth=2.5,
                linestyle="--",
                label=f"Mean ROC (AUC = {mean_auc:.3f})" if mean_auc > 0 else "Mean ROC",
            )

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, alpha=0.5, label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr_curves(
    fold_results: list,
    save_path: Path,
    title: str = "Precision-Recall Curves - Cross-Validation",
    show_mean: bool = True,
) -> None:
    """
    Plot Precision-Recall curves for each fold with optional mean curve.

    Args:
        fold_results: List of dicts with 'precision', 'recall', 'pr_auc' per fold
        save_path: Path to save the plot
        title: Plot title
        show_mean: Whether to plot mean PR curve
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if any fold has PR curve data
    has_curve_data = any("precision" in fold and "recall" in fold for fold in fold_results)
    if not has_curve_data:
        # No PR curve data available, skip plotting
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot individual folds
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(fold_results))))
    for i, fold in enumerate(fold_results):
        if "precision" in fold and "recall" in fold:
            auc = fold.get("pr_auc", 0)
            ax.plot(
                fold["recall"],
                fold["precision"],
                color=colors[i],
                linewidth=1.5,
                alpha=0.6,
                label=f"Fold {i+1} (AP = {auc:.3f})" if auc > 0 else f"Fold {i+1}",
            )

    # Plot mean PR curve if data is available
    if show_mean and fold_results:
        # Interpolate precision values at common recall points
        mean_recall = np.linspace(0, 1, 100)
        mean_precision = []

        for fold in fold_results:
            if "precision" in fold and "recall" in fold:
                prec_interp = np.interp(mean_recall, fold["recall"], fold["precision"])
                prec_interp = np.flip(np.maximum.accumulate(np.flip(prec_interp)))
                mean_precision.append(prec_interp)

        if mean_precision:
            mean_precision = np.mean(mean_precision, axis=0)
            mean_ap = fold_results[0].get("pr_auc_mean", 0)
            ax.plot(
                mean_recall,
                mean_precision,
                color="navy",
                linewidth=2.5,
                linestyle="--",
                label=f"Mean AP = {mean_ap:.3f}" if mean_ap > 0 else "Mean PR",
            )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_correlation(
    feature_df,
    save_path: Path,
    title: str = "Feature Correlation Matrix",
    figsize: tuple = (14, 12),
) -> None:
    """
    Plot heatmap of feature correlations.

    Args:
        feature_df: DataFrame with features (columns) and samples (rows)
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    feature_cols = [c for c in feature_df.columns if c not in ("label", "id1", "id2")]
    corr = feature_df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Create shorter labels for readability
    short_labels = []
    for col in feature_cols:
        short = col.replace("lex_", "L_").replace("struct_", "S_").replace("quant_", "Q_")
        if len(short) > 15:
            short = short[:12] + "..."
        short_labels.append(short)

    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_title(title, fontsize=14)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation", fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    importance: dict,
    save_path: Path,
    title: str = "Feature Importance",
    top_n: int = 20,
) -> None:
    """
    Plot horizontal bar chart of feature importance.

    Args:
        importance: Dictionary mapping feature names to importance scores
        save_path: Path to save the plot
        title: Plot title
        top_n: Number of top features to display
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="#2196F3", edgecolor="black", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.invert_yaxis()  # Highest importance at top

    # Add value labels
    for i, v in enumerate(values):
        ax.text(v * 1.01, i, f"{v:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    class_names: list = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> None:
    """
    Plot confusion matrix heatmap for any number of classes.

    Args:
        cm: Confusion matrix array (n_classes x n_classes)
        save_path: Path to save the plot
        class_names: Names of classes (default: ["Class 0", "Class 1", ...])
        title: Plot title
        cmap: Colormap name
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if cm.size == 0:
        # Empty confusion matrix, nothing to plot
        return

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    assert len(class_names) == n_classes

    fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes * 0.8)))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(title, fontsize=13)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12 if n_classes <= 5 else 9,
            )

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cv_variance(
    fold_results: list,
    save_path: Path,
    metrics: list = None,
    title: str = "Cross-Validation Metric Variance",
) -> None:
    """
    Plot box plots showing variance of metrics across CV folds.

    Args:
        fold_results: List of dicts with metric values per fold
        save_path: Path to save the plot
        metrics: List of metric names to plot (default: ["f1", "accuracy", "roc_auc"])
        title: Plot title
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics is None:
        metrics = ["f1", "accuracy", "roc_auc"]

    # Extract metric values across folds
    data = []
    labels = []
    for metric in metrics:
        values = [f.get(metric, 0) for f in fold_results if metric in f]
        if values:
            data.append(values)
            labels.append(
                f"{metric.upper()}\n(mean={np.mean(values):.3f}, std={np.std(values):.3f})"
            )

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, patch_artist=True, labels=labels)

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_distributions(
    feature_df,
    save_path: Path,
    features: list = None,
    title: str = "Feature Distributions by Class",
) -> None:
    """
    Plot box plots showing feature distributions per class.

    Args:
        feature_df: DataFrame with features and label column
        save_path: Path to save the plot
        features: List of feature names to plot (default: first 6 features)
        title: Plot title
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if features is None:
        feature_cols = [c for c in feature_df.columns if c not in ("label", "id1", "id2")]
        features = feature_cols[:6]

    n_features = len(features)
    if n_features == 0:
        return

    # Create subplots
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature in enumerate(features):
        if feature not in feature_df.columns:
            continue

        ax = axes[i]
        pos_data = feature_df.loc[feature_df["label"] == 1, feature].values
        neg_data = feature_df.loc[feature_df["label"] == 0, feature].values

        # Filter out non-finite values
        pos_data = pos_data[np.isfinite(pos_data)]
        neg_data = neg_data[np.isfinite(neg_data)]

        data_to_plot = (
            [neg_data, pos_data] if len(neg_data) > 0 and len(pos_data) > 0 else [pos_data]
        )
        labels_plot = ["NON", "Type-3"] if len(data_to_plot) == 2 else ["Type-3"]

        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels_plot)

        # Color the boxes
        colors = ["#607D8B", "#4CAF50"][: len(data_to_plot)]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Shorten feature name for title
        short_name = feature.replace("lex_", "L_").replace("struct_", "S_").replace("quant_", "Q_")
        if len(short_name) > 20:
            short_name = short_name[:17] + "..."

        ax.set_title(short_name, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_results(
    results: dict,
    save_path: Path,
    metric: str = "f1",
    title: str = "Ablation Study Results",
) -> None:
    """
    Plot bar chart comparing ablation configurations.

    Args:
        results: Dictionary mapping configuration name to metrics dict
        save_path: Path to save the plot
        metric: Metric to plot (default: "f1")
        title: Plot title
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    values = [results[name].get(metric, 0) for name in names]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color: baseline in green, ablated in orange/red
    colors = [
        "#4CAF50" if "full" in name.lower() or "baseline" in name.lower() else "#FF9800"
        for name in names
    ]

    bars = ax.bar(names, values, color=colors, edgecolor="black", alpha=0.8)

    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_lsh_recall_vs_threshold(
    thresholds: list,
    recalls: list,
    precisions: list,
    save_path: Path,
    title: str = "LSH Performance vs Threshold",
) -> None:
    """
    Plot line chart showing LSH recall and precision vs threshold.

    Args:
        thresholds: List of LSH threshold values
        recalls: List of recall values at each threshold
        precisions: List of precision values at each threshold
        save_path: Path to save the plot
        title: Plot title
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Recall on left y-axis
    color1 = "#2196F3"
    ax1.set_xlabel("LSH Threshold", fontsize=11)
    ax1.set_ylabel("Recall", color=color1, fontsize=11)
    ax1.plot(
        thresholds, recalls, color=color1, marker="o", linewidth=2, markersize=8, label="Recall"
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Precision on right y-axis
    ax2 = ax1.twinx()
    color2 = "#FF9800"
    ax2.set_ylabel("Precision", color=color2, fontsize=11)
    ax2.plot(
        thresholds,
        precisions,
        color=color2,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Precision",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_runtime_per_stage(
    stages: list,
    times: list,
    save_path: Path,
    title: str = "Runtime per Pipeline Stage",
) -> None:
    """
    Plot horizontal bar chart of runtime per stage.

    Args:
        stages: List of stage names
        times: List of times in seconds
        save_path: Path to save the plot
        title: Plot title
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not stages or not times or len(stages) != len(times):
        # Nothing to plot
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(stages) * 0.8)))

    y_pos = np.arange(len(stages))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(stages)))

    bars = ax.barh(y_pos, times, color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontsize=10)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title(title, fontsize=13)

    # Add value labels
    for bar, time_val in zip(bars, times):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{time_val:.1f}s",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_memory_usage(
    stages: list,
    memory_mb: list,
    save_path: Path,
    title: str = "Memory Usage per Pipeline Stage",
) -> None:
    """
    Plot bar chart of memory usage per stage.

    Args:
        stages: List of stage names
        memory_mb: List of memory usage in MB
        save_path: Path to save the plot
        title: Plot title
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(stages))
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(stages)))

    bars = ax.bar(x_pos, memory_mb, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stages, rotation=45, fontsize=10)
    ax.set_ylabel("Memory (MB)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, mem_val in zip(bars, memory_mb):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(memory_mb) * 0.01,
            f"{mem_val:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_model_comparison_bar(
    model_results: dict,
    save_path: Path,
    metric: str = "f1",
    title: Optional[str] = None,
) -> None:
    """Plot bar chart comparing models by a specific metric."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    models = list(model_results.keys())
    if not models:
        # No models to compare
        return
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
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

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
    scatter = ax.scatter(
        times,
        f1_scores,
        s=np.array(auc_scores) * 300,
        c=np.arange(len(models)),
        cmap="tab10",
        alpha=0.7,
        edgecolors="black",
        linewidths=1.5,
    )

    for i, model in enumerate(models):
        ax.annotate(
            model.replace("_", " ").title(),
            (times[i], f1_scores[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Training Time (seconds)", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Model Efficiency: Training Time vs Performance", fontsize=14)
    ax.grid(True, alpha=0.3)
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
        ax.plot(
            fpr,
            tpr,
            color=colors[i],
            linewidth=2,
            alpha=0.8,
            label=f"{model_name.replace('_', ' ').title()} (AUC = {auc:.3f})",
        )

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
        ax.plot(
            recall,
            precision,
            color=colors[i],
            linewidth=2,
            alpha=0.8,
            label=f"{model_name.replace('_', ' ').title()} (AP = {ap:.3f})",
        )

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
    ax.plot(
        thresholds,
        precisions,
        label="Precision",
        linewidth=2,
        color="#2196F3",
        marker="o",
        markersize=3,
    )
    ax.plot(
        thresholds, recalls, label="Recall", linewidth=2, color="#4CAF50", marker="s", markersize=3
    )
    ax.plot(
        thresholds,
        f1_scores,
        label="F1 Score",
        linewidth=2,
        color="#FF9800",
        marker="^",
        markersize=3,
    )

    optimal_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    ax.axvline(
        optimal_thresh,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Optimal Threshold ({optimal_thresh:.2f})",
    )

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
    bars = ax.bar(
        [class_names[c] if c < len(class_names) else f"Class_{c}" for c in classes],
        counts,
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + count * 0.01,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

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

    bars1 = ax.bar(
        x - width / 2,
        fp_per_class,
        width,
        label="False Positives",
        color="#FF9800",
        edgecolor="black",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        fn_per_class,
        width,
        label="False Negatives",
        color="#F44336",
        edgecolor="black",
        alpha=0.8,
    )

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
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

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
