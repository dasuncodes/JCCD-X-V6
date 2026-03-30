"""Centralized metrics computation for JCCD-X pipeline."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    zero_division: float = 0.0,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        zero_division: Value to return when division by zero occurs

    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=zero_division)),
        "recall": float(recall_score(y_true, y_pred, zero_division=zero_division)),
        "f1": float(f1_score(y_true, y_pred, zero_division=zero_division)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except (ValueError, IndexError):
            metrics["roc_auc"] = 0.0
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except (ValueError, IndexError):
            metrics["pr_auc"] = 0.0

    # Confusion matrix breakdown
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


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    fold_idx: int = 0,
) -> dict:
    """
    Compute metrics for a single cross-validation fold.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        fold_idx: Fold index for tracking

    Returns:
        Dictionary with fold metrics and fold index
    """
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    metrics["fold"] = fold_idx
    return metrics


def aggregate_cv_metrics(fold_results: list) -> dict:
    """
    Aggregate metrics across cross-validation folds.

    Args:
        fold_results: List of dictionaries, one per fold

    Returns:
        Dictionary with mean and std for each metric
    """
    if not fold_results:
        return {}

    # Exclude 'fold' index from aggregation
    metric_keys = [k for k in fold_results[0].keys() if k != "fold"]
    aggregate = {}

    for key in metric_keys:
        values = [f[key] for f in fold_results if key in f and isinstance(f[key], (int, float))]
        if values:
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))
            aggregate[f"{key}_min"] = float(np.min(values))
            aggregate[f"{key}_max"] = float(np.max(values))

    return aggregate


def compute_cv_stability(fold_results: list) -> dict:
    """
    Compute stability metrics for cross-validation results.

    Stability is measured by coefficient of variation (CV = std/mean).
    Lower CV indicates more stable model.

    Args:
        fold_results: List of dictionaries, one per fold

    Returns:
        Dictionary with stability metrics for each metric
    """
    aggregate = aggregate_cv_metrics(fold_results)
    stability = {}

    for key in aggregate:
        if key.endswith("_mean") or key.endswith("_std"):
            base_name = key.replace("_mean", "").replace("_std", "")
            mean_key = f"{base_name}_mean"
            std_key = f"{base_name}_std"

            if mean_key in aggregate and std_key in aggregate:
                mean_val = aggregate[mean_key]
                std_val = aggregate[std_key]

                # Coefficient of variation (CV)
                cv = std_val / mean_val if mean_val > 0 else float('inf')
                stability[f"{base_name}_cv"] = cv

                # Stability score (inverse of CV, capped at 1.0)
                stability[f"{base_name}_stability"] = 1.0 / (1.0 + cv)

    return stability


def compute_lsh_metrics(
    candidates: set,
    true_pairs: set,
    total_pairs: int,
    computation_time: float = None,
) -> dict:
    """
    Compute metrics for LSH candidate filtering.

    Args:
        candidates: Set of candidate pairs after LSH
        true_pairs: Set of true clone pairs (ground truth)
        total_pairs: Total number of pairs before LSH
        computation_time: Time taken for LSH computation (optional)

    Returns:
        Dictionary with LSH performance metrics
    """
    n_candidates = len(candidates)
    n_true = len(true_pairs)
    n_true_retained = len(candidates & true_pairs)

    metrics = {
        "candidates_before": total_pairs,
        "candidates_after": n_candidates,
        "true_pairs": n_true,
        "true_retained": n_true_retained,
        "reduction_ratio": float((total_pairs - n_candidates) / total_pairs) if total_pairs > 0 else 0.0,
        "candidate_recall": float(n_true_retained / n_true) if n_true > 0 else 0.0,
        "candidate_precision": float(n_true_retained / n_candidates) if n_candidates > 0 else 0.0,
    }

    if computation_time is not None:
        metrics["computation_time_sec"] = computation_time
        metrics["pairs_per_sec"] = float(total_pairs / computation_time) if computation_time > 0 else 0.0

    return metrics


def compute_feature_metrics(feature_df) -> dict:
    """
    Compute descriptive statistics for features.

    Args:
        feature_df: DataFrame with features (columns) and samples (rows)

    Returns:
        Dictionary with statistics for each feature
    """
    import pandas as pd

    feature_cols = [c for c in feature_df.columns if c not in ("label", "id1", "id2")]
    metrics = {
        "total_pairs": len(feature_df),
        "num_features": len(feature_cols),
    }

    for col in feature_cols:
        if col in feature_df.columns:
            vals = feature_df[col].values
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                metrics[col] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "median": float(np.median(vals)),
                    "q25": float(np.percentile(vals, 25)),
                    "q75": float(np.percentile(vals, 75)),
                }

    return metrics


def compute_pipeline_metrics(
    predictions: dict,
    ground_truth: dict,
    type1_count: int = 0,
    type2_count: int = 0,
    lsh_metrics: dict = None,
    timing: dict = None,
) -> dict:
    """
    Compute comprehensive pipeline evaluation metrics.

    Args:
        predictions: Dictionary mapping (id1, id2) -> predicted label
        ground_truth: Dictionary mapping (id1, id2) -> true label
        type1_count: Number of Type-1 clones detected
        type2_count: Number of Type-2 clones detected
        lsh_metrics: LSH-specific metrics (optional)
        timing: Timing information per stage (optional)

    Returns:
        Dictionary with comprehensive pipeline metrics
    """
    # Convert to arrays for sklearn metrics
    pairs = list(ground_truth.keys())
    y_true = np.array([ground_truth[p] for p in pairs])
    y_pred = np.array([predictions.get(p, 0) for p in pairs])

    # Basic classification metrics
    metrics = compute_classification_metrics(y_true, y_pred)

    # Add Type-1/2 counts
    metrics["type1_detected"] = type1_count
    metrics["type2_detected"] = type2_count

    # Add LSH metrics if provided
    if lsh_metrics:
        metrics.update(lsh_metrics)

    # Add timing if provided
    if timing:
        metrics.update(timing)

    return metrics


def compute_ablation_metrics(
    baseline_metrics: dict,
    ablated_metrics: dict,
    ablation_name: str,
) -> dict:
    """
    Compute metrics for ablation study comparison.

    Args:
        baseline_metrics: Metrics from full/complete pipeline
        ablated_metrics: Metrics from ablated configuration
        ablation_name: Name of the ablation (e.g., "no_normalization")

    Returns:
        Dictionary with ablation comparison metrics
    """
    comparison = {
        "ablation": ablation_name,
        "baseline": baseline_metrics,
        "ablated": ablated_metrics,
        "delta": {},
    }

    # Compute delta for key metrics
    for key in ["f1", "accuracy", "precision", "recall", "roc_auc"]:
        if key in baseline_metrics and key in ablated_metrics:
            delta = ablated_metrics[key] - baseline_metrics[key]
            comparison["delta"][key] = delta
            comparison["delta"][f"{key}_pct_change"] = (
                delta / baseline_metrics[key] * 100 if baseline_metrics[key] > 0 else 0.0
            )

    return comparison
