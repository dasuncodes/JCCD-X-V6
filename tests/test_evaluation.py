"""Tests for evaluation metrics module."""

import numpy as np

from src.python.evaluation.metrics import (
    compute_ablation_metrics,
    aggregate_cv_metrics,
    compute_classification_metrics,
    compute_cv_stability,
    compute_feature_metrics,
    compute_fold_metrics,
    compute_lsh_metrics,
    compute_pipeline_metrics,
)


class TestClassificationMetrics:
    def test_perfect_classification(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["tp"] == 2
        assert metrics["fp"] == 0
        assert metrics["tn"] == 2
        assert metrics["fn"] == 0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["tp"] == 0
        assert metrics["fp"] == 2

    def test_without_proba(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "roc_auc" not in metrics
        assert "pr_auc" not in metrics


class TestFoldMetrics:
    def test_fold_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        metrics = compute_fold_metrics(y_true, y_pred, fold_idx=2)

        assert metrics["fold"] == 2
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0


class TestAggregateCVMetrics:
    def test_aggregate(self):
        fold_results = [
            {"f1": 0.9, "accuracy": 0.85, "fold": 0},
            {"f1": 0.92, "accuracy": 0.87, "fold": 1},
            {"f1": 0.88, "accuracy": 0.84, "fold": 2},
        ]

        aggregate = aggregate_cv_metrics(fold_results)

        assert "f1_mean" in aggregate
        assert "f1_std" in aggregate
        assert "f1_min" in aggregate
        assert "f1_max" in aggregate
        assert abs(aggregate["f1_mean"] - 0.9) < 0.02
        assert abs(aggregate["accuracy_mean"] - 0.853) < 0.02


class TestCVStability:
    def test_stability_metrics(self):
        fold_results = [
            {"f1": 0.90, "accuracy": 0.85},
            {"f1": 0.90, "accuracy": 0.85},
            {"f1": 0.90, "accuracy": 0.85},
        ]

        stability = compute_cv_stability(fold_results)

        assert "f1_cv" in stability
        assert "f1_stability" in stability
        assert stability["f1_cv"] == 0.0  # Perfect stability
        assert stability["f1_stability"] == 1.0

    def test_unstable_cv(self):
        fold_results = [
            {"f1": 0.70, "accuracy": 0.65},
            {"f1": 0.90, "accuracy": 0.85},
            {"f1": 0.80, "accuracy": 0.75},
        ]

        stability = compute_cv_stability(fold_results)

        assert stability["f1_cv"] > 0.0
        assert stability["f1_stability"] < 1.0


class TestLSHMetrics:
    def test_lsh_metrics(self):
        candidates = {(1, 2), (3, 4), (5, 6)}
        true_pairs = {(1, 2), (3, 4), (7, 8)}

        metrics = compute_lsh_metrics(candidates, true_pairs, total_pairs=100)

        assert metrics["candidates_before"] == 100
        assert metrics["candidates_after"] == 3
        assert metrics["true_pairs"] == 3
        assert metrics["true_retained"] == 2
        assert metrics["reduction_ratio"] == 0.97
        assert metrics["candidate_recall"] == 2 / 3
        assert metrics["candidate_precision"] == 2 / 3

    def test_lsh_with_timing(self):
        candidates = {(1, 2)}
        true_pairs = {(1, 2)}

        metrics = compute_lsh_metrics(
            candidates, true_pairs,
            total_pairs=1000,
            computation_time=0.5
        )

        assert "computation_time_sec" in metrics
        assert "pairs_per_sec" in metrics
        assert metrics["pairs_per_sec"] == 2000.0


class TestFeatureMetrics:
    def test_feature_metrics(self):
        import pandas as pd

        df = pd.DataFrame({
            "feat1": [0.1, 0.2, 0.3, 0.4],
            "feat2": [0.5, 0.6, 0.7, 0.8],
            "label": [0, 1, 0, 1],
            "id1": [1, 2, 3, 4],
            "id2": [5, 6, 7, 8],
        })

        metrics = compute_feature_metrics(df)

        assert metrics["total_pairs"] == 4
        assert metrics["num_features"] == 2
        assert "feat1" in metrics
        assert "mean" in metrics["feat1"]


class TestPipelineMetrics:
    def test_pipeline_metrics(self):
        predictions = {(1, 2): 1, (3, 4): 0, (5, 6): 1}
        ground_truth = {(1, 2): 1, (3, 4): 0, (5, 6): 0}

        metrics = compute_pipeline_metrics(
            predictions, ground_truth,
            type1_count=10,
            type2_count=5,
        )

        assert metrics["type1_detected"] == 10
        assert metrics["type2_detected"] == 5
        assert "accuracy" in metrics
        assert "f1" in metrics


class TestAblationMetrics:
    def test_ablation_comparison(self):
        baseline = {"f1": 0.95, "accuracy": 0.93}
        ablated = {"f1": 0.90, "accuracy": 0.88}

        comparison = compute_ablation_metrics(
            baseline, ablated,
            ablation_name="no_normalization"
        )

        assert comparison["ablation"] == "no_normalization"
        assert "delta" in comparison
        assert comparison["delta"]["f1"] < 0  # Performance decreased
        assert comparison["delta"]["f1_pct_change"] < 0
