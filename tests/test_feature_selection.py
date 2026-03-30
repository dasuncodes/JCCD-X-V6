"""Tests for feature selection module."""

import numpy as np
import pytest

from src.python.model.feature_selection import (
    compare_feature_subsets,
    run_rfe_analysis,
)
from src.python.model.models import get_models


class TestRFEAnalysis:
    def test_rfe_basic(self):
        """Test basic RFE analysis runs without errors."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = get_models()["logistic_regression"]
        results = run_rfe_analysis(
            X, y, model, "logistic_regression",
            n_features_range=[3, 5, 8],
            cv=3,
        )

        assert results["model"] == "logistic_regression"
        assert results["total_features"] == 10
        assert len(results["experiments"]) > 0
        assert "optimal_n_features" in results
        assert "optimal_f1" in results

        # Check experiment structure
        for exp in results["experiments"]:
            assert "n_features" in exp
            assert "cv_f1_mean" in exp
            assert "cv_f1_std" in exp

    def test_rfe_with_xgboost(self):
        """Test RFE with XGBoost model."""
        np.random.seed(42)
        X = np.random.randn(200, 15)
        y = (X[:, 0] * 2 + X[:, 1] > 0).astype(int)

        model = get_models()["xgboost"]
        results = run_rfe_analysis(
            X, y, model, "xgboost",
            n_features_range=[5, 10, 15],
            cv=3,
        )

        assert results["model"] == "xgboost"
        assert len(results["experiments"]) == 3
        assert all(exp["cv_f1_mean"] > 0 for exp in results["experiments"])


class TestCompareFeatureSubsets:
    def test_subset_comparison(self):
        """Test feature subset comparison."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
        feature_names = [f"feat_{i}" for i in range(20)]

        # Create mock RFE results
        rfe_results = {
            "experiments": [
                {"n_features": 5, "selected_indices": [0, 1, 2, 3, 4]},
                {"n_features": 10, "selected_indices": list(range(10))},
            ]
        }

        model = get_models()["random_forest"]
        comparison = compare_feature_subsets(
            X, y, feature_names, rfe_results, model, cv=3,
        )

        assert "subsets" in comparison
        # Should have subsets for top 5, 10, 15, 20 features
        assert len(comparison["subsets"]) > 0

        for subset in comparison["subsets"]:
            assert "n_features" in subset
            assert "feature_names" in subset
            assert "cv_f1_mean" in subset
            assert len(subset["feature_names"]) == subset["n_features"]
