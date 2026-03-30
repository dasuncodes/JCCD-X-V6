"""Tests for LSH parameter tuning module."""

import numpy as np
import pytest

from src.python.pipeline.lsh_tuning import (
    recommend_parameters,
    simulate_lsh_performance,
    sweep_lsh_parameters,
)


class TestLSHSimulation:
    def test_simulate_basic(self):
        """Test basic LSH simulation."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.array([0] * 50 + [1] * 50)

        metrics = simulate_lsh_performance(
            X, y,
            n_hashes=128,
            n_bands=16,
            shingle_k=3,
        )

        assert "n_hashes" in metrics
        assert "n_bands" in metrics
        assert "recall" in metrics
        assert "precision" in metrics
        assert "reduction_ratio" in metrics
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["reduction_ratio"] <= 1

    def test_simulate_different_params(self):
        """Test simulation with different parameters."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.array([0] * 50 + [1] * 50)

        # More hashes should generally improve performance
        metrics_64 = simulate_lsh_performance(X, y, 64, 8, 3)
        metrics_256 = simulate_lsh_performance(X, y, 256, 32, 3)

        assert metrics_64["n_hashes"] == 64
        assert metrics_256["n_hashes"] == 256
        assert metrics_64["rows_per_band"] == 8
        assert metrics_256["rows_per_band"] == 8


class TestLSHSweep:
    def test_sweep_basic(self):
        """Test basic parameter sweep."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.array([0] * 50 + [1] * 50)

        results = sweep_lsh_parameters(
            X, y,
            hash_range=(64, 128, 64),
            bands_range=(8, 16, 8),
            shingle_k_range=(3, 4, 1),
        )

        assert "experiments" in results
        assert len(results["experiments"]) > 0
        assert "best_by_recall" in results
        assert "best_by_precision" in results
        assert "best_by_f1" in results
        assert "best_balanced" in results

        # Check experiment structure
        for exp in results["experiments"]:
            assert "n_hashes" in exp
            assert "n_bands" in exp
            assert "recall" in exp
            assert "precision" in exp
            assert "f1" in exp

    def test_sweep_divisibility(self):
        """Test that sweep only uses divisible hash/band combinations."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.array([0] * 50 + [1] * 50)

        results = sweep_lsh_parameters(
            X, y,
            hash_range=(64, 128, 32),
            bands_range=(8, 16, 8),
            shingle_k_range=(3, 4, 1),
        )

        # All experiments should have divisible hash/band counts
        for exp in results["experiments"]:
            assert exp["n_hashes"] % exp["n_bands"] == 0
            assert exp["rows_per_band"] == exp["n_hashes"] // exp["n_bands"]


class TestRecommendation:
    def test_recommend_with_valid_configs(self):
        """Test recommendation with valid configurations."""
        results = {
            "experiments": [
                {
                    "n_hashes": 128,
                    "n_bands": 16,
                    "shingle_k": 3,
                    "recall": 0.95,
                    "precision": 0.60,
                    "f1": 0.74,
                    "reduction_ratio": 0.85,
                    "computation_time_sec": 1.5,
                },
                {
                    "n_hashes": 64,
                    "n_bands": 8,
                    "shingle_k": 3,
                    "recall": 0.90,
                    "precision": 0.55,
                    "f1": 0.68,
                    "reduction_ratio": 0.80,
                    "computation_time_sec": 0.8,
                },
            ]
        }

        recommendation = recommend_parameters(
            results,
            min_recall=0.90,
            min_precision=0.50,
            max_time=5.0,
        )

        assert recommendation["recommended"] is not None
        assert recommendation["recommended"]["recall"] >= 0.90
        assert recommendation["recommended"]["precision"] >= 0.50
        assert "alternatives" in recommendation
        assert len(recommendation["alternatives"]) <= 3

    def test_recommend_no_valid_configs(self):
        """Test recommendation when no configs meet constraints."""
        results = {
            "experiments": [
                {
                    "n_hashes": 128,
                    "n_bands": 16,
                    "shingle_k": 3,
                    "recall": 0.80,  # Below threshold
                    "precision": 0.40,  # Below threshold
                    "f1": 0.53,
                    "reduction_ratio": 0.75,
                    "computation_time_sec": 1.5,
                },
            ]
        }

        recommendation = recommend_parameters(
            results,
            min_recall=0.90,
            min_precision=0.50,
            max_time=5.0,
        )

        # Should still return something (relaxed constraints)
        assert "recommended" in recommendation or "reason" in recommendation
