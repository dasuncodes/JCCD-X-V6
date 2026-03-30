"""Tests for ML model training."""

import numpy as np

from src.python.model.models import get_models
from src.python.model.train import cross_validate_model, evaluate_fold


class TestModels:
    def test_get_models(self):
        models = get_models()
        assert len(models) == 5
        assert "xgboost" in models
        assert "random_forest" in models
        assert "logistic_regression" in models
        assert "linear_svm" in models
        assert "knn" in models


class TestEvaluateFold:
    def test_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = evaluate_fold(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["tp"] == 2
        assert m["fp"] == 0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = evaluate_fold(y_true, y_pred)
        assert m["accuracy"] == 0.0
        assert m["f1"] == 0.0


class TestCrossValidate:
    def test_basic_cv(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=500)
        result = cross_validate_model(model, X, y, cv=3)
        assert len(result["folds"]) == 3
        assert "f1_mean" in result["aggregate"]
        assert result["aggregate"]["f1_mean"] > 0.5
