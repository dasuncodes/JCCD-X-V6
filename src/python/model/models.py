"""Model definitions for code clone detection."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def get_models() -> dict:
    """Return dict of model name → sklearn estimator."""
    return {
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=5000,  # Increased from 1000 for better convergence
            C=1.0,
            random_state=42,
            solver='lbfgs',
        ),
        "linear_svm": LinearSVC(
            max_iter=5000,  # Increased from 2000 for better convergence
            C=1.0,
            random_state=42,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=7,
            n_jobs=-1,
        ),
    }
