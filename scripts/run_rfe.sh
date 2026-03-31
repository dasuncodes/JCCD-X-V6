#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Feature Selection (RFE) ==="

# Determine best model name from training
MODEL_NAME="xgboost"
if [[ -f "artifacts/models/best_model_name.joblib" ]]; then
    # Load pickled string using python
    MODEL_NAME=$(poetry run python -c "import joblib; print(joblib.load('artifacts/models/best_model_name.joblib'))")
    echo "Using best model from training: $MODEL_NAME"
fi

poetry run python -m src.python.model.feature_selection \
    --features-dir data/intermediate/features \
    --output-dir artifacts/evaluation \
    --model-name "$MODEL_NAME" \
    --cv 5 \
    --output-model-dir artifacts/models \
    --selected-features-file artifacts/models/selected_features.json \
    --retrain

echo "=== Feature Selection Complete ==="
