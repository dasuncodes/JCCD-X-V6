#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Ablation Studies ==="

# Determine best model name from training
MODEL_NAME="xgboost"
if [[ -f "artifacts/models/best_model_name.joblib" ]]; then
    MODEL_NAME=$(poetry run python -c "import joblib; print(joblib.load('artifacts/models/best_model_name.joblib'))")
    echo "Using best model from training: $MODEL_NAME"
fi

FEATURE_SELECTION_ARG=""
if [[ -f "artifacts/models/selected_features.json" ]]; then
    FEATURE_SELECTION_ARG="--feature-selection artifacts/models/selected_features.json"
    echo "Using selected features from RFE"
fi

poetry run python -m src.python.evaluation.ablation \
    --features-dir data/intermediate/features \
    --output-dir artifacts/evaluation \
    --model-name "$MODEL_NAME" \
    --cv 5 \
    $FEATURE_SELECTION_ARG

echo "=== Ablation Studies Complete ==="
