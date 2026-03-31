#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Feature Selection (RFE) ==="
poetry run python -m src.python.model.feature_selection \
    --features-dir data/intermediate/features \
    --output-dir artifacts/evaluation \
    --model-name xgboost \
    --cv 5
echo "=== Feature Selection Complete ==="
