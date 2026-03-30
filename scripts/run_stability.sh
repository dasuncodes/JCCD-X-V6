#!/usr/bin/env bash
set -euo pipefail

echo "=== Running CV Stability Analysis ==="
poetry run python -m src.python.evaluation.stability \
    --features-dir data/intermediate/features \
    --output-dir artifacts/evaluation \
    --model-name xgboost \
    --cv 5
echo "=== CV Stability Analysis Complete ==="
