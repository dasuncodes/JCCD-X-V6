#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Probability Calibration ==="
poetry run python -m src.python.model.calibration \
    --features-dir data/intermediate/features \
    --model-dir artifacts/models \
    --output-dir artifacts/evaluation \
    --model-name xgboost \
    --cv 5
echo "=== Probability Calibration Complete ==="
