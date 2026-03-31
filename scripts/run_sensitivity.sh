#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Sensitivity Analysis ==="
poetry run python -m src.python.evaluation.sensitivity \
    --features-dir data/intermediate/features \
    --output-dir artifacts/evaluation \
    --model-name xgboost \
    --perturbation-types noise mask shift \
    --perturbation-levels 0.05 0.10 0.15 0.20 0.25 \
    --cv 5
echo "=== Sensitivity Analysis Complete ==="
