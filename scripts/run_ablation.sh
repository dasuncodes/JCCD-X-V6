#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Ablation Studies ==="
poetry run python -m src.python.evaluation.ablation \
    --features-dir data/intermediate/features \
    --output-dir artifacts/evaluation \
    --model-name xgboost \
    --cv 5
echo "=== Ablation Studies Complete ==="
