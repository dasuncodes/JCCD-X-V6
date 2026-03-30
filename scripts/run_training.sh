#!/usr/bin/env bash
set -euo pipefail

echo "=== Running ML Model Training ==="
poetry run python -m src.python.model.train \
    --features-dir data/intermediate/features \
    --model-dir artifacts/models \
    --eval-dir artifacts/evaluation \
    --seed 42
echo "=== Training Complete ==="
