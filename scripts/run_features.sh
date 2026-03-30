#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Feature Engineering ==="
poetry run python -m src.python.feature_engineering.features \
    --data-dir data/processed \
    --intermediate-dir data/intermediate \
    --output-dir data/intermediate/features \
    --eval-dir artifacts/evaluation
echo "=== Feature Engineering Complete ==="
