#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Data Preparation ==="
poetry run python -m src.python.preprocessing.data_prep \
    --source-dir data/raw/toma/id2sourcecode \
    --raw-dir data/raw/toma \
    --output-dir data/processed \
    --eval-dir artifacts/evaluation \
    --test-ratio 0.3 \
    --seed 42
echo "=== Data Preparation Complete ==="
