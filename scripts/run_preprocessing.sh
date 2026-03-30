#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Source Code Normalization ==="
poetry run python -m src.python.preprocessing.normalization \
    --source-dir data/raw/toma/id2sourcecode \
    --data-dir data/processed \
    --output-dir data/processed/normalized \
    --eval-dir artifacts/evaluation
echo "=== Normalization Complete ==="
