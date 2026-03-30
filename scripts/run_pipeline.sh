#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Full Pipeline ==="
poetry run python -m src.python.pipeline.full_pipeline \
    --data-dir data/processed \
    --intermediate-dir data/intermediate \
    --model-dir artifacts/models \
    --eval-dir artifacts/evaluation \
    --lsh-hashes 128 \
    --lsh-bands 16 \
    --shingle-k 3
echo "=== Pipeline Complete ==="
