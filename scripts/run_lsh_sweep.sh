#!/usr/bin/env bash
set -euo pipefail

echo "=== Running LSH Parameter Sweep ==="
poetry run python -m src.python.pipeline.lsh_tuning \
    --features-dir data/intermediate/features \
    --data-dir data/processed \
    --output-dir artifacts/evaluation \
    --hash-range 64 256 64 \
    --bands-range 8 32 8 \
    --shingle-k-range 3 5 1 \
    --min-recall 0.90 \
    --min-precision 0.50 \
    --max-time 5.0
echo "=== LSH Parameter Sweep Complete ==="
