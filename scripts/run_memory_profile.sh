#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Memory Usage Profiling ==="
poetry run python -m src.python.utils.memory \
    --features-dir data/intermediate/features \
    --data-dir data/processed \
    --model-dir artifacts/models \
    --output-dir artifacts/evaluation \
    --cv 5
echo "=== Memory Usage Profiling Complete ==="
