#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Tokenization & CST Generation ==="
poetry run python -m src.python.preprocessing.tokenizer \
    --source-dir data/raw/toma/id2sourcecode \
    --normalized-dir data/processed/normalized \
    --data-dir data/processed \
    --output-dir data/intermediate \
    --eval-dir artifacts/evaluation
echo "=== Tokenization Complete ==="
