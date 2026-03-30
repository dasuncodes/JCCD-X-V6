#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=== Running Data Preparation ==="
echo "Project root: $PROJECT_ROOT"
poetry run python -m src.python.preprocessing.data_prep \
    --source-dir data/raw/toma/id2sourcecode \
    --raw-dir data/raw/toma \
    --output-dir data/processed \
    --eval-dir artifacts/evaluation \
    --test-ratio 0.3 \
    --seed 42
echo "=== Data Preparation Complete ==="
