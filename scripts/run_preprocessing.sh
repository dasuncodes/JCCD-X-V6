#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  JCCD-X-V6 Source Code Normalization"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if data preparation was run
if [[ ! -f "data/processed/training_dataset.csv" ]] || [[ ! -f "data/processed/testing_dataset.csv" ]]; then
    echo "ERROR: Data preparation not complete!"
    echo "Please run data preparation first:"
    echo "  bash scripts/run_data_prep.sh"
    exit 1
fi

echo "=== Step 2: Normalizing Source Code ==="
echo "  - Removing comments"
echo "  - Removing blank lines"
echo "  - Standardizing whitespace"
echo "  - Standardizing brace placement"
echo ""

# Run normalization
poetry run python -m src.python.preprocessing.normalization \
    --source-dir data/raw/toma/id2sourcecode \
    --data-dir data/processed \
    --output-dir data/processed/normalized \
    --eval-dir artifacts/evaluation

echo ""
echo "=========================================="
echo "  Normalization Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - data/processed/normalized/*.java (normalized source files)"
echo "  - data/processed/normalization_impact.csv"
echo "  - artifacts/evaluation/normalization_metrics.json"
echo "  - artifacts/evaluation/plots/normalization/"
echo ""
echo "Next step: Run tokenization"
echo "  bash scripts/run_tokenization.sh"
echo ""
