#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  JCCD-X-V6 Data Preparation"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if source files exist
if [[ ! -d "data/raw/toma/id2sourcecode" ]]; then
    echo "ERROR: Source directory not found: data/raw/toma/id2sourcecode"
    echo "Please ensure the TOMA dataset is properly extracted."
    exit 1
fi

# Check if CSV files exist
for csv_file in type-1.csv type-2.csv type-3.csv type-4.csv type-5.csv nonclone.csv; do
    if [[ ! -f "data/raw/toma/$csv_file" ]]; then
        echo "ERROR: Missing dataset file: data/raw/toma/$csv_file"
        exit 1
    fi
done

echo "=== Step 1: Loading and Validating Datasets ==="
echo "  - type-1.csv (Type-1 clones)"
echo "  - type-2.csv (Type-2 clones)"
echo "  - type-3.csv (Type-3 clones)"
echo "  - type-4.csv (Type-4 clones → treated as Type-3)"
echo "  - type-5.csv (Type-5 clones → Non-clones)"
echo "  - nonclone.csv (Non-clones)"
echo ""

# Run data preparation
poetry run python -m src.python.preprocessing.data_prep \
    --source-dir data/raw/toma/id2sourcecode \
    --raw-dir data/raw/toma \
    --output-dir data/processed \
    --eval-dir artifacts/evaluation \
    --test-ratio 0.3 \
    --seed 42

echo ""
echo "=========================================="
echo "  Data Preparation Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - data/processed/training_dataset.csv (balanced, binary)"
echo "  - data/processed/testing_dataset.csv (multi-class)"
echo "  - artifacts/evaluation/data_prep_metrics.json"
echo "  - artifacts/evaluation/plots/data_prep/"
echo ""
echo "Next step: Run normalization"
echo "  bash scripts/run_preprocessing.sh"
echo ""
