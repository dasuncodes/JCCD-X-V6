#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  JCCD-X-V6 Tokenization & CST"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if normalization was run
if [[ ! -d "data/processed/normalized" ]]; then
    echo "ERROR: Normalization not complete!"
    echo "Please run normalization first:"
    echo "  bash scripts/run_preprocessing.sh"
    exit 1
fi

echo "=== Step 3: Tokenization & CST Generation ==="
echo "  - Parsing normalized source with Tree-sitter"
echo "  - Generating Concrete Syntax Trees (CST)"
echo "  - Extracting token sequences"
echo "  - Computing AST statistics"
echo ""

# Run tokenization
poetry run python -m src.python.preprocessing.tokenizer \
    --source-dir data/raw/toma/id2sourcecode \
    --normalized-dir data/processed/normalized \
    --data-dir data/processed \
    --output-dir data/intermediate \
    --eval-dir artifacts/evaluation

echo ""
echo "=========================================="
echo "  Tokenization Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - data/intermediate/token_data.pkl (token sequences + AST stats)"
echo "  - artifacts/evaluation/tokenization_metrics.json"
echo "  - artifacts/evaluation/plots/tokenization/"
echo ""
echo "Next step: Run feature engineering"
echo "  bash scripts/run_features.sh"
echo ""
