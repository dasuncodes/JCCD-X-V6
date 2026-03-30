#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  JCCD-X-V6 Feature Engineering"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if tokenization was run
if [[ ! -f "data/intermediate/token_data.pkl" ]]; then
    echo "ERROR: Tokenization not complete!"
    echo "Please run tokenization first:"
    echo "  bash scripts/run_tokenization.sh"
    exit 1
fi

echo "=== Step 4: Feature Engineering (Zig-parallelized) ==="
echo "  - Lexical features (Levenshtein, Jaccard, Dice, LCS)"
echo "  - Structural features (AST histogram, bigram cosine)"
echo "  - Quantitative features (cyclomatic complexity)"
echo "  - Zig parallelization for batch computation"
echo ""

# Run feature engineering
poetry run python -m src.python.feature_engineering.features \
    --data-dir data/processed \
    --intermediate-dir data/intermediate \
    --output-dir data/intermediate/features \
    --eval-dir artifacts/evaluation

echo ""
echo "=========================================="
echo "  Feature Engineering Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - data/intermediate/features/train_features.csv"
echo "  - data/intermediate/features/test_features.csv"
echo "  - artifacts/evaluation/feature_metrics.json"
echo "  - artifacts/evaluation/plots/features/"
echo ""
echo "Next step: Run ML training"
echo "  bash scripts/run_training.sh"
echo ""
