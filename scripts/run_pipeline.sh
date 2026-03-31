#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  JCCD-X-V6 Full Pipeline Evaluation"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check prerequisites
if [[ ! -f "artifacts/models/best_model.joblib" ]]; then
    echo "ERROR: Model training not complete!"
    echo "Please run training first:"
    echo "  bash scripts/run_training.sh"
    exit 1
fi

if [[ ! -f "data/intermediate/token_data.pkl" ]]; then
    echo "ERROR: Tokenization not complete!"
    echo "Please run tokenization first:"
    echo "  bash scripts/run_tokenization.sh"
    exit 1
fi

echo "=== Step 6: Full Pipeline Evaluation ==="
echo ""
echo "  Pipeline Stages:"
echo "    1. Type-1 Detection (exact match after normalization)"
echo "    2. Type-2 Detection (identical abstracted tokens)"
echo "    3. LSH Candidate Filtering (MinHash + LSH)"
echo "    4. ML Classification (Type-3 vs Non-clone)"
echo ""
echo "  Test Set:"
echo "    - Type-1 clones: 100% (~48,000 pairs)"
echo "    - Type-2 clones: 100% (~4,200 pairs)"
echo "    - Type-3 clones: 30% (~32,000 pairs)"
echo "    - Non-clones: 30% (~117,000 pairs)"
echo ""

# Run pipeline
# Optimized LSH parameters for better recall (93%+) while maintaining good filtering (~46% reduction)
# Adjust --lsh-bands and --lsh-hashes based on your needs:
#   - More bands (e.g., 24) = higher recall, less filtering
#   - Fewer bands (e.g., 12) = lower recall, more filtering
#   - See LSH_OPTIMIZATION_RESULTS.md for detailed comparison
poetry run python -m src.python.pipeline.full_pipeline \
    --data-dir data/processed \
    --intermediate-dir data/intermediate \
    --model-dir artifacts/models \
    --eval-dir artifacts/evaluation \
    --lsh-hashes 48 \
    --lsh-bands 16 \
    --shingle-k 3

echo ""
echo "=========================================="
echo "  Pipeline Evaluation Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - artifacts/evaluation/pipeline_metrics.json"
echo "  - artifacts/evaluation/plots/pipeline/"
echo ""
echo "Optional: Run analysis modules"
echo "  bash scripts/run_all.sh --with-analysis"
echo ""
