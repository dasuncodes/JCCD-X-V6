#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Comprehensive Ablation Study"
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

if [[ ! -f "data/intermediate/features/train_features.csv" ]]; then
    echo "ERROR: Feature engineering not complete!"
    echo "Please run feature engineering first:"
    echo "  bash scripts/run_features.sh"
    exit 1
fi

echo "Running comprehensive ablation study..."
echo ""
echo "  Studies:"
echo "    1. Individual Feature Ablation (remove each feature)"
echo "    2. Feature Group Ablation (remove feature groups)"
echo "    3. LSH Parameter Ablation (reduction vs accuracy)"
echo ""
echo "  Output:"
echo "    - artifacts/evaluation/ablation_study/"
echo "    - JSON results, CSV tables, PNG plots"
echo ""

# Run ablation study with all studies
poetry run python -m src.python.evaluation.run_ablation_study \
    --data-dir data/processed \
    --intermediate-dir data/intermediate \
    --model-dir artifacts/models \
    --output-dir artifacts/evaluation/ablation_study \
    --model-name xgboost \
    --cv 5 \
    --seed 42 \
    --study-type both

echo ""
echo "=========================================="
echo "  Ablation Study Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  Results:"
echo "    - artifacts/evaluation/ablation_study/individual_feature_ablation_results.json"
echo "    - artifacts/evaluation/ablation_study/feature_group_ablation_results.json"
echo "    - artifacts/evaluation/ablation_study/lsh_ablation_results.json"
echo "    - artifacts/evaluation/ablation_study/combined_ablation_results.json"
echo ""
echo "  Summary Tables:"
echo "    - artifacts/evaluation/ablation_study/individual_feature_ablation_summary.csv/.md"
echo "    - artifacts/evaluation/ablation_study/feature_group_ablation_summary.csv/.md"
echo "    - artifacts/evaluation/ablation_study/lsh_ablation_summary.csv/.md"
echo ""
echo "  Visualizations:"
echo "    - artifacts/evaluation/ablation_study/plots/individual_feature_ablation_comparison.png"
echo "    - artifacts/evaluation/ablation_study/plots/feature_group_ablation_comparison.png"
echo "    - artifacts/evaluation/ablation_study/plots/lsh_ablation_comparison.png"
echo ""
echo "For paper:"
echo "  - Use individual feature ablation for feature importance analysis"
echo "  - Use feature group ablation for component contribution"
echo "  - Use LSH ablation for efficiency-accuracy trade-off"
echo "  - Reference JSON for detailed metrics"
echo ""
