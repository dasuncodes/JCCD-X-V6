#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  JCCD-X-V6 ML Model Training"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if feature engineering was run
if [[ ! -f "data/intermediate/features/train_features.csv" ]] || \
   [[ ! -f "data/intermediate/features/test_features.csv" ]]; then
    echo "ERROR: Feature engineering not complete!"
    echo "Please run feature engineering first:"
    echo "  bash scripts/run_features.sh"
    exit 1
fi

echo "=== Step 5: ML Model Training (5-Fold Stratified CV) ==="
echo "  Models:"
echo "    - XGBoost"
echo "    - Random Forest"
echo "    - Logistic Regression"
echo "    - Linear SVM"
echo "    - KNN"
echo ""
echo "  Evaluation:"
echo "    - 5-Fold Stratified Cross-Validation"
echo "    - Metrics: Accuracy, Precision, Recall, F1, AUC-ROC"
echo "    - Model selection by F1 score"
echo ""

# Run training
poetry run python -m src.python.model.train \
    --features-dir data/intermediate/features \
    --model-dir artifacts/models \
    --eval-dir artifacts/evaluation

echo ""
echo "=========================================="
echo "  Training Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - artifacts/models/best_model.joblib"
echo "  - artifacts/models/best_model_name.joblib"
echo "  - artifacts/evaluation/cv_results.json"
echo "  - artifacts/evaluation/test_metrics.json"
echo "  - artifacts/evaluation/plots/training/"
echo ""
echo "Next step: Run full pipeline evaluation"
echo "  bash scripts/run_pipeline.sh"
echo ""
