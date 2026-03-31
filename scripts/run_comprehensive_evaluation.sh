#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  JCCD-X-V6 Comprehensive Evaluation"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Default parameters
DATA_DIR="${DATA_DIR:-data/processed}"
MODEL_DIR="${MODEL_DIR:-artifacts/models}"
EVAL_DIR="${EVAL_DIR:-artifacts/evaluation_comprehensive}"
SEED="${SEED:-42}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --eval-dir)
            EVAL_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-dir DIR      Path to processed data directory (default: data/processed)"
            echo "  --model-dir DIR     Path to models directory (default: artifacts/models)"
            echo "  --eval-dir DIR      Path to evaluation output directory (default: artifacts/evaluation_comprehensive)"
            echo "  --seed N            Random seed (default: 42)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Data directory:    $DATA_DIR"
echo "  Model directory:   $MODEL_DIR"
echo "  Evaluation dir:    $EVAL_DIR"
echo "  Random seed:       $SEED"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

if [[ ! -f "$MODEL_DIR/best_model.joblib" ]]; then
    echo "ERROR: Model file not found: $MODEL_DIR/best_model.joblib"
    echo "Please run training first: bash scripts/run_training.sh"
    exit 1
fi

echo "Prerequisites OK"
echo ""

# Run quick comprehensive evaluation
echo "Running comprehensive evaluation..."
echo ""

poetry run python src/python/evaluation/quick_evaluation.py \
    --data-dir "$DATA_DIR" \
    --model-dir "$MODEL_DIR" \
    --eval-dir "$EVAL_DIR" \
    --seed "$SEED"

echo ""
echo "=========================================="
echo "  Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Metrics: $EVAL_DIR/*.json"
echo "  - Plots:   $EVAL_DIR/plots/"
echo ""
echo "Key outputs:"
echo "  - model_comparison.json           - All model metrics"
echo "  - evaluation_summary.json         - Summary of results"
echo "  - full_pipeline_metrics.json      - Full pipeline metrics"
echo "  - plots/model_comparison_f1.png           - F1 score comparison"
echo "  - plots/model_comparison_auc.png          - AUC comparison"
echo "  - plots/model_efficiency_scatter.png      - Time vs F1 scatter"
echo "  - plots/roc_curves_comparison.png         - ROC curves (all models)"
echo "  - plots/pr_curves_comparison.png          - PR curves (all models)"
echo "  - plots/threshold_sensitivity.png         - Threshold analysis"
echo "  - plots/class_distribution.png            - Dataset distribution"
echo "  - plots/class_distribution_multiclass.png - Multi-class distribution"
echo "  - plots/confusion_matrix_binary.png       - Confusion matrix"
echo "  - plots/error_analysis.png                - Error breakdown"
echo "  - plots/feature_group_contribution.png    - Feature importance"
echo "  - plots/lsh_pareto_curve.png              - LSH speed vs accuracy"
echo "  - plots/lsh_recall_vs_reduction.png       - LSH recall analysis"
echo ""
