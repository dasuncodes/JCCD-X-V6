#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  JCCD-X-V6 Full Pipeline"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Parse arguments
RUN_ANALYSIS="${RUN_ANALYSIS:-false}"
if [[ "${1:-}" == "--with-analysis" ]] || [[ "${RUN_ANALYSIS}" == "true" ]]; then
    RUN_ANALYSIS=true
    echo "Running with analysis modules (Phases 7-11)"
    echo ""
fi

# Step 1: Build Zig libraries
echo "=== Step 0: Building Zig libraries ==="
zig build -Doptimize=ReleaseFast
echo ""

# Step 2: Data preparation
echo "=== Step 1: Data Preparation ==="
poetry run python -m src.python.preprocessing.data_prep \
    --source-dir data/raw/toma/id2sourcecode \
    --raw-dir data/raw/toma \
    --output-dir data/processed \
    --eval-dir artifacts/evaluation
echo ""

# Step 3: Normalization
echo "=== Step 2: Source Code Normalization ==="
poetry run python -m src.python.preprocessing.normalization \
    --source-dir data/raw/toma/id2sourcecode \
    --data-dir data/processed \
    --output-dir data/processed/normalized \
    --eval-dir artifacts/evaluation
echo ""

# Step 4: Tokenization
echo "=== Step 3: Tokenization & CST ==="
poetry run python -m src.python.preprocessing.tokenizer \
    --source-dir data/raw/toma/id2sourcecode \
    --normalized-dir data/processed/normalized \
    --data-dir data/processed \
    --output-dir data/intermediate \
    --eval-dir artifacts/evaluation
echo ""

# Step 5: Feature engineering
echo "=== Step 4: Feature Engineering ==="
poetry run python -m src.python.feature_engineering.features \
    --data-dir data/processed \
    --intermediate-dir data/intermediate \
    --output-dir data/intermediate/features \
    --eval-dir artifacts/evaluation
echo ""

# Step 6: ML Training
echo "=== Step 5: ML Training ==="
poetry run python -m src.python.model.train \
    --features-dir data/intermediate/features \
    --model-dir artifacts/models \
    --eval-dir artifacts/evaluation
echo ""

# Step 7: Full Pipeline
echo "=== Step 6: Full Pipeline Evaluation ==="
poetry run python -m src.python.pipeline.full_pipeline \
    --data-dir data/processed \
    --intermediate-dir data/intermediate \
    --model-dir artifacts/models \
    --eval-dir artifacts/evaluation
echo ""

# Optional Analysis Modules (Phases 7-11)
if [[ "$RUN_ANALYSIS" == "true" ]]; then
    echo "=========================================="
    echo "  Running Analysis Modules"
    echo "=========================================="
    echo ""

    # Phase 7: LSH Parameter Sweep
    echo "=== Step 7: LSH Parameter Sweep ==="
    poetry run python -m src.python.pipeline.lsh_tuning \
        --features-dir data/intermediate/features \
        --data-dir data/processed \
        --output-dir artifacts/evaluation \
        --hash-range 64 256 64 \
        --bands-range 8 32 8 \
        --shingle-k-range 3 5 1
    echo ""

    # Phase 8: CV Stability Analysis
    echo "=== Step 8: CV Stability Analysis ==="
    poetry run python -m src.python.evaluation.stability \
        --features-dir data/intermediate/features \
        --output-dir artifacts/evaluation \
        --model-name xgboost \
        --cv 5
    echo ""

    # Phase 9: Probability Calibration
    echo "=== Step 9: Probability Calibration ==="
    poetry run python -m src.python.model.calibration \
        --features-dir data/intermediate/features \
        --model-dir artifacts/models \
        --output-dir artifacts/evaluation \
        --model-name xgboost \
        --cv 5
    echo ""

    # Phase 10: Sensitivity Analysis
    echo "=== Step 10: Sensitivity Analysis ==="
    poetry run python -m src.python.evaluation.sensitivity \
        --features-dir data/intermediate/features \
        --output-dir artifacts/evaluation \
        --model-name xgboost \
        --perturbation-types noise mask shift \
        --perturbation-levels 0.05 0.10 0.15 0.20 0.25 \
        --cv 5
    echo ""

    # Phase 11: Memory Usage Profiling
    echo "=== Step 11: Memory Usage Profiling ==="
    poetry run python -m src.python.utils.memory \
        --features-dir data/intermediate/features \
        --data-dir data/processed \
        --model-dir artifacts/models \
        --output-dir artifacts/evaluation \
        --cv 5
    echo ""

    echo "=========================================="
    echo "  Analysis Complete!"
    echo "=========================================="
    echo ""
fi

echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="
echo "Results: artifacts/evaluation/"
echo "Models:  artifacts/models/"
echo "Plots:   artifacts/evaluation/plots/"
echo ""
echo "To run with analysis modules:"
echo "  bash scripts/run_all.sh --with-analysis"
echo "  or: RUN_ANALYSIS=true bash scripts/run_all.sh"
