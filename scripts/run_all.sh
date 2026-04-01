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
    echo "Running with analysis modules (Phases 8-12)"
    echo ""
fi

# Step 0: Build Zig libraries
echo "=========================================="
echo "  Step 0: Building Zig Libraries"
echo "=========================================="
zig build -Doptimize=ReleaseFast
echo ""

# Step 1: Data preparation
echo "=========================================="
echo "  Step 1: Data Preparation"
echo "=========================================="
bash scripts/run_data_prep.sh

# Step 2: Normalization
echo "=========================================="
echo "  Step 2: Source Code Normalization"
echo "=========================================="
bash scripts/run_preprocessing.sh

# Step 3: Tokenization
echo "=========================================="
echo "  Step 3: Tokenization & CST"
echo "=========================================="
bash scripts/run_tokenization.sh

# Step 4: Feature engineering
echo "=========================================="
echo "  Step 4: Feature Engineering"
echo "=========================================="
bash scripts/run_features.sh

# Step 5: ML Training
echo "=========================================="
echo "  Step 5: ML Training"
echo "=========================================="
bash scripts/run_training.sh

# Step 5.5: Feature Selection (RFE)
echo "=========================================="
echo "  Step 5.5: Feature Selection (RFE)"
echo "=========================================="
bash scripts/run_rfe.sh


# Step 6: Full Pipeline
echo "=========================================="
echo "  Step 6: Full Pipeline Evaluation"
echo "=========================================="
bash scripts/run_pipeline.sh

# Step 6.5: Ablation Study (on selected features)
echo "=========================================="
echo "  Step 6.5: Ablation Study (Selected Features)"
echo "=========================================="
bash scripts/run_ablation.sh




# Optional Analysis Modules (Phases 8-12)
if [[ "$RUN_ANALYSIS" == "true" ]]; then
    echo "=========================================="
    echo "  Running Analysis Modules"
    echo "=========================================="
    echo ""

    # Phase 8: LSH Parameter Sweep
    echo "=========================================="
    echo "  Phase 8: LSH Parameter Sweep"
    echo "=========================================="
    bash scripts/run_lsh_sweep.sh

    # Phase 9: CV Stability Analysis
    echo "=========================================="
    echo "  Phase 9: CV Stability Analysis"
    echo "=========================================="
    bash scripts/run_stability.sh

    # Phase 10: Probability Calibration
    echo "=========================================="
    echo "  Phase 10: Probability Calibration"
    echo "=========================================="
    bash scripts/run_calibration.sh

    # Phase 11: Sensitivity Analysis
    echo "=========================================="
    echo "  Phase 11: Sensitivity Analysis"
    echo "=========================================="
    bash scripts/run_sensitivity.sh

    # Phase 12: Memory Usage Profiling
    echo "=========================================="
    echo "  Phase 12: Memory Usage Profiling"
    echo "=========================================="
    bash scripts/run_memory_profile.sh

    echo "=========================================="
    echo "  Analysis Complete!"
    echo "=========================================="
    echo ""
fi

echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Metrics: artifacts/evaluation/*.json"
echo "  - Models:  artifacts/models/"
echo "  - Plots:   artifacts/evaluation/plots/"
echo ""
echo ""
echo "To run with analysis modules:"
echo "  bash scripts/run_all.sh --with-analysis"
echo "  or: RUN_ANALYSIS=true bash scripts/run_all.sh"
echo ""
