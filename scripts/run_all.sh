#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "  JCCD-X-V6 Full Pipeline"
echo "=========================================="
echo ""

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

echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="
echo "Results: artifacts/evaluation/"
echo "Models:  artifacts/models/"
echo "Plots:   artifacts/evaluation/plots/"
