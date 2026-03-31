#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  JCCD-X-V6 Quick Test (Limited Samples)"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""
echo "This script runs the full pipeline with a limited"
echo "number of samples for quick testing and validation."
echo ""

# Configuration
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"  # Number of samples per class
echo "Sample size per class: $SAMPLE_SIZE"
echo ""

# Step 1: Create limited datasets
echo "=== Step 1: Creating Limited Datasets ==="
poetry run python << PYTHON_SCRIPT
import pandas as pd
import numpy as np

# Load full datasets
train_full = pd.read_csv('data/processed/training_dataset.csv')
test_full = pd.read_csv('data/processed/testing_dataset.csv')

print(f"Full training set: {len(train_full)} samples")
print(f"Full test set: {len(test_full)} samples")

# Sample training set (balanced)
train_0 = train_full[train_full['label'] == 0].sample(n=min($SAMPLE_SIZE, len(train_full[train_full['label'] == 0])), random_state=42)
train_1 = train_full[train_full['label'] == 1].sample(n=min($SAMPLE_SIZE, len(train_full[train_full['label'] == 1])), random_state=42)
train_limited = pd.concat([train_0, train_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# Sample test set (preserve distribution but limit)
test_limited = test_full.sample(n=min($SAMPLE_SIZE * 5, len(test_full)), random_state=42).reset_index(drop=True)

# Save limited datasets
train_limited.to_csv('data/processed/training_dataset_limited.csv', index=False)
test_limited.to_csv('data/processed/testing_dataset_limited.csv', index=False)

print(f"Limited training set: {len(train_limited)} samples")
print(f"Limited test set: {len(test_limited)} samples")
print("Saved limited datasets")
PYTHON_SCRIPT
echo ""

# Step 2: Run feature engineering on limited data
echo "=== Step 2: Feature Engineering (Limited) ==="
poetry run python -m src.python.feature_engineering.features \
    --data-dir data/processed \
    --intermediate-dir data/intermediate \
    --output-dir data/intermediate/features_limited \
    --eval-dir artifacts/evaluation \
    --source-dir data/raw/toma/id2sourcecode
echo ""

# Step 3: Run training on limited data
echo "=== Step 3: ML Training (Limited) ==="
poetry run python << PYTHON_SCRIPT
import argparse
import sys
sys.path.append('src/python')

from src.python.model.train import main as train_main

# Mock arguments for limited training
class Args:
    features_dir = type('obj', (object,), {'__truediv__': lambda self, x: f'data/intermediate/features_limited/{x}'})()
    model_dir = type('obj', (object,), {'__truediv__': lambda self, x: f'artifacts/models_limited/{x}'})()
    eval_dir = type('obj', (object,), {'__truediv__': lambda self, x: f'artifacts/evaluation_limited/{x}'})()
    seed = 42

import sys
sys.argv = ['train.py',
            '--features-dir', 'data/intermediate/features_limited',
            '--model-dir', 'artifacts/models_limited',
            '--eval-dir', 'artifacts/evaluation_limited',
            '--seed', '42']

from src.python.model.train import main
main()
PYTHON_SCRIPT
echo ""

# Step 4: Run pipeline evaluation on limited data
echo "=== Step 4: Pipeline Evaluation (Limited) ==="
poetry run python << PYTHON_SCRIPT
import sys
sys.argv = ['full_pipeline.py',
            '--data-dir', 'data/processed',
            '--intermediate-dir', 'data/intermediate',
            '--model-dir', 'artifacts/models_limited',
            '--eval-dir', 'artifacts/evaluation_limited']

from src.python.pipeline.full_pipeline import main
main()
PYTHON_SCRIPT
echo ""

echo "=========================================="
echo "  Quick Test Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Limited training: data/processed/training_dataset_limited.csv"
echo "  - Limited features: data/intermediate/features_limited/"
echo "  - Limited models:  artifacts/models_limited/"
echo "  - Limited metrics: artifacts/evaluation_limited/"
echo ""
echo "To run with different sample size:"
echo "  SAMPLE_SIZE=500 bash scripts/run_quick_test.sh"
echo ""
