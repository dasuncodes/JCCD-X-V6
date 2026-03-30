# JCCD-X-V6: Java Code Clone Detection Pipeline

Detect Type 1-3 syntactic code clones using a Python + Zig hybrid pipeline.

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data processing, normalization, tokenization | Zig | Fast I/O and computation |
| AST statistics, feature engineering | Zig (parallelized) | Batch pairwise feature computation |
| Shingling, MinHash, LSH | Zig | Candidate reduction |
| ML training, evaluation, visualization | Python (scikit-learn, XGBoost) | Model selection and analysis |
| Full pipeline orchestration | Python + Zig | End-to-end clone detection |

## Setup

```bash
# Install Python dependencies
poetry install

# Build Zig shared libraries
zig build -Doptimize=ReleaseFast

# Or use Make
make setup
```

## Pipeline Steps

Run individually:

```bash
# 1. Data preparation (train/test split, validation)
bash scripts/run_data_prep.sh

# 2. Source code normalization (comments, whitespace)
bash scripts/run_preprocessing.sh

# 3. Tokenization & CST generation (tree-sitter)
bash scripts/run_tokenization.sh

# 4. Feature engineering (26 features per pair, Zig-parallelized)
bash scripts/run_features.sh

# 5. ML training (5 models, 5-fold CV)
bash scripts/run_training.sh

# 6. Full pipeline evaluation (Type-1/2 rules + ML classifier)
bash scripts/run_pipeline.sh
```

Or run everything at once:

```bash
bash scripts/run_all.sh
```

## Testing

```bash
# All tests (Zig + Python + lint)
bash scripts/test_all.sh

# Python tests only
poetry run pytest tests/ -v

# Zig tests only
zig build test

# Lint
poetry run ruff check src/python/ tests/
```

## Dataset

Source: TOMA dataset in `data/raw/toma/`

| File | Rows | Description |
|------|------|-------------|
| type-1.csv | 48,116 | Type-1 clones (identical) |
| type-2.csv | 4,234 | Type-2 clones (parameterized) |
| type-3.csv | 21,395 | Type-3 clones |
| type-4.csv | 86,341 | Type-4 clones (treated as Type-3) |
| type-5.csv | 109,914 | Type-5 clones (NON) |
| nonclone.csv | 279,032 | Non-clones (NON) |
| id2sourcecode/ | 73,319 | Java source files |

**Training set:** 70% of (type-3 + type-4 + type-5 + nonclone), balanced  
**Testing set:** 30% of abc + type-1 + type-2

## Results

### Model Performance (5-Fold CV)

| Model | F1 | AUC | Time |
|-------|-----|------|------|
| **XGBoost** | **0.9725** | **0.9956** | 5.6s |
| Random Forest | 0.9523 | 0.9900 | 63.7s |
| KNN | 0.9264 | 0.9689 | 67.2s |
| Linear SVM | 0.9238 | 0.9778 | 16.6s |
| Logistic Regression | 0.9216 | 0.9769 | 86.7s |

### Full Pipeline Results

| Metric | Value |
|--------|-------|
| Type-1 detected | 52,331 |
| Accuracy | 96.9% |
| Precision | 88.7% |
| Recall | 98.2% |
| F1 | 93.2% |
| LSH reduction | 85.8% |
| Total runtime | 37s |

### Top Features (XGBoost importance)

1. `lex_lcs_ratio` (0.653)
2. `struct_node_count2` (0.052)
3. `lex_levenshtein_ratio` (0.045)
4. `struct_depth_ratio` (0.030)
5. `quant_cc2` (0.030)

## Project Structure

```
jccd-x/
├── data/
│   ├── raw/toma/              # Raw dataset
│   ├── processed/             # Training/testing CSVs, normalized files
│   └── intermediate/          # Token data, features
│
├── src/
│   ├── python/
│   │   ├── preprocessing/     # Data prep, normalization, tokenizer
│   │   ├── feature_engineering/  # Feature computation wrapper
│   │   ├── model/             # ML training, model definitions
│   │   ├── pipeline/          # Full pipeline orchestration
│   │   └── utils/             # I/O utilities
│   │
│   ├── zig/
│   │   ├── preprocessing/     # Normalization (comment/whitespace removal)
│   │   ├── tokenization/      # Token abstraction and encoding
│   │   ├── features/          # Batch feature computation (parallelized)
│   │   ├── ast_stats/         # AST statistics
│   │   ├── shingling/         # K-gram shingling
│   │   └── lsh/               # MinHash signatures and LSH
│   │
│   └── bindings/              # Python ↔ Zig ctypes wrappers
│
├── tests/                     # Python + Zig unit tests
├── scripts/                   # Shell scripts for each pipeline step
├── artifacts/
│   ├── models/                # Trained models (joblib)
│   └── evaluation/            # Metrics JSON + plots
│
├── build.zig                  # Zig build configuration
├── Makefile                   # Build automation
└── pyproject.toml             # Python project config
```

## Metrics & Outputs

Each pipeline step produces:

- **Metrics:** `artifacts/evaluation/{step}_metrics.json`
- **Plots:** `artifacts/evaluation/plots/{step}/`

Key output files:
- `data/processed/training_dataset.csv` — Balanced training pairs
- `data/processed/testing_dataset.csv` — Test pairs with type-1/2
- `data/intermediate/features/train_features.csv` — 26 features per training pair
- `data/intermediate/features/test_features.csv` — 26 features per test pair
- `artifacts/models/best_model.joblib` — Best trained model (XGBoost)
- `artifacts/evaluation/cv_results.json` — Cross-validation results for all models
- `artifacts/evaluation/pipeline_metrics.json` — Full pipeline evaluation
