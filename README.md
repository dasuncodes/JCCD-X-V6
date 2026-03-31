# JCCD-X-V6: Java Code Clone Detection Pipeline

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Zig 0.11+](https://img.shields.io/badge/zig-0.11+-orange.svg)](https://ziglang.org/download/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**JCCD-X-V6** is a hybrid Python + Zig pipeline for detecting Type 1-3 syntactic code clones in Java source code. It combines rule-based detection for exact matches with ML-based classification for semantic similarity, achieving **95.9% F1 score** on the TOMA dataset.

## 🎯 Key Features

- **Multi-Type Detection**: Type-1 (exact), Type-2 (renamed), Type-3 (modified statements)
- **Hybrid Architecture**: Zig for performance-critical paths, Python for ML/evaluation
- **LSH Candidate Filtering**: 50% reduction with <2% F1 loss
- **RFE Feature Selection**: 3 optimal features from 26 candidates
- **Comprehensive Evaluation**: 3-level ablation studies, cross-validation, statistical analysis
- **Publication-Ready Outputs**: JSON metrics, CSV tables, PNG visualizations

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        JCCD-X-V6 Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Data Prep  │───▶│  Normalize   │───▶│  Tokenize    │              │
│  │   (Python)   │    │    (Zig)     │    │  (Zig+TS)    │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │              Feature Engineering (Zig)                    │          │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │          │
│  │  │  Lexical   │  │ Structural │  │   Quantitative     │ │          │
│  │  │  (7 feat)  │  │  (7 feat)  │  │    (12 feat)       │ │          │
│  │  └────────────┘  └────────────┘  └────────────────────┘ │          │
│  └──────────────────────────────────────────────────────────┘          │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │              LSH Candidate Filtering (Zig)                │          │
│  │         MinHash + LSH → 50% pair reduction                │          │
│  └──────────────────────────────────────────────────────────┘          │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │           ML Classification (Python + XGBoost)            │          │
│  │         RFE-selected 3 features, 5-fold CV                │          │
│  └──────────────────────────────────────────────────────────┘          │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │        Evaluation & Ablation Studies (Python)             │          │
│  │  • Individual feature ablation                           │          │
│  │  • Feature group ablation                                │          │
│  │  • LSH parameter ablation                                │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Layer | Component | Technology | Purpose | Performance |
|-------|-----------|------------|---------|-------------|
| **Data** | Data preparation | Python (pandas) | Train/test split, validation | ~2s |
| **Preprocessing** | Normalization | Zig | Remove comments, whitespace | ~3s |
| **Analysis** | Tokenization | Zig + tree-sitter | CST generation, token IDs | ~5s |
| **Features** | Feature engineering | Zig (parallel) | 26 features per pair | ~15s |
| **Filtering** | Shingling + LSH | Zig | MinHash signatures, candidate reduction | ~5s |
| **ML** | Training | Python (XGBoost) | 5-fold CV, RFE selection | ~12s |
| **ML** | Inference | Python (XGBoost) | Type-3 classification | <1s |
| **Rules** | Type-1/2 detection | Python | Hash-based exact match | ~3s |
| **Evaluation** | Ablation studies | Python | 3-level analysis, visualizations | ~105s |

## 🚀 Quick Start

### Prerequisites

- Python 3.14+
- Zig 0.11+
- Poetry (Python package manager)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/JCCD-X-V6.git
cd JCCD-X-V6

# Install Python dependencies
poetry install

# Build Zig shared libraries
zig build -Doptimize=ReleaseFast

# Or use Make for complete setup
make setup
```

### Run Complete Pipeline

```bash
# Run all steps (recommended for first-time users)
bash scripts/run_all.sh

# Or run individual steps
bash scripts/run_data_prep.sh      # 1. Prepare dataset
bash scripts/run_preprocessing.sh  # 2. Normalize source code
bash scripts/run_tokenization.sh   # 3. Generate tokens + CST
bash scripts/run_features.sh       # 4. Compute 26 features
bash scripts/run_training.sh       # 5. Train ML models
bash scripts/run_pipeline.sh       # 6. Full pipeline evaluation
bash scripts/run_ablation_study.sh # 7. Ablation studies
```

## 📊 Results

### Model Performance (5-Fold Cross-Validation)

| Model | F1 Score | AUC | Precision | Recall | Training Time |
|-------|----------|-----|-----------|--------|---------------|
| **XGBoost** (selected) | **0.9725** | **0.9956** | 0.9681 | 0.9770 | 5.6s |
| Random Forest | 0.9523 | 0.9900 | 0.9456 | 0.9591 | 63.7s |
| KNN | 0.9264 | 0.9689 | 0.9198 | 0.9332 | 67.2s |
| Linear SVM | 0.9238 | 0.9778 | 0.9167 | 0.9310 | 16.6s |
| Logistic Regression | 0.9216 | 0.9769 | 0.9145 | 0.9288 | 86.7s |

### Full Pipeline Performance (Test Set)

| Metric | Value | Notes |
|--------|-------|-------|
| **Type-1 Detection** | 48,109 pairs | Recall: 99.98%, Precision: 100% |
| **Type-2 Detection** | 4,222 pairs | Recall: 99.65%, Precision: 100% |
| **Type-3 Detection** | 30,130 pairs | Recall: 93.24%, Precision: 82.61% |
| **Overall Accuracy** | 94.28% | On balanced test set |
| **Overall F1 Score** | 87.61% | Macro-averaged |
| **LSH Reduction** | 45.7% | 149k → 73k pairs |
| **Total Runtime** | 39.0s | End-to-end pipeline |

### Feature Importance (RFE-Selected)

After Recursive Feature Elimination with 5-fold CV:

| Rank | Feature | Group | ΔF1 if Removed | Importance |
|------|---------|-------|----------------|------------|
| 1 | `lex_dice_tokens` | Lexical | **-8.64%** | 🔴 Critical |
| 2 | `struct_node_count2` | Structural | -2.28% | 🟡 Important |
| 3 | `quant_cc2` | Quantitative | -0.94% | 🟢 Optional |

### Ablation Study Results

#### Individual Feature Ablation (Test Set)

| Configuration | F1 Score | Accuracy | Δ F1 (%) |
|--------------|----------|----------|----------|
| **Full (All 3 features)** | **0.9590** | **0.9587** | — |
| Without lex_dice_tokens | 0.8761 | 0.8684 | **-8.64%** |
| Without struct_node_count2 | 0.9371 | 0.9369 | -2.28% |
| Without quant_cc2 | 0.9500 | 0.9496 | -0.94% |
| Rule-Based Only | 0.8152 | 0.8628 | -15.00% |

#### LSH Parameter Trade-offs

| Configuration | F1 Score | Reduction | Pairs | Time | Speedup |
|--------------|----------|-----------|-------|------|---------|
| **No LSH** | 0.9590 | 0% | 150,820 | 7.38s | 1.0× |
| Aggressive | 0.9091 | 80% | 30,163 | 2.25s | **3.3×** |
| Balanced | 0.9410 | 50% | 75,410 | 4.12s | 1.8× |
| Conservative | 0.9501 | 20% | 120,656 | 5.95s | 1.2× |

## 📁 Project Structure

```
JCCD-X-V6/
├── data/
│   ├── raw/toma/                  # Raw TOMA dataset
│   ├── processed/                 # Train/test splits, normalized files
│   │   ├── training_dataset.csv   # Balanced training pairs
│   │   ├── testing_dataset.csv    # Test pairs with labels
│   │   └── normalized/            # Normalized Java files
│   └── intermediate/              # Intermediate artifacts
│       ├── token_data.pkl         # Token sequences + AST data
│       └── features/              # Computed features
│           ├── train_features.csv # Training features (26 cols)
│           └── test_features.csv  # Test features (26 cols)
│
├── src/
│   ├── python/
│   │   ├── preprocessing/         # Data prep, normalization wrapper
│   │   ├── feature_engineering/   # Feature computation (Python side)
│   │   ├── model/                 # ML training, RFE, models
│   │   │   ├── train.py           # Training script
│   │   │   ├── models.py          # Model definitions
│   │   │   └── feature_selection.py # RFE implementation
│   │   ├── pipeline/              # Full pipeline orchestration
│   │   │   └── full_pipeline.py   # End-to-end evaluation
│   │   ├── evaluation/            # Metrics, plots, ablation
│   │   │   ├── metrics.py         # Metric computation
│   │   │   ├── plots.py           # Visualization functions
│   │   │   ├── ablation.py        # Ablation study module
│   │   │   └── run_ablation_study.py # Comprehensive ablation runner
│   │   └── utils/                 # I/O utilities
│   │
│   ├── zig/
│   │   ├── preprocessing/         # Normalization (comments, whitespace)
│   │   ├── tokenization/          # Token abstraction, CST generation
│   │   ├── features/              # Batch feature computation (parallel)
│   │   ├── ast_stats/             # AST statistics extraction
│   │   ├── shingling/             # K-gram shingling
│   │   └── lsh/                   # MinHash + LSH bucketing
│   │
│   └── bindings/                  # Python ↔ Zig FFI wrappers
│       ├── features.py            # Feature computation bindings
│       ├── shingling.py           # Shingling bindings
│       └── minhash.py             # MinHash bindings
│
├── tests/                         # Unit + integration tests
│   ├── test_preprocessing.py      # Normalization tests
│   ├── test_features.py           # Feature computation tests
│   ├── test_evaluation.py         # Metrics tests
│   └── test_ablation.py           # Ablation study tests
│
├── scripts/                       # Shell scripts for each step
│   ├── run_data_prep.sh
│   ├── run_preprocessing.sh
│   ├── run_tokenization.sh
│   ├── run_features.sh
│   ├── run_training.sh
│   ├── run_pipeline.sh
│   ├── run_ablation_study.sh      # NEW: Comprehensive ablation
│   ├── run_all.sh
│   └── test_all.sh
│
├── artifacts/                     # Generated outputs
│   ├── models/                    # Trained models
│   │   ├── best_model.joblib      # XGBoost model
│   │   ├── best_model_rfe.joblib  # RFE-optimized model
│   │   ├── best_model_name.joblib # Model identifier
│   │   └── selected_features.json # RFE-selected features
│   └── evaluation/                # Metrics + plots
│       ├── cv_results.json        # Cross-validation results
│       ├── pipeline_metrics.json  # Pipeline evaluation
│       ├── ablation_study/        # Ablation study outputs
│       │   ├── individual_feature_ablation_results.json
│       │   ├── feature_group_ablation_results.json
│       │   ├── lsh_ablation_results.json
│       │   ├── combined_ablation_results.json
│       │   ├── *_summary.csv      # Summary tables
│       │   ├── plots/             # Combined 4-panel plots
│       │   └── individual_feature_plots/ # Individual plots
│       └── plots/                 # Visualization PNGs
│
├── build.zig                      # Zig build configuration
├── Makefile                       # Build automation
├── pyproject.toml                 # Python project config
└── README.md                      # This file
```

## 🔬 Ablation Studies

The pipeline includes comprehensive 3-level ablation analysis:

### Level 1: Individual Feature Ablation
- Removes each feature one-by-one
- Identifies most critical features
- **Output**: Feature importance ranking with ΔF1 scores

### Level 2: Feature Group Ablation
- Removes entire feature groups (lexical, structural, quantitative)
- Shows contribution of each group
- **Output**: Group-wise contribution analysis

### Level 3: LSH Parameter Ablation
- Varies LSH filtering aggressiveness
- Analyzes efficiency vs accuracy trade-off
- **Output**: Pareto frontier for deployment decisions

### Run Ablation Studies

```bash
# Run all ablation studies (uses test data by default)
bash scripts/run_ablation_study.sh

# Output directory
artifacts/evaluation/ablation_study/
├── individual_feature_ablation_results.json
├── feature_group_ablation_results.json
├── lsh_ablation_results.json
├── combined_ablation_results.json
├── *_summary.csv              # Summary tables
├── plots/                     # Combined 4-panel figures
│   ├── individual_feature_ablation_comparison.png
│   ├── feature_group_ablation_comparison.png
│   └── lsh_ablation_comparison.png
└── individual_feature_plots/  # Individual plots
    ├── 01_f1_score_comparison.png
    ├── 02_f1_degradation.png
    ├── 03_feature_ranking.png
    └── 04_baseline_comparison.png
```

## 🧪 Testing

```bash
# Run all tests (Zig + Python + lint)
bash scripts/test_all.sh

# Python tests only
poetry run pytest tests/ -v

# Zig tests only
zig build test

# Lint Python code
poetry run ruff check src/python/ tests/
```

## 📚 Dataset

Source: **TOMA Dataset** (Table Of Multiple Applications)

| File | Rows | Description |
|------|------|-------------|
| type-1.csv | 48,116 | Type-1 clones (identical after normalization) |
| type-2.csv | 4,234 | Type-2 clones (renamed identifiers/literals) |
| type-3.csv | 21,395 | Type-3 clones (modified statements) |
| type-4.csv | 86,341 | Type-4 clones (treated as Type-3) |
| type-5.csv | 109,914 | Type-5 clones (NON - treated as non-clones) |
| nonclone.csv | 279,032 | Non-clones (NON) |
| id2sourcecode/ | 73,319 | Java source files |

**Data Split:**
- **Training**: 70% of (type-3 + type-4 + type-5 + nonclone), balanced
- **Testing**: 30% + 100% of type-1 + type-2

## 📈 Metrics & Outputs

Each pipeline step produces:
- **Metrics**: `artifacts/evaluation/{step}_metrics.json`
- **Plots**: `artifacts/evaluation/plots/{step}/`

### Key Output Files

| File | Description |
|------|-------------|
| `data/processed/training_dataset.csv` | Balanced training pairs (150k) |
| `data/processed/testing_dataset.csv` | Test pairs with labels (201k) |
| `data/intermediate/features/train_features.csv` | 26 features per training pair |
| `data/intermediate/features/test_features.csv` | 26 features per test pair |
| `artifacts/models/best_model.joblib` | Trained XGBoost model |
| `artifacts/models/selected_features.json` | RFE-selected 3 features |
| `artifacts/evaluation/cv_results.json` | 5-fold CV results (all models) |
| `artifacts/evaluation/pipeline_metrics.json` | Full pipeline evaluation |
| `artifacts/evaluation/ablation_study/` | Comprehensive ablation results |

## 📖 Documentation

- **[ABLATION_STUDY_QUICK_REF.md](ABLATION_STUDY_QUICK_REF.md)** - Quick start guide
- **[ABLATION_STUDY_COMPLETE_GUIDE.md](ABLATION_STUDY_COMPLETE_GUIDE.md)** - Comprehensive paper guide with LaTeX tables
- **[LSH_OPTIMIZATION_RESULTS.md](LSH_OPTIMIZATION_RESULTS.md)** - LSH parameter analysis
- **[LSH_PERFORMANCE_ANALYSIS.md](LSH_PERFORMANCE_ANALYSIS.md)** - Runtime analysis
- **[PIPELINE_FIX_SUMMARY.md](PIPELINE_FIX_SUMMARY.md)** - Pipeline bug fixes

## 🔧 Configuration

### LSH Parameters

```bash
# Default (balanced)
--lsh-hashes 48 --lsh-bands 16

# For higher recall
--lsh-hashes 72 --lsh-bands 24

# For faster processing
--lsh-hashes 36 --lsh-bands 12

# Skip LSH entirely
--skip-lsh
```

### Classification Threshold

```bash
# Default threshold
--threshold 0.30

# For higher recall (more positives)
--threshold 0.20

# For higher precision (fewer positives)
--threshold 0.40
```

## 🎓 Research Use

### Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{jccd-x-v6,
  title = {JCCD-X-V6: Java Code Clone Detection Pipeline},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/your-org/JCCD-X-V6}
}
```

### Key Findings for Papers

> "Our ablation study reveals that lexical features (Dice coefficient on token sets) contribute most significantly to clone detection performance, with an 8.64% F1 score degradation when removed. Structural features (AST node counts) provide moderate improvement (2.28%), while quantitative features (cyclomatic complexity) contribute marginally (0.94%). The ML classifier improves F1 score by 14.38% compared to rule-based detection alone."

> "For LSH filtering, a balanced configuration (50% reduction) achieves 1.8× speedup with only 1.88% F1 degradation, making it suitable for production deployment. Aggressive filtering (80%) provides 3.3× speedup at the cost of 5.21% F1 loss."

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📊 Performance Summary

| Aspect | Metric | Value |
|--------|--------|-------|
| **Detection** | Type-1 Recall | 99.98% |
| **Detection** | Type-2 Recall | 99.65% |
| **Detection** | Type-3 Recall | 93.24% |
| **ML Model** | XGBoost F1 | 97.25% |
| **Features** | RFE-Selected | 3 features |
| **LSH** | Optimal Reduction | 50% |
| **Pipeline** | Total Runtime | 39s |
| **Evaluation** | Ablation Studies | 3 levels |

---

**Last Updated**: March 31, 2026  
**Version**: 6.0.0  
**Maintained by**: Research Team
