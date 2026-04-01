# System Architecture

This document provides a comprehensive overview of the JCCD-X-V6 system architecture, including component breakdown, technology stack, and data flow.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Component Architecture](#component-architecture)
3. [Technology Stack](#technology-stack)
4. [Data Flow](#data-flow)
5. [Module Breakdown](#module-breakdown)
6. [Build System](#build-system)
7. [Performance Characteristics](#performance-characteristics)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        JCCD-X-V6 Pipeline                                │
│              Java Code Clone Detection (Type 1-3)                        │
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

---

## Component Architecture

### Layer 1: Data Layer

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  Raw Data       │    │  Processed Data │                │
│  │  (TOMA)         │    │  (CSV/Pickle)   │                │
│  │                 │    │                 │                │
│  │  • type-1.csv   │    │  • train.csv    │                │
│  │  • type-2.csv   │    │  • test.csv     │                │
│  │  • type-3.csv   │    │  • features.csv │                │
│  │  • type-4.csv   │    │  • labels.pkl   │                │
│  │  • nonclone.csv │    │                 │                │
│  └─────────────────┘    └─────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Layer 2: Preprocessing Layer

```
┌─────────────────────────────────────────────────────────────┐
│                  Preprocessing Layer                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Normalization (Zig)                      │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │  Remove    │  │  Remove    │  │  Normalize │     │  │
│  │  │  Comments  │  │ Whitespace │  │  Strings   │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Tokenization (Zig + tree-sitter)           │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │   Lexer    │  │   Parser   │  │   CST      │     │  │
│  │  │            │  │            │  │  Builder   │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Layer 3: Feature Engineering Layer

```
┌─────────────────────────────────────────────────────────────┐
│                Feature Engineering Layer                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Lexical Features (7 features)               │  │
│  │  • lex_dice_tokens    • lex_jaccard_tokens            │  │
│  │  • lex_overlap_tokens • lex_dice_identifiers          │  │
│  │  • lex_jaccard_ids    • lex_overlap_ids               │  │
│  │  • lex_common_ids     • (1 more)                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Structural Features (7 features)             │  │
│  │  • struct_node_count1  • struct_node_count2          │  │
│  │  • struct_depth1       • struct_depth2               │  │
│  │  • struct_branching1   • struct_branching2           │  │
│  │  • struct_similarity   • (1 more)                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Quantitative Features (12 features)           │  │
│  │  • quant_cc1           • quant_cc2                   │  │
│  │  • quant_lines1        • quant_lines2                │  │
│  │  • quant_methods1      • quant_methods2              │  │
│  │  • quant_params1       • quant_params2               │  │
│  │  • quant_variables1    • quant_variables2            │  │
│  │  • quant_complexity1   • quant_complexity2           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Layer 4: Filtering Layer

```
┌─────────────────────────────────────────────────────────────┐
│                   Filtering Layer (LSH)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Shingling (K-gram)                       │  │
│  │         Token sequences → K-grams                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              MinHash Signatures                       │  │
│  │    128 hash functions → 128-byte signature            │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           LSH Bucketing (Bands/Rows)                  │  │
│  │    Bands: 12, Rows: 3 → Candidate pairs               │  │
│  │    Reduction: 50% with <2% F1 loss                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Layer 5: Classification Layer

```
┌─────────────────────────────────────────────────────────────┐
│                  Classification Layer                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Rule-Based Detection (Type 1-2)             │  │
│  │  • Hash-based exact matching                          │  │
│  │  • Recall: 99.98% (Type-1), 99.65% (Type-2)           │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         ML Classification (Type-3)                    │  │
│  │  • Model: XGBoost (RFE-selected 3 features)           │  │
│  │  • Training: 5-fold cross-validation                  │  │
│  │  • F1 Score: 91.96%, AUC: 99.32%                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Layer 6: Evaluation Layer

```
┌─────────────────────────────────────────────────────────────┐
│                   Evaluation Layer                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Metrics Computation                      │  │
│  │  • Accuracy, Precision, Recall, F1, AUC               │  │
│  │  • Confusion Matrix, ROC/PR Curves                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Ablation Studies (3 Levels)                 │  │
│  │  • Level 1: Individual Feature Ablation               │  │
│  │  • Level 2: Feature Group Ablation                    │  │
│  │  • Level 3: LSH Parameter Ablation                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Visualization & Reporting                   │  │
│  │  • PNG plots, JSON metrics, CSV tables                │  │
│  │  • Publication-ready outputs                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Core Technologies

| Technology | Version | Purpose | Usage |
|------------|---------|---------|-------|
| **Python** | 3.10-3.14 | Main pipeline orchestration, ML, evaluation | Data prep, training, evaluation, ablation |
| **Zig** | 0.11+ | Performance-critical preprocessing, features | Normalization, tokenization, features, LSH |
| **tree-sitter** | 0.23 | Java parsing, CST generation | Tokenization, structural analysis |

### Python Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **pandas** | ^2.0 | Data manipulation, CSV processing |
| **numpy** | ^1.24 | Numerical computations |
| **scikit-learn** | ^1.3 | ML utilities, metrics, preprocessing |
| **xgboost** | ^2.0 | Primary ML classifier |
| **lightgbm** | ^4.0 | Alternative gradient boosting |
| **matplotlib** | ^3.7 | Static visualizations |
| **seaborn** | ^0.12 | Statistical visualizations |
| **plotly** | ^5.15 | Interactive visualizations |
| **tree-sitter** | ^0.23 | Python bindings for parsing |
| **tree-sitter-java** | ^0.23 | Java grammar for tree-sitter |
| **rapidfuzz** | ^3.0 | Fast string similarity (fuzzy matching) |
| **cffi** | ^1.15 | Foreign function interface (Zig ↔ Python) |
| **jupyter** | ^1.0 | Interactive notebooks |
| **joblib** | ^1.3 | Model serialization, parallel processing |

### Development Tools

| Tool | Version | Purpose |
|------|---------|---------|
| **Poetry** | Latest | Python dependency management |
| **pytest** | ^7.4 | Python unit testing |
| **pytest-cov** | ^4.1 | Test coverage reporting |
| **ruff** | ^0.1 | Python linting, formatting |
| **Zig Build** | 0.11+ | Zig compilation, testing |

### Build & Automation

| Tool | Purpose |
|------|---------|
| **Make** | Build automation, setup scripts |
| **Bash** | Pipeline orchestration scripts |
| **zig build** | Zig library compilation |

---

## Data Flow

### End-to-End Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Data Flow Diagram                                 │
└──────────────────────────────────────────────────────────────────────────┘

     ┌─────────────┐
     │  Raw Java   │
     │  Source     │
     │  Files      │
     │  (73,319)   │
     └──────┬──────┘
            │
            ▼
     ┌─────────────┐
     │  Data Prep  │  Python (pandas)
     │  (Split)    │  • Train/test split (70/30)
     │             │  • Balance classes
     └──────┬──────┘
            │
            ▼
     ┌─────────────┐
     │ Normalize   │  Zig
     │ (Comments,  │  • Remove comments
     │ whitespace) │  • Normalize whitespace
     └──────┬──────┘
            │
            ▼
     ┌─────────────┐
     │  Tokenize   │  Zig + tree-sitter
     │  (CST +     │  • Parse Java → CST
     │   Tokens)   │  • Extract token IDs
     └──────┬──────┘
            │
            ▼
     ┌─────────────┐
     │  Features   │  Zig (parallel)
     │  (26 feat)  │  • Lexical (7)
     │             │  • Structural (7)
     │             │  • Quantitative (12)
     └──────┬──────┘
            │
            ▼
     ┌─────────────┐
     │  Shingling  │  Zig
     │  (K-gram)   │  • Token k-grams
     │             │  • MinHash signatures
     └──────┬──────┘
            │
            ▼
     ┌─────────────┐
     │     LSH     │  Zig
     │  (Filter)   │  • Band/row bucketing
     │  → 50%      │  • Candidate reduction
     └──────┬──────┘
            │
            ▼
     ┌─────────────┐
     │  Rule-Based │  Python
     │  (Type 1-2) │  • Hash matching
     │             │  • 99.98% recall
     └──────┬──────┘
            │
            ▼
     ┌─────────────┐
     │  XGBoost    │  Python (scikit-learn)
     │  (Type-3)   │  • 3 features (RFE)
     │             │  • 5-fold CV
     └──────┬──────┘
            │
            ▼
     ┌─────────────┐
     │ Evaluation  │  Python
     │  + Ablation │  • Metrics, plots
     │             │  • 3-level ablation
     └─────────────┘
```

### Intermediate Artifacts

| Stage | Input | Output | Format |
|-------|-------|--------|--------|
| Data Prep | Raw TOMA CSV | `training_dataset.csv`, `testing_dataset.csv` | CSV |
| Normalization | Java source files | Normalized `.java` files | Text |
| Tokenization | Normalized files | `token_data.pkl` | Pickle |
| Features | Token data | `train_features.csv`, `test_features.csv` | CSV |
| LSH | Feature pairs | Filtered candidate pairs | In-memory |
| Training | Features + labels | `best_model.joblib` | Joblib |
| Evaluation | Model + test data | `pipeline_metrics.json` | JSON |

---

## Module Breakdown

### Python Modules

```
src/python/
├── preprocessing/
│   ├── data_prep.py          # Train/test split, balancing
│   └── normalization_wrapper.py  # Zig normalization FFI
│
├── feature_engineering/
│   ├── feature_extractor.py  # Feature computation wrapper
│   └── feature_validator.py  # Feature quality checks
│
├── model/
│   ├── train.py              # Model training orchestration
│   ├── models.py             # Model definitions (XGBoost, RF, etc.)
│   ├── feature_selection.py  # RFE implementation
│   └── calibration.py        # Probability calibration
│
├── pipeline/
│   └── full_pipeline.py      # End-to-end pipeline
│
├── evaluation/
│   ├── metrics.py            # Metric computation
│   ├── plots.py              # Visualization functions
│   ├── ablation.py           # Ablation study module
│   ├── comprehensive_evaluation.py  # Full evaluation suite
│   ├── sensitivity.py        # Sensitivity analysis
│   └── stability.py          # Stability analysis
│
└── utils/
    ├── io.py                 # File I/O utilities
    └── config.py             # Configuration management
```

### Zig Modules

```
src/zig/
├── preprocessing/
│   └── normalization.zig     # Comment/whitespace removal
│
├── tokenization/
│   └── tokenizer.zig         # CST generation, token extraction
│
├── features/
│   └── features.zig          # 26 feature computation (parallel)
│
├── ast_stats/
│   └── ast_stats.zig         # AST statistics extraction
│
├── shingling/
│   └── shingling.zig         # K-gram shingling
│
├── lsh/
│   └── minhash.zig           # MinHash + LSH bucketing
│
└── utils/
    └── helpers.zig           # Utility functions
```

### Python ↔ Zig Bindings

```
src/bindings/
├── features.py               # Feature computation FFI
├── shingling.py              # Shingling FFI
├── minhash.py                # MinHash FFI
├── normalization.py          # Normalization FFI
└── tokenizer.py              # Tokenization FFI
```

---

## Build System

### Zig Build Configuration

```zig
// build.zig
Components:
├── shingling (library)       # K-gram shingling
├── minhash (library)         # MinHash + LSH
├── ast_stats (library)       # AST statistics
├── normalization (library)   # Code normalization
├── tokenizer (library)       # Tokenization
└── features (library)        # Feature engineering
```

### Python Build Configuration

```toml
# pyproject.toml
[tool.poetry]
name = "jccd-x"
version = "0.1.0"
dependencies = [
    "pandas ^2.0",
    "numpy ^1.24",
    "scikit-learn ^1.3",
    "xgboost ^2.0",
    ...
]
```

### Makefile Targets

```makefile
# Makefile
Targets:
├── setup          # Install dependencies, build Zig
├── build          # Build Zig libraries
├── test           # Run all tests
├── lint           # Lint Python code
├── clean          # Clean build artifacts
└── run-all        # Execute complete pipeline
```

---

## Performance Characteristics

### Component Performance

| Component | Technology | Runtime | Optimization |
|-----------|------------|---------|--------------|
| Data Preparation | Python (pandas) | ~2s | Vectorized operations |
| Normalization | Zig | ~3s | Parallel processing |
| Tokenization | Zig + tree-sitter | ~5s | Incremental parsing |
| Feature Engineering | Zig (parallel) | ~15s | Multi-threading |
| LSH Filtering | Zig | ~5s | Hash table optimization |
| ML Training | Python (XGBoost) | ~6s | GPU acceleration (optional) |
| ML Inference | Python (XGBoost) | <1s | Batch prediction |
| Rule-Based Detection | Python | ~3s | Hash-based lookup |
| Evaluation | Python | ~105s | Cached computations |

### Memory Usage

| Component | Peak Memory | Notes |
|-----------|-------------|-------|
| Data Loading | ~500 MB | CSV parsing |
| Feature Matrix | ~200 MB | 26 features × 200k pairs |
| LSH Signatures | ~100 MB | 128 hashes × pairs |
| Model Training | ~1 GB | XGBoost internal buffers |
| Total Pipeline | ~2 GB | Concurrent operations |

### Scalability

| Factor | Scaling | Notes |
|--------|---------|-------|
| Source Files | O(n) | Linear with file count |
| Pair Generation | O(n²) | Quadratic (mitigated by LSH) |
| LSH Filtering | O(n log n) | Sub-linear with LSH |
| Feature Computation | O(n) | Parallel across cores |
| ML Inference | O(n) | Batch processing |

---

## Deployment Architecture

### Development Environment

```
┌─────────────────────────────────────────┐
│         Development Machine             │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Python 3.10-3.14               │   │
│  │  • Poetry virtualenv            │   │
│  │  • Jupyter notebooks            │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Zig 0.11+                      │   │
│  │  • Compiled libraries (.so)     │   │
│  │  • FFI bindings                 │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Dataset (TOMA)                 │   │
│  │  • 73,319 Java files            │   │
│  │  • ~500 MB raw data             │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Production Deployment

```
┌─────────────────────────────────────────┐
│         Production Server               │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Pipeline Orchestrator          │   │
│  │  • Bash scripts                 │   │
│  │  • Cron jobs / Airflow          │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Processing Cluster             │   │
│  │  • Multi-core CPU               │   │
│  │  • 16+ GB RAM                   │   │
│  │  • Optional: GPU for XGBoost    │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Storage                        │   │
│  │  • SSD for fast I/O             │   │
│  │  • 1 TB+ for datasets           │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

**Last Updated**: April 1, 2026  
**Pipeline Version**: 6.0.0  
**Architecture Version**: Hybrid Python + Zig
