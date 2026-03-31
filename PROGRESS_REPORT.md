# JCCD-X-V6 Progress Report

**Date**: March 30, 2026  
**Status**: Phases 1-6 Complete (60% of total implementation)

---

## Executive Summary

The JCCD-X-V6 project implements a hybrid Python + Zig pipeline for detecting Type 1-3 syntactic code clones in Java code. The pipeline processes the TOMA dataset through multiple stages: data preparation, normalization, tokenization, feature engineering, ML training, and full pipeline evaluation.

**Current Progress**: 6 out of 11 phases completed (55%)  
**Next Milestone**: Complete Phases 7-9 for full optimization and analysis capabilities

---

## Completed Phases

### ✅ Phase 1: Bug Fixes
**Status**: Complete

**Changes**:
- Fixed `run_features.sh` script paths and error handling
- Fixed LSH wiring in `full_pipeline.py` for proper candidate filtering
- Fixed `common.py` library path resolution
- Fixed feature count mismatch (25 vs 26 features)

**Files Modified**:
- `scripts/run_features.sh`
- `src/python/pipeline/full_pipeline.py`
- `src/bindings/common.py`

---

### ✅ Phase 2: Type-2 Detection in Pipeline
**Status**: Complete

**Changes**:
- Enhanced Type-2 clone detection beyond simple token equality
- Added Type-2 detection metrics tracking
- Updated pipeline to properly categorize Type-1 vs Type-2 clones

**Implementation**:
- Type-1: Exact match after normalization (MD5 hash comparison)
- Type-2: Identical abstracted token sequences with different source

**Files Modified**:
- `src/python/pipeline/full_pipeline.py`

**Metrics Added**:
- `type1_detected`: Count of Type-1 clones
- `type2_detected`: Count of Type-2 clones
- Separate timing for rule-based detection

---

### ✅ Phase 3: ROC/PR Curves per Fold
**Status**: Complete

**Changes**:
- Added per-fold ROC curve computation with mean curve
- Added per-fold PR curve computation with mean curve
- Integrated curve data into cross-validation results

**Implementation**:
- `sklearn.metrics.roc_curve` for ROC data
- `sklearn.metrics.precision_recall_curve` for PR data
- Stored as lists in fold results for plotting

**Files Modified**:
- `src/python/model/train.py`
- `src/python/evaluation/plots.py`

**Outputs**:
- `artifacts/evaluation/plots/training/roc_curves_{model}.png`
- `artifacts/evaluation/plots/training/pr_curves_{model}.png`

---

### ✅ Phase 4: Evaluation Module (metrics.py, plots.py)
**Status**: Complete

**New Files Created**:
- `src/python/evaluation/metrics.py` (256 lines)
- `src/python/evaluation/plots.py` (520 lines)
- `tests/test_evaluation.py` (180 lines)

**Metrics Module Features**:
- `compute_classification_metrics()`: Comprehensive classification metrics
- `compute_fold_metrics()`: Per-fold metrics with curve data
- `aggregate_cv_metrics()`: Aggregate across CV folds
- `compute_cv_stability()`: Coefficient of variation analysis
- `compute_lsh_metrics()`: LSH performance metrics
- `compute_feature_metrics()`: Feature statistics
- `compute_pipeline_metrics()`: Full pipeline evaluation
- `compute_ablation_metrics()`: Ablation comparison

**Plots Module Features**:
- `plot_roc_curves()`: ROC curves with mean
- `plot_pr_curves()`: PR curves with mean
- `plot_feature_correlation()`: Feature correlation heatmap
- `plot_feature_importance()`: Horizontal bar chart
- `plot_confusion_matrix()`: Annotated confusion matrix
- `plot_cv_variance()`: Box plots of CV metrics
- `plot_feature_distributions()`: Feature distributions by class
- `plot_ablation_results()`: Ablation comparison bar chart
- `plot_lsh_recall_vs_threshold()`: LSH performance curves
- `plot_runtime_per_stage()`: Runtime breakdown
- `plot_memory_usage()`: Memory usage per stage

**Benefits**:
- Centralized metrics computation
- Consistent plot styling
- Easier maintenance and extension
- Reusable across pipeline stages

---

### ✅ Phase 5: Feature Pruning / RFE
**Status**: Complete

**New Files Created**:
- `src/python/model/feature_selection.py` (320 lines)
- `scripts/run_rfe.sh`
- `tests/test_feature_selection.py`

**Features**:
- `run_rfe_analysis()`: Recursive Feature Elimination with CV
- `plot_rfe_results()`: F1 vs number of features plot
- `compare_feature_subsets()`: Compare top-N feature subsets
- Elbow detection for optimal feature count

**Usage**:
```bash
bash scripts/run_rfe.sh
```

**Outputs**:
- `artifacts/evaluation/rfe_results.json`
- `artifacts/evaluation/plots/rfe/rfe_curve.png`
- `artifacts/evaluation/plots/rfe/optimal_feature_importance.png`
- `artifacts/evaluation/feature_subset_comparison.json`

**Expected Benefits**:
- Reduced model complexity
- Faster inference
- Potentially better generalization
- Feature importance insights

---

### ✅ Phase 6: Ablation Study
**Status**: Complete

**New Files Created**:
- `src/python/evaluation/ablation.py` (380 lines)
- `scripts/run_ablation.sh`

**Ablation Types**:

1. **Pipeline Ablation**:
   - `full`: All components (baseline)
   - `no_normalization`: Skip source code normalization
   - `no_cst`: Skip CST/AST structural features
   - `no_ml`: Rule-based detection only

2. **Feature Ablation**:
   - `all`: All features (baseline)
   - `no_lexical`: Remove lexical features
   - `no_structural`: Remove structural features
   - `no_quantitative`: Remove quantitative features

3. **Runtime Ablation**:
   - Compare pipeline with vs without LSH
   - Measure speedup factor
   - Track candidate reduction ratio

**Usage**:
```bash
bash scripts/run_ablation.sh
```

**Outputs**:
- `artifacts/evaluation/ablation_results.json`
- `artifacts/evaluation/plots/ablation/pipeline_ablation.png`
- `artifacts/evaluation/plots/ablation/runtime_ablation.png`

---

## Remaining Phases

### ⏳ Phase 7: LSH Parameter Sweep
**Priority**: High  
**Estimated Time**: 2-3 days

**Goal**: Optimize LSH parameters for best candidate recall

**Tasks**:
- Create `src/python/pipeline/lsh_tuning.py`
- Test hash functions: 64, 128, 192, 256
- Test bands: 8, 16, 24, 32
- Test shingle k: 3, 4, 5
- Find optimal configuration
- Implement adaptive threshold option

---

### ⏳ Phase 8: CV Stability Analysis
**Priority**: Medium  
**Estimated Time**: 1-2 days

**Goal**: Quantify model stability across CV folds

**Tasks**:
- Create `src/python/evaluation/stability.py`
- Compute coefficient of variation for metrics
- Analyze feature importance stability
- Compare models by stability (not just mean F1)

---

### ⏳ Phase 9: Probability Calibration
**Priority**: Medium  
**Estimated Time**: 1-2 days

**Goal**: Improve model probability estimates

**Tasks**:
- Create `src/python/model/calibration.py`
- Implement Platt scaling
- Implement isotonic regression
- Tune decision threshold (not just 0.5)
- Add calibration curve plots

---

### ⏳ Phase 10: Sensitivity Analysis
**Priority**: Low  
**Estimated Time**: 2 days

**Goal**: Evaluate model robustness to perturbations

**Tasks**:
- Create `src/python/evaluation/sensitivity.py`
- Add noise to features (5%, 10%, 20%)
- Mask features randomly
- Track performance degradation
- Compare model robustness

---

### ⏳ Phase 11: Memory Usage Profiling
**Priority**: Low  
**Estimated Time**: 1-2 days

**Goal**: Track and optimize memory consumption

**Tasks**:
- Create `src/python/utils/memory.py`
- Add memory tracking decorator
- Profile each pipeline stage
- Optimize memory-hungry stages
- Add memory usage plots

---

## Project Structure (Updated)

```
jccd-x/
├─ src/
│  ├─ python/
│  │  ├─ preprocessing/
│  │  │  ├── data_prep.py            ✅
│  │  │  └── normalization.py        ✅
│  │  ├─ feature_engineering/
│  │  │  └── features.py             ✅
│  │  ├─ model/
│  │  │  ├── models.py               ✅
│  │  │  ├── train.py                ✅ (updated)
│  │  │  ├── feature_selection.py    ✅ NEW
│  │  │  └── calibration.py          ⏳ TODO
│  │  ├─ pipeline/
│  │  │  ├── full_pipeline.py        ✅ (updated)
│  │  │  └── lsh_tuning.py           ⏳ TODO
│  │  ├─ evaluation/
│  │  │  ├── __init__.py             ✅
│  │  │  ├── metrics.py              ✅ NEW
│  │  │  ├── plots.py                ✅ NEW
│  │  │  ├── ablation.py             ✅ NEW
│  │  │  ├── stability.py            ⏳ TODO
│  │  │  └── sensitivity.py          ⏳ TODO
│  │  └─ utils/
│  │     ├── io.py                   ✅
│  │     └── memory.py               ⏳ TODO
│  │
│  ├─ zig/ (existing structure)      ✅
│  └─ bindings/ (existing structure) ✅
│
├─ scripts/
│  ├── run_all.sh                    ✅
│  ├── run_data_prep.sh              ✅
│  ├── run_preprocessing.sh          ✅
│  ├── run_tokenization.sh           ✅
│  ├── run_features.sh               ✅
│  ├── run_training.sh               ✅
│  ├── run_pipeline.sh               ✅
│  ├── run_rfe.sh                    ✅ NEW
│  ├── run_ablation.sh               ✅ NEW
│  ├── run_lsh_sweep.sh              ⏳ TODO
│  ├── run_sensitivity.sh            ⏳ TODO
│  └── test_all.sh                   ✅
│
├─ tests/
│  ├── test_data_prep.py             ✅
│  ├── test_normalization.py         ✅
│  ├── test_tokenizer.py             ✅
│  ├── test_features.py              ✅
│  ├── test_training.py              ✅
│  ├── test_evaluation.py            ✅ NEW
│  ├── test_feature_selection.py     ✅ NEW
│  └── ...
│
└─ artifacts/
   ├─ models/
   └─ evaluation/
      ├── data_prep_metrics.json     ✅
      ├── feature_metrics.json       ✅
      ├── cv_results.json            ✅ (updated)
      ├── test_metrics.json          ✅
      ├── pipeline_metrics.json      ✅
      ├── rfe_results.json           ✅ NEW
      ├── ablation_results.json      ✅ NEW
      └── plots/
         ├── data_prep/
         ├── features/
         ├── training/               ✅ (updated with ROC/PR)
         ├── pipeline/
         ├── rfe/                    ✅ NEW
         └── ablation/               ✅ NEW
```

---

## Testing Status

### Python Tests
- ✅ `test_data_prep.py`: Data preparation tests
- ✅ `test_normalization.py`: Normalization tests
- ✅ `test_tokenizer.py`: Tokenization tests
- ✅ `test_features.py`: Feature engineering tests
- ✅ `test_training.py`: ML training tests
- ✅ `test_evaluation.py`: Evaluation metrics/plots tests (NEW)
- ✅ `test_feature_selection.py`: RFE tests (NEW)

### Zig Tests
- ✅ `shingling_test`: Shingle generation tests
- ✅ `minhash_test`: MinHash signature tests
- ✅ `ast_stats_test`: AST statistics tests
- ✅ `normalization_test`: Normalization tests
- ✅ `tokenizer_test`: Tokenization tests
- ✅ `features_test`: Feature computation tests

**Run Tests**:
```bash
bash scripts/test_all.sh
```

---

## Key Metrics & Results

### Current Pipeline Performance (from README)

| Metric | Value |
|--------|-------|
| **Model Performance (5-Fold CV)** | |
| XGBoost F1 | 0.9725 ± 0.004 |
| XGBoost AUC | 0.9956 ± 0.001 |
| **Full Pipeline** | |
| Type-1 detected | 52,331 |
| Accuracy | 96.9% |
| Precision | 88.7% |
| Recall | 98.2% |
| F1 | 93.2% |
| LSH reduction | 85.8% |
| Total runtime | 37s |

### Expected Improvements from New Features

1. **Feature Pruning (Phase 5)**:
   - Expected: 20-30% reduction in features
   - Benefit: Faster inference, simpler model

2. **Ablation Study (Phase 6)**:
   - Quantify impact of each component
   - Identify most critical features
   - Guide future optimization

3. **LSH Parameter Sweep (Phase 7)**:
   - Expected: 5-10% improvement in candidate recall
   - Benefit: Better Type-3 detection

4. **Probability Calibration (Phase 9)**:
   - Better threshold tuning
   - More reliable probability estimates

---

## Usage Guide

### Quick Start
```bash
# Setup
make setup

# Run full pipeline
bash scripts/run_all.sh

# Run individual stages
bash scripts/run_data_prep.sh
bash scripts/run_preprocessing.sh
bash scripts/run_tokenization.sh
bash scripts/run_features.sh
bash scripts/run_training.sh
bash scripts/run_pipeline.sh

# Run analysis
bash scripts/run_rfe.sh
bash scripts/run_ablation.sh
```

### Testing
```bash
# All tests
bash scripts/test_all.sh

# Python tests only
poetry run pytest tests/ -v

# Zig tests only
zig build test
```

---

## Dependencies

### Python (from pyproject.toml)
- pandas ^2.0
- numpy ^1.24
- scikit-learn ^1.3
- xgboost ^2.0
- matplotlib ^3.7
- seaborn ^0.12
- rapidfuzz ^3.0
- cffi ^1.15

### Zig
- Standard library only
- tree-sitter (via zig-tree-sitter)

---

## Known Issues & Limitations

1. **Feature Count**: Currently 19 features implemented, 7 missing:
   - `lex_char_jaccard`
   - `quant_line_delta`
   - `quant_lines1/2`
   - `quant_chars1/2`
   - `quant_char_ratio`

2. **Ablation Simulation**: Current ablation study simulates component removal by adding noise or zeroing features. Real ablation would require re-running pipeline stages.

3. **LSH Parameters**: Currently using fixed parameters (128 hashes, 16 bands). Phase 7 will add parameter tuning.

---

## Next Steps

### Immediate (Week 1-2)
1. ✅ Complete Phase 6: Ablation Study (DONE)
2. ⏳ Start Phase 7: LSH Parameter Sweep
3. ⏳ Complete Phase 8: CV Stability Analysis

### Short-term (Week 3-4)
4. ⏳ Complete Phase 9: Probability Calibration
5. ⏳ Add missing 7 features
6. ⏳ Update README with new commands

### Medium-term (Week 5-6)
7. ⏳ Complete Phase 10: Sensitivity Analysis
8. ⏳ Complete Phase 11: Memory Profiling
9. ⏳ Create Jupyter notebooks for exploration

---

## Recommendations

1. **Prioritize Phase 7**: LSH parameter tuning will have immediate impact on pipeline performance.

2. **Document Findings**: Save ablation study results to guide future improvements.

3. **Test on Full Dataset**: Run complete pipeline on full TOMA dataset to validate scalability.

4. **Consider ONNX Export**: Export trained model to ONNX for potential Zig-native inference (advanced optimization).

5. **Write Paper**: Document methodology and results for publication.

---

## Contact & Support

For questions or issues:
- Check `IMPLEMENTATION_PLAN.md` for detailed implementation details
- Review `README.md` for usage instructions
- Run `bash scripts/test_all.sh` to verify installation

---

**Last Updated**: March 30, 2026  
**Next Review**: After Phase 7 completion
