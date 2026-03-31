# JCCD-X-V6 Implementation Plan

## Current Status Assessment

### ✅ Implemented Components

#### Phase 1: Data Preparation
- **Status**: 90% Complete
- **Location**: `src/python/preprocessing/data_prep.py`
- **What Works**:
  - Dataset loading from CSV files (type-1.csv through type-5.csv, nonclone.csv)
  - Label assignment (type-3/4 = positive, type-5/nonclone = negative, type-1/2 = -1)
  - Source file validation (checks if .java files exist)
  - Train/test splitting (70/30 with stratification)
  - Training set balancing
  - Basic metrics computation (dataset sizes, split ratios, removal stats)
  - Code length statistics (lines, tokens, characters)
  - Plots: samples per class, code length histograms

- **Gaps**:
  - ❌ Missing metrics: normalization impact (comments/blank lines removed)
  - ❌ Missing metrics: token abstraction stats (unique tokens before/after)
  - ❌ Missing plots: token type distribution pie chart
  - ❌ Missing: token-to-type abstraction distribution analysis

#### Phase 2: Source Code Preprocessing & Normalization
- **Status**: 85% Complete
- **Location**: `src/zig/preprocessing/normalization.zig`
- **What Works**:
  - Comment removal
  - Blank line removal
  - Whitespace normalization
  - Brace/indentation standardization

- **Gaps**:
  - ❌ Missing: Configurable literal abstraction (NUM/STR replacement)
  - ❌ Missing: Optional identifier length preservation
  - ❌ Missing: Selective token-to-type abstraction

#### Phase 3: CST Generation
- **Status**: 80% Complete
- **Location**: `src/zig/ast_stats/ast_stats.zig`
- **What Works**:
  - Tree-sitter integration for Java parsing
  - AST node counting
  - Max depth calculation
  - Basic AST histogram generation

- **Gaps**:
  - ❌ Missing metrics: Average AST depth per snippet tracking
  - ❌ Missing: Distribution of AST node types (keyword, identifier, operator, literal)
  - ❌ Missing: Shallow vs deep AST ratios
  - ❌ Missing: Token-to-type mapping coverage tracking

#### Phase 4: Token-to-Type Abstraction
- **Status**: 75% Complete
- **Location**: `src/zig/tokenization/tokenizer.zig`
- **What Works**:
  - Token extraction from CST
  - Basic token-to-type mapping (IDENTIFIER, MODIFIER, OPERATOR)
  - Token sequence encoding

- **Gaps**:
  - ❌ Missing: Configurable abstraction levels
  - ❌ Missing: Token type distribution tracking
  - ❌ Missing: Unique token counting before/after abstraction

#### Phase 5: Feature Engineering
- **Status**: 70% Complete
- **Location**: `src/zig/features/features.zig`, `src/python/feature_engineering/features.py`
- **What Works**:
  - 19 features computed (out of 26 planned)
  - Lexical: Levenshtein, Jaccard, Dice, LCS, SequenceMatcher
  - Structural: AST histogram cosine, bigram cosine, depth diff/ratio, node counts
  - Quantitative: Token ratio, identifier ratio, cyclomatic complexity metrics
  - Zig parallelization for batch feature computation
  - Feature correlation heatmap
  - Feature box plots per class

- **Gaps**:
  - ❌ Missing features (7):
    - `lex_char_jaccard` (character-level Jaccard)
    - `struct_bigram_cosine` (properly wired)
    - `quant_line_delta` (normalized line count difference)
    - `quant_lines1/2` (line counts per snippet)
    - `quant_chars1/2` (character counts per snippet)
    - `quant_char_ratio` (character count ratio)
  - ❌ Missing: Shingle coverage metrics
  - ❌ Missing: Violin plots for feature distributions

#### Phase 6: ML Model Training
- **Status**: 85% Complete
- **Location**: `src/python/model/train.py`, `src/python/model/models.py`
- **What Works**:
  - 5 models: XGBoost, Random Forest, Logistic Regression, Linear SVM, KNN
  - 5-fold stratified cross-validation
  - Primary metrics: Accuracy, Precision, Recall, F1, ROC-AUC
  - Secondary metrics: PR-AUC, confusion matrix, FPR/FNR
  - Feature importance extraction
  - Model selection by F1 score
  - Plots: ROC-AUC per fold, CV metric variance, feature importance, confusion matrix

- **Gaps**:
  - ❌ Missing: ROC curves per fold + mean curve (only AUC values plotted)
  - ❌ Missing: PR curves per fold + mean curve
  - ❌ Missing: Prediction probability calibration analysis
  - ❌ Missing: Cross-validation stability analysis (variance tracking)

#### Phase 7: Full Pipeline
- **Status**: 80% Complete
- **Location**: `src/python/pipeline/full_pipeline.py`
- **What Works**:
  - Type-1 detection (exact match after normalization)
  - Type-2 detection (identical abstracted tokens)
  - LSH candidate filtering with MinHash
  - ML classification for remaining pairs
  - Basic pipeline metrics (TP/FP/TN/FN, precision, recall, F1)
  - LSH reduction ratio tracking
  - Runtime per stage tracking
  - Plots: clones per type, results pie chart, runtime per stage

- **Gaps**:
  - ❌ Missing: Proper Type-2 detection validation (currently uses token equality only)
  - ❌ Missing: Adaptive LSH threshold based on method length
  - ❌ Missing: Candidate recall/precision metrics for LSH
  - ❌ Missing: Memory usage tracking per stage

---

## Implementation Plan by Phase

### Phase 1: Bug Fixes (COMPLETED ✅)
**Goal**: Fix critical bugs in existing implementation

**Tasks**:
1. ✅ Fix `run_features.sh` - ensure correct paths and error handling
2. ✅ Fix LSH wiring in `full_pipeline.py` - proper candidate filtering
3. ✅ Fix `common.py` - correct library path resolution
4. ✅ Fix feature count mismatch (25 vs 26 features)

**Status**: Complete

---

### Phase 2: Type-2 Detection in Pipeline (IN PROGRESS)
**Goal**: Improve Type-2 clone detection beyond simple token equality

**Tasks**:
1. **Enhance Type-2 Detection Logic**
   - Current: Only checks if `t1 == t2` (exact token sequence match)
   - Needed: Check for structural similarity with identifier/literal variations
   - Implementation: Use token abstraction patterns, not just equality

2. **Add Type-2 Validation Metrics**
   - Track number of Type-2 candidates found
   - Measure Type-2 detection recall (vs ground truth type-2.csv)
   - Compute false positive rate for Type-2 detection

3. **Update Pipeline Metrics**
   - Add Type-2 specific metrics to `pipeline_metrics.json`
   - Track Type-2 detection time separately

**Files to Modify**:
- `src/python/pipeline/full_pipeline.py`
- `src/python/evaluation/metrics.py` (new file)

**Tests**:
- Test Type-2 detection on known type-2.csv samples
- Verify Type-2 doesn't overlap with Type-1 detection

---

### Phase 3: ROC/PR Curves per Fold
**Goal**: Add proper ROC and PR curve visualization for each CV fold

**Tasks**:
1. **Create Curve Computation Module**
   ```python
   # src/python/evaluation/metrics.py
   def compute_roc_curve_per_fold(y_true, y_proba, n_folds=5)
   def compute_pr_curve_per_fold(y_true, y_proba, n_folds=5)
   ```

2. **Update Training Module**
   - Store y_true and y_proba for each fold
   - Compute mean ROC curve with confidence intervals
   - Compute mean PR curve

3. **Create Visualization Functions**
   - ROC curves: per-fold (light colors) + mean (bold line)
   - PR curves: per-fold (light colors) + mean (bold line)
   - Add AUC values to legend

**Files to Create/Modify**:
- `src/python/evaluation/metrics.py` (new)
- `src/python/evaluation/plots.py` (new)
- `src/python/model/train.py` (update to use new modules)

**Plots to Generate**:
- `artifacts/evaluation/plots/training/roc_curves.png`
- `artifacts/evaluation/plots/training/pr_curves.png`

---

### Phase 4: Evaluation Module (metrics.py, plots.py)
**Goal**: Centralize all metrics computation and plotting logic

**Tasks**:
1. **Create `src/python/evaluation/metrics.py`**
   ```python
   # Core metrics functions
   - compute_classification_metrics(y_true, y_pred, y_proba)
   - compute_cv_metrics(fold_results)
   - compute_pipeline_metrics(predictions, ground_truth)
   - compute_lsh_metrics(candidates, true_pairs)
   - compute_feature_metrics(feat_df)
   ```

2. **Create `src/python/evaluation/plots.py`**
   ```python
   # Plotting functions
   - plot_roc_curves(fold_results, save_path)
   - plot_pr_curves(fold_results, save_path)
   - plot_feature_correlation(feat_df, save_path)
   - plot_feature_importance(importance, names, save_path)
   - plot_confusion_matrix(cm, save_path)
   - plot_cv_variance(fold_results, save_path)
   - plot_pipeline_results(metrics, save_path)
   - plot_ablation_results(results, save_path)
   ```

3. **Refactor Existing Code**
   - Update `train.py` to use new modules
   - Update `features.py` to use new modules
   - Update `full_pipeline.py` to use new modules

**Benefits**:
- Single source of truth for metrics
- Consistent plot styling
- Easier to maintain and extend

---

### Phase 5: Feature Pruning / RFE
**Goal**: Identify optimal feature subset using Recursive Feature Elimination

**Tasks**:
1. **Create RFE Module**
   ```python
   # src/python/model/feature_selection.py
   def run_rfe_analysis(X, y, model, n_features_to_select, cv=5)
   def plot_rfe_results(rfe_results, save_path)
   ```

2. **Run RFE Experiments**
   - Start with all 26 features
   - Eliminate 1 feature at a time (or groups)
   - Track F1 score vs number of features
   - Identify "elbow point" where adding more features doesn't help

3. **Compare Feature Subsets**
   - Train models with top-N features (N=5, 10, 15, 20, 26)
   - Compare performance vs training time
   - Measure inference speedup

4. **Update Pipeline**
   - Option to use pruned feature set
   - Track which features are used

**Files to Create**:
- `src/python/model/feature_selection.py`
- `scripts/run_rfe.sh`

**Outputs**:
- `artifacts/evaluation/rfe_results.json`
- `artifacts/evaluation/plots/rfe/rfe_curve.png`
- `artifacts/evaluation/plots/rfe/feature_ranking.png`

---

### Phase 6: Ablation Study
**Goal**: Systematically evaluate contribution of each pipeline component

**Tasks**:
1. **Pipeline Ablation**
   ```python
   # src/python/evaluation/ablation.py
   configurations = {
       "full": ["normalization", "cst", "token_abs", "ml"],
       "no_norm": ["cst", "token_abs", "ml"],
       "no_cst": ["normalization", "token_abs", "ml"],
       "no_ml": ["normalization", "cst", "token_abs"],  # rule-based only
   }
   ```

2. **Feature Ablation**
   ```python
   feature_groups = {
       "all": ["lexical", "structural", "quantitative"],
       "no_lexical": ["structural", "quantitative"],
       "no_structural": ["lexical", "quantitative"],
       "no_quantitative": ["lexical", "structural"],
   }
   ```

3. **Runtime Ablation**
   - Run pipeline with LSH enabled vs disabled
   - Measure candidate reduction vs time tradeoff
   - Compute speedup ratio

4. **Create Visualization**
   - Line chart: F1 vs pipeline step removed
   - Line chart: Accuracy vs feature group removed
   - Bar plot: Runtime with vs without LSH
   - Heatmap: Feature group impact on metrics

**Files to Create**:
- `src/python/evaluation/ablation.py`
- `scripts/run_ablation.sh`

**Outputs**:
- `artifacts/evaluation/ablation_results.json`
- `artifacts/evaluation/plots/ablation/` (multiple plots)

---

### Phase 7: LSH Parameter Sweep
**Goal**: Optimize LSH parameters for best candidate recall

**Tasks**:
1. **Create Parameter Sweep Module**
   ```python
   # src/python/pipeline/lsh_tuning.py
   def sweep_lsh_parameters(token_data, true_pairs,
                           hash_range=(64, 256),
                           bands_range=(8, 32))
   ```

2. **Test Parameter Combinations**
   - Hash functions: 64, 128, 192, 256
   - Bands: 8, 16, 24, 32
   - Shingle k: 3, 4, 5
   - Track: recall, precision, candidate count, time

3. **Find Optimal Configuration**
   - Maximize recall while keeping candidates < threshold
   - Consider method length distribution
   - Implement adaptive threshold if needed

4. **Update Pipeline**
   - Use optimal LSH parameters
   - Add adaptive threshold option

**Files to Create**:
- `src/python/pipeline/lsh_tuning.py`
- `scripts/run_lsh_sweep.sh`

**Outputs**:
- `artifacts/evaluation/lsh_sweep_results.json`
- `artifacts/evaluation/plots/lsh/recall_vs_params.png`
- `artifacts/evaluation/plots/lsh/candidates_vs_threshold.png`

---

### Phase 8: CV Stability Analysis
**Goal**: Analyze model stability across cross-validation folds

**Tasks**:
1. **Compute Stability Metrics**
   ```python
   # src/python/evaluation/stability.py
   def compute_cv_stability(fold_results):
       - Metric variance (std/mean)
       - Rank correlation across folds
       - Feature importance stability
   ```

2. **Visualize Stability**
   - Box plots: metric distribution across folds
   - Heatmap: fold-to-fold correlation
   - Line chart: feature importance per fold

3. **Compare Models**
   - Rank models by stability (not just mean F1)
   - Identify models with low variance

**Files to Create**:
- `src/python/evaluation/stability.py`

**Outputs**:
- `artifacts/evaluation/stability_results.json`
- `artifacts/evaluation/plots/training/stability_analysis.png`

---

### Phase 9: Probability Calibration
**Goal**: Improve model probability estimates for better threshold tuning

**Tasks**:
1. **Add Calibration Methods**
   ```python
   # src/python/model/calibration.py
   from sklearn.calibration import CalibratedClassifierCV
   
   def calibrate_model(model, X_val, y_val, method='sigmoid')
   def evaluate_calibration(y_true, y_proba)
   ```

2. **Compare Calibration Methods**
   - Platt scaling (sigmoid)
   - Isotonic regression
   - No calibration (baseline)

3. **Update Pipeline**
   - Option to use calibrated probabilities
   - Tune decision threshold (not just 0.5)

**Files to Create**:
- `src/python/model/calibration.py`

**Outputs**:
- `artifacts/evaluation/calibration_results.json`
- `artifacts/evaluation/plots/training/calibration_curve.png`

---

### Phase 10: Sensitivity Analysis
**Goal**: Evaluate model robustness to input perturbations

**Tasks**:
1. **Define Perturbations**
   ```python
   # src/python/evaluation/sensitivity.py
   perturbations = {
       'noise': add_random_noise_to_features,
       'missing': randomly_mask_features,
       'shift': shift_feature_values,
   }
   ```

2. **Run Sensitivity Experiments**
   - Add 5%, 10%, 20% noise
   - Mask 5%, 10%, 20% of features
   - Track performance degradation

3. **Visualize Results**
   - Line chart: F1 vs noise level
   - Line chart: F1 vs missing feature %
   - Compare model robustness

**Files to Create**:
- `src/python/evaluation/sensitivity.py`
- `scripts/run_sensitivity.sh`

**Outputs**:
- `artifacts/evaluation/sensitivity_results.json`
- `artifacts/evaluation/plots/sensitivity/` (multiple plots)

---

### Phase 11: Memory Usage Profiling
**Goal**: Track and optimize memory consumption per pipeline stage

**Tasks**:
1. **Add Memory Tracking**
   ```python
   # src/python/utils/memory.py
   import tracemalloc
   
   def profile_memory(func):
       # Decorator to track memory usage
   ```

2. **Profile Each Stage**
   - Data preparation
   - Normalization
   - Tokenization
   - Feature engineering
   - LSH computation
   - ML inference

3. **Optimize Memory-Hungry Stages**
   - Use generators instead of lists
   - Process in batches
   - Clear caches between stages

4. **Add to Pipeline Metrics**
   - Track peak memory per stage
   - Track total memory footprint

**Files to Create**:
- `src/python/utils/memory.py`

**Outputs**:
- `artifacts/evaluation/memory_profile.json`
- `artifacts/evaluation/plots/pipeline/memory_usage.png`

---

## Testing Strategy

### Unit Tests
- Test each Zig module independently
- Test each Python module independently
- Focus on edge cases (empty input, single token, etc.)

### Integration Tests
- Test data flow between stages
- Verify output format compatibility
- Test with small synthetic datasets

### End-to-End Tests
- Run full pipeline on subset of data
- Verify final metrics are reasonable
- Test with known ground truth

### Test Files to Create/Update
```
tests/
├── test_data_prep.py          # ✅ exists
├── test_normalization.py      # ✅ exists
├── test_tokenizer.py          # ✅ exists
├── test_features.py           # ✅ exists
├── test_training.py           # ✅ exists
├── test_evaluation.py         # NEW: metrics.py, plots.py
├── test_ablation.py           # NEW: ablation study
├── test_lsh_tuning.py         # NEW: LSH parameter sweep
├── test_sensitivity.py        # NEW: sensitivity analysis
└── test_memory.py             # NEW: memory profiling
```

---

## Project Structure Updates

```
jccd-x/
├─ src/
│  ├─ python/
│  │  ├─ preprocessing/
│  │  │  ├── data_prep.py            # ✅ exists
│  │  │  └── normalization.py        # TODO: create wrapper
│  │  ├─ feature_engineering/
│  │  │  └── features.py             # ✅ exists
│  │  ├─ model/
│  │  │  ├── models.py               # ✅ exists
│  │  │  ├── train.py                # ✅ exists
│  │  │  ├── feature_selection.py    # NEW (Phase 5)
│  │  │  └── calibration.py          # NEW (Phase 9)
│  │  ├─ pipeline/
│  │  │  ├── full_pipeline.py        # ✅ exists
│  │  │  └── lsh_tuning.py           # NEW (Phase 7)
│  │  ├─ evaluation/
│  │  │  ├── __init__.py             # ✅ exists
│  │  │  ├── metrics.py              # NEW (Phase 4)
│  │  │  ├── plots.py                # NEW (Phase 4)
│  │  │  ├── ablation.py             # NEW (Phase 6)
│  │  │  ├── stability.py            # NEW (Phase 8)
│  │  │  └── sensitivity.py          # NEW (Phase 10)
│  │  └─ utils/
│  │     ├── io.py                   # TODO: create
│  │     └── memory.py               # NEW (Phase 11)
│  │
│  ├─ zig/ (existing structure)
│  └─ bindings/ (existing structure)
│
├─ scripts/
│  ├── run_all.sh                    # ✅ exists
│  ├── run_data_prep.sh              # ✅ exists
│  ├── run_preprocessing.sh          # ✅ exists
│  ├── run_tokenization.sh           # ✅ exists
│  ├── run_features.sh               # ✅ exists
│  ├── run_training.sh               # ✅ exists
│  ├── run_pipeline.sh               # ✅ exists
│  ├── run_rfe.sh                    # NEW (Phase 5)
│  ├── run_ablation.sh               # NEW (Phase 6)
│  ├── run_lsh_sweep.sh              # NEW (Phase 7)
│  ├── run_sensitivity.sh            # NEW (Phase 10)
│  └── test_all.sh                   # ✅ exists
│
├─ tests/ (as listed above)
├─ notebooks/
│  ├── exploration.ipynb             # TODO: create
│  ├── ablation_study.ipynb          # TODO: create
│  └── lsh_tuning.ipynb              # TODO: create
│
└─ artifacts/
   ├─ models/
   ├─ evaluation/
   │  ├── data_prep_metrics.json
   │  ├── feature_metrics.json
   │  ├── cv_results.json
   │  ├── test_metrics.json
   │  ├── pipeline_metrics.json
   │  ├── rfe_results.json           # NEW
   │  ├── ablation_results.json      # NEW
   │  ├── lsh_sweep_results.json     # NEW
   │  ├── stability_results.json     # NEW
   │  ├── calibration_results.json   # NEW
   │  ├── sensitivity_results.json   # NEW
   │  └── memory_profile.json        # NEW
   └─ plots/
      ├── data_prep/
      ├── features/
      ├── training/
      ├── pipeline/
      ├── rfe/                       # NEW
      ├── ablation/                  # NEW
      ├── lsh/                       # NEW
      └── sensitivity/               # NEW
```

---

## Implementation Order

**Priority 1 (Critical for completion)**:
1. Phase 2: Type-2 Detection (currently in progress)
2. Phase 4: Evaluation Module (foundation for other phases)
3. Phase 3: ROC/PR Curves (needed for proper evaluation)

**Priority 2 (Important for quality)**:
4. Phase 5: Feature Pruning / RFE
5. Phase 6: Ablation Study
6. Phase 7: LSH Parameter Sweep

**Priority 3 (Nice to have)**:
7. Phase 8: CV Stability Analysis
8. Phase 9: Probability Calibration
9. Phase 10: Sensitivity Analysis
10. Phase 11: Memory Usage Profiling

---

## Timeline Estimate

| Phase | Estimated Time | Dependencies |
|-------|---------------|--------------|
| Phase 2 | 1-2 days | None |
| Phase 3 | 1 day | Phase 4 |
| Phase 4 | 2-3 days | None |
| Phase 5 | 2 days | Phase 4 |
| Phase 6 | 3-4 days | Phase 4, 5 |
| Phase 7 | 2-3 days | None |
| Phase 8 | 1-2 days | Phase 3 |
| Phase 9 | 1-2 days | Phase 3 |
| Phase 10 | 2 days | Phase 4 |
| Phase 11 | 1-2 days | None |

**Total**: 16-23 days (working sequentially)

---

## Success Criteria

### Phase 1-4 (Core Pipeline)
- ✅ Type-1, Type-2, Type-3 detection working
- ✅ All 26 features computed and wired
- ✅ ROC/PR curves plotted per fold
- ✅ Centralized evaluation module

### Phase 5-7 (Optimization)
- Optimal feature subset identified
- Ablation study completed
- LSH parameters tuned

### Phase 8-11 (Analysis)
- Model stability quantified
- Probability calibration analyzed
- Sensitivity to perturbations measured
- Memory usage profiled

---

## Notes

1. **Zig Integration**: All computation-heavy tasks should remain in Zig. Python is only for ML, evaluation, and visualization.

2. **Reproducibility**: Save all intermediate results to avoid recomputation. Use seeds for randomness.

3. **Documentation**: Update README.md after each phase with new commands and outputs.

4. **Testing**: Write tests alongside implementation. Run `bash scripts/test_all.sh` frequently.

5. **Version Control**: Create feature branches for each phase. Merge to dev after completion.

---

## Next Steps

1. **Complete Phase 2** (Type-2 Detection) - Currently in progress
2. **Start Phase 4** (Evaluation Module) - Foundation for other phases
3. **Complete Phase 3** (ROC/PR Curves) - Using new evaluation module
4. **Proceed with remaining phases** in priority order
