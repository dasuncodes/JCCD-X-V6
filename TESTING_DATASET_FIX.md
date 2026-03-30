# Testing Dataset Labeling Fix

**Date**: March 30, 2026  
**Issue**: Testing dataset was using binary labels instead of multi-class labels  
**Status**: ✅ Fixed

---

## Problem

The testing dataset was incorrectly using binary labels (clone vs non-clone) instead of preserving the original clone type labels. This prevented proper evaluation of the full pipeline which uses:
- Rule-based detection for Type-1 and Type-2 clones
- ML classifier for Type-3 clones and Non-clones

### Previous (Incorrect) Labeling

**Testing Set** (Binary - WRONG):
- label=1: Type-3 + Type-4 clones
- label=0: Type-5 + Non-clones
- label=-1: Type-1 + Type-2 (excluded)

This was problematic because:
1. Type-1 and Type-2 were excluded from evaluation
2. couldn't evaluate rule-based detection separately
3. Didn't match the actual pipeline workflow

---

## Solution

### New (Correct) Labeling

**Testing Set** (Multi-class - CORRECT):
| Label | Clone Type | Source | Count |
|-------|-----------|--------|-------|
| 0 | Non-clones | 30% of type-5.csv + nonclone.csv | ~116,000 |
| 1 | Type-1 clones | 100% of type-1.csv | ~48,000 |
| 2 | Type-2 clones | 100% of type-2.csv | ~4,200 |
| 3 | Type-3 clones | 30% of type-3.csv + type-4.csv | ~32,000 |

**Training Set** (Binary - Unchanged):
| Label | Clone Type | Source | Count |
|-------|-----------|--------|-------|
| 0 | Non-clones | 70% of type-5.csv + nonclone.csv (balanced) | ~57,000 |
| 1 | Type-3 clones | 70% of type-3.csv + type-4.csv (balanced) | ~57,000 |

*Note: Type-1 and Type-2 are excluded from training as the ML model only needs to detect Type-3 clones.*

---

## Implementation

### Changes to `data_prep.py`

1. **Added `assign_labels()` function**:
   - Assigns multi-class labels for testing dataset
   - Preserves actual clone types (0, 1, 2, 3)

2. **Added `assign_training_labels()` function**:
   - Assigns binary labels for training dataset
   - Type-3 = 1, Non-clone = 0, Type-1/2 = -1 (excluded)

3. **Updated `build_datasets()` function**:
   - Uses stratified split with binary labels for training
   - Preserves multi-class labels for testing
   - Properly distributes Type-1/2/3 and Non-clones

### Changes to `full_pipeline.py`

1. **Updated test dataset loading**:
   ```python
   # Filter by multi-class labels
   rule_based_df = test_df[test_df["label"].isin([1, 2])]  # Type-1, Type-2
   ml_df = test_df[test_df["label"].isin([0, 3])]  # Non-clones, Type-3
   
   # Convert to binary for ML evaluation
   ml_df["binary_label"] = ml_df["label"].apply(lambda x: 1 if x == 3 else 0)
   ```

2. **Proper pipeline evaluation**:
   - Type-1: Detected via exact match after normalization
   - Type-2: Detected via identical abstracted tokens
   - Type-3/Non-clone: Detected via ML classifier

---

## Pipeline Workflow

```
Testing Dataset (Multi-class labels)
         │
         ├── label=1,2 (Type-1, Type-2) ──→ Rule-based Detection
         │                                      ├── Exact match → Type-1
         │                                      └── Token match → Type-2
         │
         └── label=0,3 (Non-clone, Type-3) ──→ LSH Filtering
                                                    │
                                                    └── ML Classifier
                                                         ├── Predict 1 → Type-3
                                                         └── Predict 0 → Non-clone
```

---

## Metrics

### Pipeline Evaluation

With the corrected labeling, the pipeline now properly evaluates:

1. **Type-1 Detection**:
   - Count: Number of Type-1 clones detected
   - Precision: TP / (TP + FP) for Type-1
   - Recall: TP / (TP + FN) for Type-1

2. **Type-2 Detection**:
   - Count: Number of Type-2 clones detected
   - Precision: TP / (TP + FP) for Type-2
   - Recall: TP / (TP + FN) for Type-2

3. **Type-3 Detection (ML)**:
   - F1 Score: Harmonic mean of precision and recall
   - Accuracy: Overall correctness
   - AUC-ROC: Area under ROC curve

4. **Overall Pipeline**:
   - Combined metrics across all types
   - Per-type breakdown
   - Confusion matrix

---

## Files Modified

1. **`src/python/preprocessing/data_prep.py`**
   - Added `assign_labels()` for multi-class testing labels
   - Added `assign_training_labels()` for binary training labels
   - Updated `build_datasets()` to handle both

2. **`src/python/pipeline/full_pipeline.py`**
   - Updated test data filtering for rule-based vs ML
   - Added `binary_label` conversion for ML evaluation
   - Updated ground truth mapping

---

## Testing

To verify the fix:

```bash
# Re-run data preparation
bash scripts/run_data_prep.sh

# Check test set distribution
poetry run python -c "
import pandas as pd
df = pd.read_csv('data/processed/testing_dataset.csv')
print('Test set distribution:')
print(df['label'].value_counts().sort_index())
"

# Expected output:
# label
# 0    116XXX  (Non-clones)
# 1     48XXX  (Type-1)
# 2      4XXX  (Type-2)
# 3     32XXX  (Type-3)
```

---

## Impact

### Before Fix
- ❌ Type-1 and Type-2 excluded from evaluation
- ❌ Couldn't measure rule-based detection performance
- ❌ Inconsistent with pipeline workflow

### After Fix
- ✅ All clone types included in evaluation
- ✅ Proper measurement of rule-based detection
- ✅ Accurate pipeline performance metrics
- ✅ Better insights for optimization

---

## Next Steps

1. **Re-run data preparation** with the corrected code:
   ```bash
   bash scripts/run_data_prep.sh
   ```

2. **Re-generate features** for the new test set:
   ```bash
   bash scripts/run_features.sh
   ```

3. **Re-evaluate pipeline**:
   ```bash
   bash scripts/run_pipeline.sh
   ```

4. **Update results** in README and documentation

---

## References

- Original issue: Testing dataset binary labeling
- Related files: `data_prep.py`, `full_pipeline.py`
- Related phases: Phase 2 (Type-2 Detection), Phase 6 (Ablation Study)

---

**Status**: ✅ Complete  
**Verified**: Pending re-run of pipeline  
**Documentation**: Updated
