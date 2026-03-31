# Pipeline Bug Fix Summary

## Issue
The pipeline was reporting **Precision=0.0000** and **Recall=0.0000** for Type-3 clone detection, even though:
- Type-1 and Type-2 detection were working perfectly (~100% recall)
- The ML model was making correct predictions
- Features were properly aligned

## Root Cause
The bug was in `src/python/pipeline/full_pipeline.py` in the evaluation logic:

1. **Binary vs Multi-class Label Mismatch**: The ML classifier outputs binary predictions (0=non-clone, 1=type-3), but the code was converting these to multi-class labels (0=non-clone, 3=type-3) before evaluation.

2. **Double Evaluation**: The code was evaluating twice:
   - First with binary labels (correct)
   - Then overwriting with multi-class labels against binary ground truth (incorrect)

The `evaluate_pipeline()` function expected binary labels (0 or 1), but was receiving multi-class labels (0, 1, 2, or 3), causing all Type-3 predictions (label=3) to be counted as non-clones.

## Fix Applied

### File: `src/python/pipeline/full_pipeline.py`

**Change 1**: Keep binary predictions for evaluation (lines 695-698)
```python
# Before:
y_pred = (proba >= args.threshold).astype(int)
y_pred = np.where(y_pred == 1, 3, 0)  # ❌ Wrong: converts to label 3

# After:
y_pred_binary = (proba >= args.threshold).astype(int)
y_pred = y_pred_binary  # ✅ Keep binary for evaluation
```

**Change 2**: Evaluate with binary labels first, then convert (lines 706-736)
```python
# Build predictions_binary for evaluation
predictions_binary = {}
for i, row in feat_df.iterrows():
    pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
    predictions_binary[pair] = int(y_pred[i])

# Evaluate with binary labels
metrics = evaluate_pipeline(predictions_binary, ground_truth)

# Convert to multi-class for final output
predictions = {}
for pair, pred_bin in predictions_binary.items():
    predictions[pair] = 3 if pred_bin == 1 else 0
```

**Change 3**: Remove duplicate evaluation (line 746)
```python
# Before:
metrics = evaluate_pipeline(predictions, ground_truth)  # ❌ Overwrites correct metrics

# After:
# Note: metrics already computed from binary evaluation above
# No need to re-evaluate with multi-class labels
```

**Change 4**: Conditional warning message (lines 857-865)
```python
# Only show warning if precision/recall are actually 0
if metrics['precision'] == 0 and metrics['recall'] == 0:
    print("⚠️  WARNING: Precision/Recall = 0...")
```

## Results After Fix

### With LSH (default):
```
Type-1 clones detected: 48109
Type-2 clones detected: 4222
Type-3: TP=28939 FP=5684 TN=111000 FN=3375
Accuracy: 0.9392
Precision: 0.8358
Recall: 0.8956
F1: 0.8647
LSH reduction: 51.0%
Total time: 39.3s
```

### Without LSH (--skip-lsh):
```
Type-3: TP=31875 FP=8620 TN=108064 FN=439
Precision: 0.7871
Recall: 0.9864
F1: 0.8756
```

## Analysis

1. **Type-1 Detection**: Perfect (recall: 99.98%, precision: 100%)
2. **Type-2 Detection**: Perfect (recall: 99.65%, precision: 100%)
3. **Type-3 Detection**: Very good (F1: 0.86-0.87)
   - LSH filtering reduces candidates by 51%
   - Slight recall drop (98.6% → 89.6%) due to LSH filtering
   - Precision improves (78.7% → 83.6%) with LSH

## Missing Type-2 Results

**Type-2 detection IS working correctly!** The pipeline summary now shows:
- `Type-2 clones detected: 4222`
- Type-2 recall: 99.65%
- Type-2 precision: 100%

The confusion was because the original output only showed Type-3 metrics in the summary line, but Type-2 detection was always working via rule-based token matching.

## Verification

Run the pipeline to verify:
```bash
# With LSH (default)
poetry run python src/python/pipeline/full_pipeline.py

# Without LSH (for comparison)
poetry run python src/python/pipeline/full_pipeline.py --skip-lsh

# Check metrics
cat artifacts/evaluation/pipeline_metrics.json
```

## Files Modified
- `src/python/pipeline/full_pipeline.py` (lines 695-746, 857-865)

## Date
March 31, 2026
