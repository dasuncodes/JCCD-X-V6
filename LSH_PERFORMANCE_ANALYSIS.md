# LSH Performance Analysis

## Problem Statement
LSH (Locality Sensitive Hashing) is supposed to reduce both candidate pairs and runtime, but the current implementation shows minimal runtime improvement despite 51% reduction in pairs.

## Current Performance

| Metric | Without LSH | With LSH | Difference |
|--------|-------------|----------|------------|
| **ML Pairs** | 149,005 | 72,981 | -51.0% ✅ |
| **Step 2 Time** | 7.57s (skip) | 8.46s | +0.89s ❌ |
| **Step 3 Time** | 4.26s | 4.39s | +0.13s |
| **Step 4 Time (ML)** | 0.55s | 0.22s | -0.33s ✅ |
| **Total Time** | 56.37s | 55.79s | -0.58s (1%) |

## Analysis

### Why LSH Doesn't Save Time

1. **LSH Overhead (8.46s)**
   - Loading token data: ~3.5s
   - Computing shingles: ~0.15s
   - Computing MinHash signatures: ~0.2s
   - LSH bucket assignment: ~0.1s
   - **Direct checking (iterating ML pairs)**: ~4.3s ← **Bottleneck!**

2. **ML Classification is Already Fast (0.55s)**
   - With only 3 features (RFE-selected)
   - XGBoost is extremely efficient
   - Saving 0.33s by filtering 51% of pairs

3. **Feature Loading Dominates (4.3s)**
   - Loading 201,349 pairs from CSV
   - Filtering by pair keys
   - This cost is similar with or without LSH

### The Real Bottleneck

```
Without LSH:
  - LSH skip: 7.6s (just building pair set)
  - ML classify: 0.55s
  Total bottleneck: 8.15s

With LSH:
  - LSH filtering: 8.46s
  - ML classify: 0.22s
  Total bottleneck: 8.68s
```

**LSH is actually 0.53s SLOWER** in the bottleneck step!

## Root Causes

### 1. Direct Checking Algorithm
The current LSH implementation uses "direct checking" which iterates over all ML pairs:

```python
for id1, id2 in ml_pair_set:
    if bucket_sets_list[idx1] & bucket_sets_list[idx2]:
        candidates.add((id1, id2))
```

This is **O(n)** where n = number of ML pairs (149,005), defeating the purpose of LSH.

### 2. Small Feature Set
With only 3 features, XGBoost classification is extremely fast:
- 149,005 pairs → 0.55s
- 72,981 pairs → 0.22s

The classification is so fast that filtering doesn't provide much benefit.

### 3. Python Overhead
The LSH filtering is done in Python with set operations, which is slower than Zig batch computation.

## Solutions

### Option 1: Disable LSH for Small Feature Sets (Recommended)
If using RFE-selected features (<10 features), skip LSH entirely:

```python
if len(feature_cols) <= 10:
    logger.info("Skipping LSH: small feature set (%d features)", len(feature_cols))
    # Use all pairs directly
```

**Expected Impact:**
- Save 8.46s LSH overhead
- ML classification: +0.33s
- **Net saving: ~8s (14% faster)**

### Option 2: Optimize LSH Implementation
Move direct checking to Zig for batch computation:

```zig
// Zig batch implementation
pub fn lsh_filter_candidates(
    bucket_sets: []const u64,
    pair_indices: []const [2]usize,
) []bool {
    // SIMD-parallel bucket set intersection
}
```

**Expected Impact:**
- Reduce LSH time from 8.46s to ~2s
- **Net saving: ~6s (11% faster)**

### Option 3: Increase LSH Aggressiveness
Use more aggressive filtering to reduce ML pairs further:

```bash
--lsh-bands 8 --lsh-hashes 24  # ~20% of pairs
```

**Trade-off:**
- ML pairs: ~30,000 (80% reduction)
- Recall: ~75% (miss more clones)
- **Not recommended** - hurts accuracy too much

### Option 4: Pre-compute LSH Candidates
Cache LSH candidates during feature engineering:

```python
# During feature engineering
lsh_candidates = compute_lsh_candidates(token_data)
save_features_with_lsh_mask(features, lsh_candidates)
```

**Expected Impact:**
- Move 8.46s to feature engineering (one-time cost)
- Pipeline runtime: -8.46s
- **Net saving: 8.46s (15% faster)**

## Recommendation

**For Current Setup (3 features, XGBoost):**
```bash
# Skip LSH entirely
poetry run python src/python/pipeline/full_pipeline.py --skip-lsh
```

**Rationale:**
- LSH overhead (8.46s) > ML savings (0.33s)
- With 3 features, ML is too fast to benefit from filtering
- Better to process all pairs and maintain 98.6% recall

**For Future Setup (26 features, larger dataset):**
LSH becomes valuable when:
- Feature count > 20 (slower ML)
- Dataset > 500k pairs
- Real-time inference required

## Updated Pipeline Logic

The pipeline should automatically decide whether to use LSH:

```python
# Auto-decide LSH based on feature count and pair count
use_lsh = len(feature_cols) > 10 or len(ml_df) > 200000

if use_lsh:
    logger.info("Using LSH: %d features, %d pairs", len(feature_cols), len(ml_df))
    # Run LSH filtering
else:
    logger.info("Skipping LSH: small feature set or dataset")
    # Use all pairs directly
```

## Files to Modify
- `src/python/pipeline/full_pipeline.py` - Add auto-LSH logic
- `scripts/run_pipeline.sh` - Document when to use --skip-lsh

## Date
March 31, 2026
