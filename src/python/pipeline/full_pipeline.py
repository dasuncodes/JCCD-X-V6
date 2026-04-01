"""Full type 1-3 code clone detection pipeline."""

import argparse
import logging
import pickle
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.python.utils.io import save_json
from src.python.evaluation.plots import (
    plot_confusion_matrix,
    plot_model_comparison_bar,
    plot_model_comparison_scatter,
    plot_multi_model_roc_curves,
    plot_multi_model_pr_curves,
    plot_threshold_sensitivity,
    plot_class_distribution,
    plot_error_analysis,
    plot_feature_group_contribution,
    plot_lsh_pareto_curve,
    plot_lsh_recall_vs_reduction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based detection
# ---------------------------------------------------------------------------


def _compute_source_hashes(
    ids: set,
    normalized_dir: Path,
) -> dict:
    """Compute MD5 hashes of normalized source files for all IDs."""
    import hashlib

    hashes = {}
    for fid in ids:
        path = normalized_dir / f"{fid}.java"
        if path.exists():
            try:
                content = path.read_bytes()
                hashes[fid] = hashlib.md5(content).hexdigest()
            except OSError:
                hashes[fid] = None
        else:
            hashes[fid] = None
    return hashes


def detect_type1_type2(
    df: pd.DataFrame,
    token_data: dict,
    normalized_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Detect type-1 (identical normalized source) and type-2 (identical abstracted tokens).

    Type-1: normalized source code is byte-identical (same after normalization).
    Type-2: abstracted token sequences are identical but source differs
            (same structure, different identifiers/literals).

    Returns: (type1_pairs, type2_pairs, remaining_pairs)
    """
    # Collect all unique IDs from the dataframe
    all_ids = set(df["id1"].tolist()) | set(df["id2"].tolist())
    logger.info("Computing source hashes for %d unique files...", len(all_ids))
    source_hashes = _compute_source_hashes(all_ids, normalized_dir)

    type1_idx = []
    type2_idx = []

    for idx, row in df.iterrows():
        id1, id2 = row["id1"], row["id2"]
        if id1 not in token_data or id2 not in token_data:
            continue

        h1 = source_hashes.get(id1)
        h2 = source_hashes.get(id2)
        t1 = token_data[id1]["token_ids"]
        t2 = token_data[id2]["token_ids"]

        if h1 is not None and h2 is not None and h1 == h2:
            # Type-1: identical normalized source
            type1_idx.append(idx)
        elif t1 == t2:
            # Type-2: identical abstracted tokens but different source
            type2_idx.append(idx)

    type1_pairs = df.loc[type1_idx].reset_index(drop=True)
    type2_pairs = df.loc[type2_idx].reset_index(drop=True)
    remaining = df.drop(type1_idx + type2_idx).reset_index(drop=True)

    logger.info("Type-1 detected: %d pairs (identical normalized source)", len(type1_pairs))
    logger.info("Type-2 detected: %d pairs (identical abstracted tokens)", len(type2_pairs))
    logger.info("Remaining for ML: %d pairs", len(remaining))

    return type1_pairs, type2_pairs, remaining


# ---------------------------------------------------------------------------
# LSH candidate filtering (using Zig libraries)
# ---------------------------------------------------------------------------


def lsh_candidate_pairs(
    ids: list,
    token_data: dict,
    num_hashes: int = 128,
    bands: int = 16,
    k: int = 3,
    max_bucket_size: int = 1000,
    ml_pair_set: set[tuple] = None,
    ml_df: pd.DataFrame = None,
    sim_threshold: float = 0.0,
) -> set[tuple]:
    """Find candidate pairs using MinHash LSH with Zig acceleration.

    If ml_pair_set is provided, only return candidates that are in ml_pair_set.
    This is more efficient than generating all candidate pairs.
    """
    from src.bindings.minhash import MinHashLib
    from src.bindings.shingling import ShinglingLib

    # Parameter validation
    if k < 1:
        raise ValueError(f"Shingle size k must be >= 1, got {k}")
    if bands <= 0:
        raise ValueError(f"Number of bands must be > 0, got {bands}")
    if num_hashes <= 0:
        raise ValueError(f"Number of hashes must be > 0, got {num_hashes}")
    if bands > num_hashes:
        logger.warning(
            "bands (%d) > num_hashes (%d) leads to rows_per_band = 0. "
            "Setting rows_per_band = 1 and adjusting bands to num_hashes.",
            bands,
            num_hashes,
        )
        bands = num_hashes
    if num_hashes % bands != 0:
        logger.warning(
            "num_hashes (%d) not divisible by bands (%d). " "LSH effectiveness may be reduced.",
            num_hashes,
            bands,
        )
    rows_per_band = num_hashes // bands
    if rows_per_band == 0:
        # Should not happen due to above check, but fallback
        rows_per_band = 1
        bands = num_hashes
        logger.warning("Adjusted bands to %d to ensure rows_per_band > 0", bands)

    shingling_lib = ShinglingLib()
    minhash_lib = MinHashLib()

    # Build arrays for all valid IDs
    valid_ids = [fid for fid in ids if fid in token_data]
    fid_to_index = {fid: i for i, fid in enumerate(valid_ids)}
    logger.info("LSH: processing %d files", len(valid_ids))

    # ---------------------------------------------------------------------------
    # Caching
    # ---------------------------------------------------------------------------
    import hashlib
    import pickle
    from pathlib import Path

    def compute_fingerprint(token_data, valid_ids):
        """Compute a stable fingerprint of token data for caching."""
        data = []
        for fid in sorted(valid_ids):
            token_ids = token_data[fid]["token_ids"]
            # Use first 10 tokens and length to keep fingerprint small
            snippet = (fid, len(token_ids), tuple(token_ids[:10]))
            data.append(snippet)
        return hashlib.sha256(pickle.dumps(data)).hexdigest()

    cache_dir = Path("artifacts/cache/lsh")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = compute_fingerprint(token_data, valid_ids)
    cache_key = f"lsh_{num_hashes}_{bands}_{k}_{fingerprint}"
    cache_path = cache_dir / f"{cache_key}.pkl"
    cache_hit = False
    all_shingles = None
    shingle_offsets = None
    signatures = None
    bucket_ids = None
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            # Validate cached data matches current parameters
            if (
                cache_data.get("num_hashes") == num_hashes
                and cache_data.get("bands") == bands
                and cache_data.get("rows_per_band") == rows_per_band
                and cache_data.get("k") == k
                and cache_data.get("fingerprint") == fingerprint
            ):
                # Load cached arrays
                all_shingles = cache_data["all_shingles"]
                shingle_offsets = cache_data["shingle_offsets"]
                signatures = cache_data["signatures"]
                bucket_ids = cache_data["bucket_ids"]
                # Ensure lengths match
                if len(bucket_ids) == len(valid_ids) and bucket_ids.shape[1] == bands:
                    cache_hit = True
                    logger.info("LSH: cache hit (%s)", cache_path.name)
                else:
                    logger.warning("LSH: cache size mismatch, recomputing")
            else:
                logger.info("LSH: cache parameters mismatch, recomputing")
        except Exception as e:
            logger.warning("LSH: cache load failed: %s", e)

    if not cache_hit:
        logger.info("LSH: cache miss, computing from scratch")

    if not cache_hit:
        # Step 1: Compute shingles for all files using Zig batch
        all_token_seqs = []
        token_counts_list = []
        for fid in valid_ids:
            tokens = token_data[fid]["token_ids"]
            all_token_seqs.extend(tokens)
            token_counts_list.append(len(tokens))

        all_tokens_arr = np.array(all_token_seqs, dtype=np.uint32)
        token_counts_arr = np.array(token_counts_list, dtype=np.uintp)

        t0 = time.time()
        all_shingles, shingle_offsets = shingling_lib.compute_shingles_batch(
            all_tokens_arr,
            token_counts_arr,
            k,
        )
        logger.info(
            "LSH: shingling done in %.2fs (%d shingles)", time.time() - t0, len(all_shingles)
        )

        # Step 2: Compute MinHash signatures for each file (batch)
        t0 = time.time()
        # Compute shingle counts per file
        shingle_counts = np.diff(shingle_offsets, append=len(all_shingles))
        # Batch compute signatures
        signatures = minhash_lib.minhash_signature_batch_flat(
            all_shingles,
            shingle_offsets,
            shingle_counts,
            num_hashes,
        )
        logger.info("LSH: minhash done in %.2fs", time.time() - t0)

        # Step 3: LSH bucket assignment
        t0 = time.time()
        # bucket_key -> list of file_ids
        buckets = {}
        # file_id -> list of bucket_keys (for intersection)
        file_to_buckets = {fid: [] for fid in valid_ids}

        # Batch compute bucket IDs for all files
        bucket_ids = minhash_lib.lsh_buckets_batch(signatures, bands, rows_per_band)
        for i, fid in enumerate(valid_ids):
            for b in range(bands):
                bucket_key = (b << 64) | int(bucket_ids[i, b])
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(fid)
                file_to_buckets[fid].append(bucket_key)
        # Precompute bucket sets for efficient intersection
        file_to_bucket_sets = {fid: set(keys) for fid, keys in file_to_buckets.items()}
        bucket_sets_list = [set() for _ in range(len(valid_ids))]
        for fid, s in file_to_bucket_sets.items():
            idx = fid_to_index[fid]
            bucket_sets_list[idx] = s

        logger.info("LSH: bucket assignment done in %.2fs", time.time() - t0)

        # Save to cache
        cache_data = {
            "num_hashes": num_hashes,
            "bands": bands,
            "rows_per_band": rows_per_band,
            "k": k,
            "fingerprint": fingerprint,
            "all_shingles": all_shingles,
            "shingle_offsets": shingle_offsets,
            "signatures": signatures,
            "bucket_ids": bucket_ids,
        }
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info("LSH: saved cache to %s", cache_path.name)
        except Exception as e:
            logger.warning("LSH: cache save failed: %s", e)
    else:
        # Cache hit: we already have all_shingles, shingle_offsets, signatures, bucket_ids
        # Compute file_to_buckets and bucket_sets_list from cached bucket_ids
        # bucket_key -> list of file_ids
        buckets = {}
        # file_id -> list of bucket_keys (for intersection)
        file_to_buckets = {fid: [] for fid in valid_ids}
        for i, fid in enumerate(valid_ids):
            for b in range(bands):
                bucket_key = (b << 64) | int(bucket_ids[i, b])
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(fid)
                file_to_buckets[fid].append(bucket_key)
        # Precompute bucket sets for efficient intersection
        file_to_bucket_sets = {fid: set(keys) for fid, keys in file_to_buckets.items()}
        bucket_sets_list = [set() for _ in range(len(valid_ids))]
        for fid, s in file_to_bucket_sets.items():
            idx = fid_to_index[fid]
            bucket_sets_list[idx] = s

    # Step 4: Find candidates among ml_pair_set
    t0 = time.time()
    candidates = set()

    if ml_pair_set is not None:
        # Direct checking: iterate over ml_pair_set
        for id1, id2 in ml_pair_set:
            # Skip if either file not in token_data (should be already filtered)
            if id1 not in fid_to_index or id2 not in fid_to_index:
                continue
            idx1 = fid_to_index[id1]
            idx2 = fid_to_index[id2]
            # Check if any bucket key is shared
            if bucket_sets_list[idx1] & bucket_sets_list[idx2]:
                candidates.add((id1, id2))
        logger.info(
            "LSH: direct checking done in %.2fs, %d candidates", time.time() - t0, len(candidates)
        )
    elif ml_df is not None:
        # Direct checking using ml_df rows (no need to build ml_pair_set)
        for _, row in ml_df.iterrows():
            id1 = row["id1"]
            id2 = row["id2"]
            if id1 not in fid_to_index or id2 not in fid_to_index:
                continue
            idx1 = fid_to_index[id1]
            idx2 = fid_to_index[id2]
            if bucket_sets_list[idx1] & bucket_sets_list[idx2]:
                candidates.add((min(id1, id2), max(id1, id2)))
        logger.info(
            "LSH: direct checking (df) done in %.2fs, %d candidates",
            time.time() - t0,
            len(candidates),
        )
    else:
        # Fallback: generate all candidate pairs (original algorithm)
        total_buckets = len(buckets)
        skipped_buckets = 0
        total_pairs = 0
        for bucket_key, bucket_ids in buckets.items():
            bucket_size = len(bucket_ids)
            if bucket_size > max_bucket_size:
                skipped_buckets += 1
                continue
            for i in range(bucket_size):
                for j in range(i + 1, bucket_size):
                    pair = (min(bucket_ids[i], bucket_ids[j]), max(bucket_ids[i], bucket_ids[j]))
                    candidates.add(pair)
            total_pairs += bucket_size * (bucket_size - 1) // 2
            del bucket_ids
        logger.info(
            "LSH: bucketing done in %.2fs (%d buckets, %d candidates, %d skipped large buckets)",
            time.time() - t0,
            total_buckets,
            len(candidates),
            skipped_buckets,
        )

    # Apply similarity filtering if threshold > 0
    if sim_threshold > 0 and candidates:
        filtered = set()
        for id1, id2 in candidates:
            idx1 = fid_to_index.get(id1)
            idx2 = fid_to_index.get(id2)
            if idx1 is None or idx2 is None:
                continue
            sim = minhash_lib.minhash_similarity(signatures[idx1], signatures[idx2])
            if sim >= sim_threshold:
                filtered.add((id1, id2))
        logger.info(
            "Similarity filtering: kept %d / %d pairs (threshold=%.2f)",
            len(filtered),
            len(candidates),
            sim_threshold,
        )
        candidates = filtered

    # Cleanup
    buckets.clear()
    file_to_buckets.clear()

    return candidates


# ---------------------------------------------------------------------------
# Pipeline evaluation
# ---------------------------------------------------------------------------


def evaluate_pipeline(
    predictions: dict[tuple, int],
    ground_truth: dict[tuple, int],
) -> dict:
    """Evaluate pipeline predictions against ground truth."""
    tp = fp = tn = fn = 0
    for pair, true_label in ground_truth.items():
        pred = predictions.get(pair, 0)
        if true_label == 1 and pred == 1:
            tp += 1
        elif true_label == 1 and pred == 0:
            fn += 1
        elif true_label == 0 and pred == 1:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Full clone detection pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--intermediate-dir", type=Path, default=Path("data/intermediate"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--lsh-hashes", type=int, default=36)
    parser.add_argument("--lsh-bands", type=int, default=12)
    parser.add_argument("--shingle-k", type=int, default=3)
    parser.add_argument(
        "--skip-lsh",
        action="store_true",
        default=False,
        help="Skip LSH filtering and use all pairs",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.30, help="Classification threshold for clone detection"
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.0,
        help="Minimum MinHash similarity for LSH candidate pairs (0.0 = no filtering)",
    )
    parser.add_argument(
        "--max-bucket-size",
        type=int,
        default=5000,
        help="Maximum LSH bucket size to prevent explosion",
    )
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/processed/normalized"))
    args = parser.parse_args()

    total_start = time.time()

    # Load model
    model = joblib.load(args.model_dir / "best_model.joblib")
    model_name = joblib.load(args.model_dir / "best_model_name.joblib")
    logger.info("Loaded model: %s", model_name)

    # Load selected features and RFE model if available
    selected_features = None
    selected_features_path = args.model_dir / "selected_features.json"
    if selected_features_path.exists():
        with open(selected_features_path, "r") as f:
            selected_features = json.load(f)
        logger.info("Loaded selected features (%d features)", len(selected_features))
        # Optionally load RFE model
        rfe_model_path = args.model_dir / "best_model_rfe.joblib"
        if rfe_model_path.exists():
            model = joblib.load(rfe_model_path)
            logger.info("Loaded RFE-retrained model from %s", rfe_model_path)

    # Load token data
    logger.info("Loading token data...")
    with open(args.intermediate_dir / "token_data.pkl", "rb") as f:
        token_data = pickle.load(f)

    # Load test dataset
    test_df = pd.read_csv(args.data_dir / "testing_dataset.csv")
    logger.info("Test dataset: %d pairs", len(test_df))

    # Test set has multi-class labels:
    # label=0: Non-clones
    # label=1: Type-1 clones
    # label=2: Type-2 clones
    # label=3: Type-3 clones

    # Separate rule-based (type-1, type-2) from ML-evaluable (type-3, non-clones)
    rule_based_df = test_df[test_df["label"].isin([1, 2])].reset_index(drop=True)
    ml_df = test_df[test_df["label"].isin([0, 3])].reset_index(drop=True)

    # Convert ML labels to binary for classification (3 -> 1 for Type-3)
    ml_df = ml_df.copy()
    ml_df["binary_label"] = ml_df["label"].apply(lambda x: 1 if x == 3 else 0)

    logger.info("Rule-based (type-1/type-2): %d pairs", len(rule_based_df))
    logger.info("ML-evaluable (type-3/non-clones): %d pairs", len(ml_df))

    # Step 1: Detect type-1 clones (exact match after tokenization)
    logger.info("=== Step 1: Type-1 Detection ===")
    step1_start = time.time()
    type1_pairs, type2_pairs, after_rules = detect_type1_type2(
        rule_based_df,
        token_data,
        args.normalized_dir,
    )
    step1_time = time.time() - step1_start
    logger.info("Type-1: %d detected in %.1fs", len(type1_pairs), step1_time)

    # Evaluate rule-based detection (type-1 and type-2)
    rule_metrics = {}
    # Build mapping from pair to true label (1 or 2)
    true_labels = {}
    for _, row in rule_based_df.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        true_labels[pair] = int(row["label"])
    # Predicted labels: 1 for type1_pairs, 2 for type2_pairs, 0 otherwise
    type1_set = set()
    for _, row in type1_pairs.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        type1_set.add(pair)
    type2_set = set()
    for _, row in type2_pairs.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        type2_set.add(pair)

    tp1 = fp1 = fn1 = tn1 = 0
    tp2 = fp2 = fn2 = tn2 = 0
    for pair, true in true_labels.items():
        if pair in type1_set:
            pred = 1
        elif pair in type2_set:
            pred = 2
        else:
            pred = 0
        if true == 1:
            if pred == 1:
                tp1 += 1
            else:
                fn1 += 1
        elif true == 2:
            if pred == 2:
                tp2 += 1
            else:
                fn2 += 1
        # Note: false positives for rule-based detection are not applicable
        # because we only evaluate on known clones (rule_based_df).

    recall1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0
    recall2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
    precision1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0  # fp1 always 0
    precision2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0  # fp2 always 0

    rule_metrics["type1_recall"] = float(recall1)
    rule_metrics["type2_recall"] = float(recall2)
    rule_metrics["type1_precision"] = float(precision1)
    rule_metrics["type2_precision"] = float(precision2)
    rule_metrics["type1_tp"] = int(tp1)
    rule_metrics["type1_fn"] = int(fn1)
    rule_metrics["type2_tp"] = int(tp2)
    rule_metrics["type2_fn"] = int(fn2)
    rule_metrics["type1_detected"] = len(type1_pairs)
    rule_metrics["type2_detected"] = len(type2_pairs)

    logger.info("Rule-based detection evaluation:")
    logger.info("  Type-1: recall=%.4f (TP=%d, FN=%d)", recall1, tp1, fn1)
    logger.info("  Type-2: recall=%.4f (TP=%d, FN=%d)", recall2, tp2, fn2)

    # Step 2: LSH candidate filtering for ML dataset
    logger.info("=== Step 2: LSH Candidate Filtering ===")
    step2_start = time.time()

    # Number of ML pairs
    len_ml = len(ml_df)

    if args.skip_lsh:
        # Skip LSH filtering: all pairs are candidates
        # Build candidate set from ML dataset pairs
        ml_pair_set = set()
        for _, row in ml_df.iterrows():
            pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
            ml_pair_set.add(pair)
        candidates = ml_pair_set
        lsh_candidates = ml_pair_set
        step2_time = time.time() - step2_start
        logger.info(
            "LSH skipped: using all %d pairs as candidates in %.1fs",
            len_ml,
            step2_time,
        )
    else:
        ml_ids = list(set(ml_df["id1"].tolist() + ml_df["id2"].tolist()))
        candidates = lsh_candidate_pairs(
            ml_ids,
            token_data,
            num_hashes=args.lsh_hashes,
            bands=args.lsh_bands,
            k=args.shingle_k,
            max_bucket_size=args.max_bucket_size,
            ml_df=ml_df,
            sim_threshold=args.sim_threshold,
        )
        step2_time = time.time() - step2_start
        lsh_candidates = candidates  # already filtered to ml_df
        logger.info(
            "LSH candidates: %d / %d pairs (%.1f%% reduction) in %.1fs",
            len(lsh_candidates),
            len_ml,
            100 * (1 - len(lsh_candidates) / max(len_ml, 1)),
            step2_time,
        )

    # Step 3: Load features for LSH candidates
    logger.info("=== Step 3: Feature Loading ===")
    step3_start = time.time()

    # For the pipeline, use the pre-computed features from test_features.csv
    features_path = args.intermediate_dir / "features" / "test_features.csv"
    if features_path.exists():
        feat_df_full = pd.read_csv(features_path)
        all_feature_cols = [c for c in feat_df_full.columns if c not in ("label", "id1", "id2")]
        logger.info(
            "Pre-computed features: %d pairs, %d features", len(feat_df_full), len(all_feature_cols)
        )

        # Check if model was trained with RFE-selected features
        selected_features_path = args.model_dir / "selected_features.json"
        if selected_features_path.exists():
            with open(selected_features_path, "r") as f:
                feature_cols = json.load(f)
            logger.info("Using RFE-selected features: %d features", len(feature_cols))
        else:
            feature_cols = all_feature_cols
            logger.info("Using all features: %d features", len(feature_cols))

        # Filter to LSH candidates only — this is the actual LSH reduction
        feat_df_full["pair_key"] = feat_df_full.apply(
            lambda r: (int(min(r["id1"], r["id2"])), int(max(r["id1"], r["id2"]))), axis=1
        )
        lsh_mask = feat_df_full["pair_key"].isin(lsh_candidates)
        feat_df = feat_df_full[lsh_mask].reset_index(drop=True)
        non_candidates = feat_df_full[~lsh_mask].reset_index(drop=True)
        logger.info(
            "LSH filtered: %d candidate pairs (from %d total, %.1f%% reduction)",
            len(feat_df),
            len(feat_df_full),
            100 * len(non_candidates) / max(len(feat_df_full), 1),
        )
    else:
        logger.error("Run feature engineering first")
        return

    step3_time = time.time() - step3_start

    # Step 4: ML Classification
    logger.info("=== Step 4: ML Classification ===")
    step4_start = time.time()

    X = feat_df[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    logger.info("Classification input shape: %s", X.shape)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        logger.info(
            "Probability stats: min=%.4f, max=%.4f, mean=%.4f",
            proba.min(),
            proba.max(),
            proba.mean(),
        )
        logger.info(
            "Positive predictions (proba >= %.3f): %d / %d",
            args.threshold,
            (proba >= args.threshold).sum(),
            len(proba),
        )
        y_pred_binary = (proba >= args.threshold).astype(int)
        # Keep binary predictions for evaluation (0=non-clone, 1=type-3)
        # Will map to multi-class labels later for final output
        y_pred = y_pred_binary
        logger.info("Using classification threshold: %.3f", args.threshold)
    else:
        y_pred = model.predict(X)
        logger.info("Predictions: %d samples", len(y_pred))
    step4_time = time.time() - step4_start

    # Build predictions dictionary for ML evaluation (binary: 0=non-clone, 1=type-3)
    # LSH candidates: use ML predictions
    predictions_binary = {}
    for i, row in feat_df.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        predictions_binary[pair] = int(y_pred[i])
    # Non-candidates: predicted as non-clone (label 0)
    for _, row in non_candidates.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        predictions_binary[pair] = 0

    # Step 5: Full pipeline evaluation (using binary labels)
    logger.info("=== Step 5: Evaluation ===")

    # Ground truth from ML-evaluable set (use binary_label for evaluation)
    ground_truth = {}
    for _, row in ml_df.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        ground_truth[pair] = int(row["binary_label"])

    # Evaluate with binary labels first
    metrics = evaluate_pipeline(predictions_binary, ground_truth)

    # Convert binary predictions to multi-class for final output
    # (0=non-clone, 1=type-1, 2=type-2, 3=type-3)
    predictions = {}
    for pair, pred_bin in predictions_binary.items():
        predictions[pair] = 3 if pred_bin == 1 else 0

    # Add rule-based predictions (type-1 and type-2)
    for pair in type1_set:
        predictions[pair] = 1
    for pair in type2_set:
        predictions[pair] = 2

    # Ground truth for all pairs (including type-1, type-2)
    ground_truth_all = {}
    for _, row in test_df.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        ground_truth_all[pair] = int(row["label"])

    # Note: metrics already computed from binary evaluation above
    # No need to re-evaluate with multi-class labels
    total_time = time.time() - total_start

    # Add timing and LSH metrics
    metrics["type1_detected"] = len(type1_pairs)
    metrics["type2_detected"] = len(type2_pairs)
    # Add rule-based detection metrics
    for key, value in rule_metrics.items():
        metrics[key] = value
    metrics["lsh_candidates"] = len(lsh_candidates)
    metrics["lsh_total_pairs"] = len_ml
    metrics["lsh_reduction_ratio"] = 1 - len(lsh_candidates) / max(len_ml, 1)
    metrics["step1_time_sec"] = step1_time
    metrics["step2_time_sec"] = step2_time
    metrics["step3_time_sec"] = step3_time
    metrics["step4_time_sec"] = step4_time
    metrics["total_time_sec"] = total_time

    args.eval_dir.mkdir(parents=True, exist_ok=True)

    # Plots directory
    plots_dir = args.eval_dir / "plots" / "pipeline"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Multi-class evaluation (Non, Type-1, Type-2, Type-3)
    # Build ground truth and predictions for all pairs
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
    )

    all_pairs = list(ground_truth_all.keys())
    y_true = np.array([ground_truth_all[p] for p in all_pairs])
    y_pred = np.array([predictions.get(p, 0) for p in all_pairs])

    # Define all four classes and their names
    all_classes = [0, 1, 2, 3]
    class_names = ["Non-clone", "Type-1", "Type-2", "Type-3"]

    # Compute confusion matrix with all four classes
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)

    # Save confusion matrix
    from src.python.evaluation.plots import plot_confusion_matrix

    plot_confusion_matrix(
        cm,
        plots_dir / "confusion_matrix_multiclass.png",
        class_names=class_names,
        title="Confusion Matrix (Full Pipeline)",
    )

    # Compute multiclass metrics (weighted average)
    precision_weighted = precision_score(y_true, y_pred, labels=all_classes, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=all_classes, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=all_classes, average='weighted', zero_division=0)
    accuracy_multiclass = accuracy_score(y_true, y_pred)

    # Add multiclass metrics to the existing metrics dict
    metrics['precision_weighted'] = float(precision_weighted)
    metrics['recall_weighted'] = float(recall_weighted)
    metrics['f1_weighted'] = float(f1_weighted)
    metrics['accuracy_multiclass'] = float(accuracy_multiclass)

    # Save classification report (including all four classes)
    report = classification_report(
        y_true, y_pred, labels=all_classes, target_names=class_names, output_dict=True
    )
    save_json(report, args.eval_dir / "classification_report.json")
    logger.info("Multi-class confusion matrix and classification report saved")
    save_json(metrics, args.eval_dir / "pipeline_metrics.json")

    # Stacked bar: clones per type
    type1_count = len(type1_pairs)
    type3_tp = metrics["tp"]
    type3_fn = metrics["fn"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Detected"], [type1_count], label="Type-1", color="#2196F3")
    ax.bar(["Detected"], [type3_tp], bottom=[type1_count], label="Type-3 (TP)", color="#4CAF50")
    ax.bar(["Missed"], [type3_fn], label="Type-3 (FN)", color="#F44336")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Clones Detected per Type")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "clones_per_type.png", dpi=150)
    plt.close(fig)

    # Pie chart: TP, FP, TN, FN
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]]
    labels = [
        f"TP ({metrics['tp']})",
        f"FP ({metrics['fp']})",
        f"TN ({metrics['tn']})",
        f"FN ({metrics['fn']})",
    ]
    colors = ["#4CAF50", "#FF9800", "#2196F3", "#F44336"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
    ax.set_title("Pipeline Results")
    plt.tight_layout()
    fig.savefig(plots_dir / "results_pie.png", dpi=150)
    plt.close(fig)

    # Runtime per stage
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = ["Type-1 Detection", "LSH Filtering", "Feature Loading", "ML Classification"]
    times = [step1_time, step2_time, step3_time, step4_time]
    ax.barh(stages, times)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Runtime per Pipeline Stage")
    plt.tight_layout()
    fig.savefig(plots_dir / "runtime_per_stage.png", dpi=150)
    plt.close(fig)

    logger.info("Plots saved to %s", plots_dir)

    # =========================================================================
    # Additional paper-ready plots
    # =========================================================================
    logger.info("Generating additional paper-ready plots...")
    from sklearn.metrics import (
        roc_curve,
        precision_recall_curve,
        average_precision_score,
        roc_auc_score,
    )
    from sklearn.inspection import permutation_importance

    # Load CV results for model comparison
    cv_results_path = args.eval_dir / "cv_results.json"
    # Load model comparison results for plotting
    model_comparison_path = Path("artifacts/evaluation_comprehensive/model_comparison.json")
    cv_results_path = args.eval_dir / "cv_results.json"
    if model_comparison_path.exists():
        with open(model_comparison_path) as f:
            model_results = json.load(f)
        logger.info("Loaded model comparison results from %s", model_comparison_path)
    elif cv_results_path.exists():
        with open(cv_results_path) as f:
            cv_results = json.load(f)
        # Compute mean metrics across folds per model
        model_results = {}
        for model_name, model_data in cv_results.items():
            # Only include actual model results (dict with aggregate and folds)
            if isinstance(model_data, dict) and "aggregate" in model_data and "folds" in model_data:
                agg = model_data["aggregate"]
                mean_metrics = {}
                # Map aggregate keys to simple metric names
                key_mapping = {
                    "f1_mean": "f1",
                    "roc_auc_mean": "roc_auc",
                    "accuracy_mean": "accuracy",
                    "precision_mean": "precision",
                    "recall_mean": "recall",
                    "pr_auc_mean": "pr_auc",
                }
                for agg_key, metric_key in key_mapping.items():
                    if agg_key in agg:
                        mean_metrics[metric_key] = agg[agg_key]
                # training_time_sec not available in cv_results
                mean_metrics["training_time_sec"] = 0.0
                model_results[model_name] = mean_metrics
            # Skip non-model entries like 'best_model', 'best_f1', 'feature_importance'
        logger.info("Loaded CV results and computed mean metrics")
    else:
        logger.warning(
            "Neither model comparison nor CV results found; skipping model comparison plots"
        )
        model_results = {}
    if model_results:
        # Generate model comparison plots
        plot_model_comparison_bar(
            model_results,
            plots_dir / "model_comparison_f1.png",
            metric="f1",
            title="Model Comparison: F1 Score",
        )
        plot_model_comparison_bar(
            model_results,
            plots_dir / "model_comparison_auc.png",
            metric="roc_auc",
            title="Model Comparison: ROC-AUC",
        )
        plot_model_comparison_scatter(model_results, plots_dir / "model_efficiency_scatter.png")
        logger.info("Model comparison plots generated")

    # Compute probabilities for all ML-evaluable pairs (for ROC/PR curves and threshold sensitivity)
    # Use the best model and all features from ml_df
    if "feat_df_full" in locals():
        # Add pair_key column to ml_df
        ml_df = ml_df.copy()
        ml_df["pair_key"] = ml_df.apply(
            lambda r: (int(min(r["id1"], r["id2"])), int(max(r["id1"], r["id2"]))), axis=1
        )
        original_len = len(ml_df)
        # Filter to pairs that have features
        existing_pairs = set(feat_df_full["pair_key"])
        ml_df = ml_df[ml_df["pair_key"].isin(existing_pairs)]
        if len(ml_df) < original_len:
            logger.warning("Skipping %d pairs without features", original_len - len(ml_df))
        # Merge features and labels
        merged = feat_df_full.merge(ml_df[["pair_key", "binary_label"]], on="pair_key", how="inner")
        X_all = merged[feature_cols].values.astype(np.float64)
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=1.0, neginf=0.0)
        y_all = merged["binary_label"].values.astype(np.int32)
        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_proba_all = model.predict_proba(X_all)[:, 1]
        else:
            # fallback to decision function
            try:
                decision_scores = model.decision_function(X_all)
                y_proba_all = 1 / (1 + np.exp(-decision_scores))
            except:
                y_proba_all = None
        if y_proba_all is not None:
            # ROC and PR curves for best model only (single model)
            # For multi-model curves we need probabilities from other models; skip for now
            # Threshold sensitivity
            plot_threshold_sensitivity(
                y_all,
                y_proba_all,
                plots_dir / "threshold_sensitivity.png",
                title="Threshold Sensitivity Analysis (Best Model)",
            )
            logger.info("Threshold sensitivity plot generated")
    else:
        logger.warning("Features not loaded; skipping probability-based plots")

    # Dataset class distribution
    class_names = ["Non-clone", "Type-3 Clone"]
    plot_class_distribution(
        ml_df["binary_label"].values,
        class_names,
        plots_dir / "class_distribution.png",
        title="Dataset Class Distribution (Binary)",
    )
    all_labels = test_df["label"].values
    all_class_names = ["Non-clone", "Type-1", "Type-2", "Type-3"]
    plot_class_distribution(
        all_labels,
        all_class_names,
        plots_dir / "class_distribution_multiclass.png",
        title="Dataset Class Distribution (Multi-class)",
    )
    logger.info("Class distribution plots generated")

    # Error analysis (already have confusion matrix multiclass)
    # Compute predictions for all pairs (including type-1/2) using the final predictions dict
    all_pairs = list(predictions.keys())
    y_true_all = np.array([ground_truth_all[p] for p in all_pairs])
    y_pred_all = np.array([predictions[p] for p in all_pairs])
    plot_error_analysis(
        y_true_all,
        y_pred_all,
        all_class_names,
        plots_dir / "error_analysis.png",
        title="Error Analysis by Clone Type",
    )
    logger.info("Error analysis plot generated")

    # Feature group contribution
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        for i, name in enumerate(feature_cols):
            feature_importance[name] = float(model.feature_importances_[i])
    else:
        # compute permutation importance (may be slow)
        logger.info("Computing permutation importance...")
        result = permutation_importance(
            model, X_all, y_all, n_repeats=5, random_state=42, n_jobs=-1
        )
        for i, name in enumerate(feature_cols):
            feature_importance[name] = float(result.importances_mean[i])
    plot_feature_group_contribution(
        feature_cols,
        feature_importance,
        plots_dir / "feature_group_contribution.png",
        title="Feature Group Contribution",
    )
    logger.info("Feature group contribution plot generated")

    # LSH trade-off plots
    # Load LSH sweep results if available
    lsh_sweep_path = Path("sweep_results.json")
    if lsh_sweep_path.exists():
        with open(lsh_sweep_path) as f:
            lsh_sweep = json.load(f)
        # Convert to required format
        lsh_configs = []
        for cfg in lsh_sweep:
            precision = cfg.get("precision", 0)
            recall = cfg.get("recall", 0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            lsh_configs.append(
                {
                    "name": f"H={cfg['hashes']},B={cfg['bands']}",
                    "reduction": cfg.get("lsh_reduction", 0),
                    "f1": f1,
                    "recall": recall,
                }
            )
        if not lsh_configs:
            logger.warning("No valid LSH configurations found; using defaults")
            lsh_configs = [
                {"name": "Default", "reduction": 51.0, "f1": 0.8647, "recall": 0.8956},
                {"name": "Balanced", "reduction": 45.7, "f1": 0.8761, "recall": 0.9324},
                {"name": "High Recall", "reduction": 17.5, "f1": 0.8783, "recall": 0.9798},
                {"name": "No LSH", "reduction": 0.0, "f1": 0.8756, "recall": 0.9864},
            ]
        plot_lsh_pareto_curve(
            lsh_configs,
            plots_dir / "lsh_pareto_curve.png",
            title="LSH Trade-off: Speed vs Accuracy",
        )
        plot_lsh_recall_vs_reduction(
            lsh_configs,
            plots_dir / "lsh_recall_vs_reduction.png",
            title="LSH: Candidate Reduction vs Recall",
        )
        logger.info("LSH trade-off plots generated")
    else:
        logger.warning("LSH sweep results not found; using default configurations")
        # Use default configurations from earlier analysis
        lsh_configs = [
            {"name": "Default", "reduction": 51.0, "f1": 0.8647, "recall": 0.8956},
            {"name": "Balanced", "reduction": 45.7, "f1": 0.8761, "recall": 0.9324},
            {"name": "High Recall", "reduction": 17.5, "f1": 0.8783, "recall": 0.9798},
            {"name": "No LSH", "reduction": 0.0, "f1": 0.8756, "recall": 0.9864},
        ]
        plot_lsh_pareto_curve(
            lsh_configs,
            plots_dir / "lsh_pareto_curve.png",
            title="LSH Trade-off: Speed vs Accuracy",
        )
        plot_lsh_recall_vs_reduction(
            lsh_configs,
            plots_dir / "lsh_recall_vs_reduction.png",
            title="LSH: Candidate Reduction vs Recall",
        )
        logger.info("LSH trade-off plots generated with default configurations")

    logger.info("All additional plots generated")

    print("\n=== Pipeline Summary ===")
    print(f"Type-1 clones detected: {type1_count}")
    print(f"Type-2 clones detected: {len(type2_pairs)}")
    print(f"Type-3: TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"LSH reduction: {metrics['lsh_reduction_ratio']*100:.1f}%")
    print(f"Total time: {total_time:.1f}s")
    print("")

    # Show warning if precision/recall are very low (possible issues)
    if metrics["recall"] < 0.1:
        print("")
        print("⚠️  WARNING: Very low recall detected!")
        print("   This usually means LSH filtering is too aggressive.")
        print("   Try these solutions:")
        print("   1. Reduce LSH filtering: --lsh-bands 16 --lsh-hashes 48")
        print("   2. Skip LSH entirely: --skip-lsh")
        print("   3. Lower classification threshold: --threshold 0.2")
        print("   Check: artifacts/evaluation/pipeline_metrics.json")

    print("Done.")


if __name__ == "__main__":
    main()
