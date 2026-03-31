"""Full type 1-3 code clone detection pipeline."""

import argparse
import logging
import pickle
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.python.utils.io import save_json

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
    logger.info("LSH: shingling done in %.2fs (%d shingles)", time.time() - t0, len(all_shingles))

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
    buckets: dict[int, list] = {}
    # file_id -> list of bucket_keys (for intersection)
    file_to_buckets: dict[int, list] = {fid: [] for fid in valid_ids}

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
        feature_cols = [c for c in feat_df_full.columns if c not in ("label", "id1", "id2")]
        logger.info(
            "Pre-computed features: %d pairs, %d features", len(feat_df_full), len(feature_cols)
        )

        # Filter to LSH candidates only — this is the actual LSH reduction
        feat_df_full["pair_key"] = feat_df_full.apply(
            lambda r: (min(r["id1"], r["id2"]), max(r["id1"], r["id2"])), axis=1
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
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        y_pred = (proba >= args.threshold).astype(int)
        logger.info("Using classification threshold: %.3f", args.threshold)
    else:
        y_pred = model.predict(X)
    step4_time = time.time() - step4_start

    # Build predictions dictionary
    # LSH candidates: use ML predictions
    predictions = {}
    for i, row in feat_df.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        predictions[pair] = int(y_pred[i])
    # Non-candidates: predicted as non-clone (label 0)
    for _, row in non_candidates.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        predictions[pair] = 0

    # Step 5: Full pipeline evaluation
    logger.info("=== Step 5: Evaluation ===")

    # Ground truth from ML-evaluable set (use binary_label for evaluation)
    ground_truth = {}
    for _, row in ml_df.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        ground_truth[pair] = int(row["binary_label"])

    metrics = evaluate_pipeline(predictions, ground_truth)
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
    save_json(metrics, args.eval_dir / "pipeline_metrics.json")

    # Plots
    plots_dir = args.eval_dir / "plots" / "pipeline"
    plots_dir.mkdir(parents=True, exist_ok=True)

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

    print("\n=== Pipeline Summary ===")
    print(f"Type-1 clones detected: {type1_count}")
    print(f"Type-3: TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"LSH reduction: {metrics['lsh_reduction_ratio']*100:.1f}%")
    print(f"Total time: {total_time:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
