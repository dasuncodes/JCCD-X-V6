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

def detect_type1_type2(
    df: pd.DataFrame, token_data: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Detect type-1 (identical normalized) and type-2 (identical abstracted) clones.

    Returns: (type1_pairs, type2_pairs, remaining_pairs)
    """
    type1_idx = []

    for idx, row in df.iterrows():
        id1, id2 = row["id1"], row["id2"]
        if id1 not in token_data or id2 not in token_data:
            continue

        t1 = token_data[id1]["token_ids"]
        t2 = token_data[id2]["token_ids"]

        # Type-1: identical token sequences
        if t1 == t2:
            type1_idx.append(idx)
        # Type-2: could check AST structure similarity (skip for now,
        # since token abstraction already handles this)
        # The ML model handles type-3 vs non-clone

    type1_pairs = df.loc[type1_idx].reset_index(drop=True)
    remaining = df.drop(type1_idx).reset_index(drop=True)

    logger.info("Type-1 detected: %d pairs", len(type1_pairs))
    logger.info("Remaining for ML: %d pairs", len(remaining))

    return type1_pairs, pd.DataFrame(), remaining


# ---------------------------------------------------------------------------
# LSH candidate filtering (using Zig libraries)
# ---------------------------------------------------------------------------

def lsh_candidate_pairs(
    ids: list, token_data: dict, num_hashes: int = 128,
    bands: int = 16, k: int = 3,
) -> set[tuple]:
    """Find candidate pairs using MinHash LSH with Zig acceleration."""
    from src.bindings.minhash import MinHashLib
    from src.bindings.shingling import ShinglingLib

    shingling_lib = ShinglingLib()
    minhash_lib = MinHashLib()
    rows_per_band = num_hashes // bands

    # Build arrays for all valid IDs
    valid_ids = [fid for fid in ids if fid in token_data]
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
        all_tokens_arr, token_counts_arr, k,
    )
    logger.info("LSH: shingling done in %.2fs (%d shingles)", time.time() - t0, len(all_shingles))

    # Step 2: Compute MinHash signatures for each file
    t0 = time.time()
    signatures = np.zeros((len(valid_ids), num_hashes), dtype=np.uint64)

    for i, fid in enumerate(valid_ids):
        start = shingle_offsets[i]
        if i + 1 < len(valid_ids):
            end = shingle_offsets[i + 1]
        else:
            end = len(all_shingles)
        shingles = all_shingles[start:end]
        if len(shingles) > 0:
            signatures[i] = minhash_lib.minhash_signature(shingles, num_hashes)
        else:
            signatures[i] = np.iinfo(np.uint64).max

    logger.info("LSH: minhash done in %.2fs", time.time() - t0)

    # Step 3: LSH bucket assignment
    t0 = time.time()
    buckets: dict[tuple[int, int], list] = {}

    for i, fid in enumerate(valid_ids):
        band_buckets = minhash_lib.lsh_buckets(signatures[i], bands, rows_per_band)
        for b in range(bands):
            bucket_key = (b, int(band_buckets[b]))
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(fid)

    # Generate candidate pairs (limit bucket size to prevent explosion)
    MAX_BUCKET_SIZE = 500
    candidates = set()
    for bucket_ids in buckets.values():
        if len(bucket_ids) > MAX_BUCKET_SIZE:
            continue  # skip overly large buckets
        for i in range(len(bucket_ids)):
            for j in range(i + 1, len(bucket_ids)):
                pair = (min(bucket_ids[i], bucket_ids[j]),
                        max(bucket_ids[i], bucket_ids[j]))
                candidates.add(pair)

    logger.info("LSH: bucketing done in %.2fs (%d buckets, %d candidates)",
                time.time() - t0, len(buckets), len(candidates))

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
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
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
    parser.add_argument("--lsh-hashes", type=int, default=128)
    parser.add_argument("--lsh-bands", type=int, default=16)
    parser.add_argument("--shingle-k", type=int, default=3)
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

    # Separate type-1/2 (label=-1) from ML-evaluable (label 0/1)
    rule_based_df = test_df[test_df["label"] == -1].reset_index(drop=True)
    ml_df = test_df[test_df["label"] >= 0].reset_index(drop=True)
    logger.info("Rule-based (type-1/2): %d pairs", len(rule_based_df))
    logger.info("ML-evaluable: %d pairs", len(ml_df))

    # Step 1: Detect type-1 clones (exact match after tokenization)
    logger.info("=== Step 1: Type-1 Detection ===")
    step1_start = time.time()
    type1_pairs, type2_pairs, after_rules = detect_type1_type2(rule_based_df, token_data)
    step1_time = time.time() - step1_start
    logger.info("Type-1: %d detected in %.1fs", len(type1_pairs), step1_time)

    # Step 2: LSH candidate filtering for ML dataset
    logger.info("=== Step 2: LSH Candidate Filtering ===")
    step2_start = time.time()

    ml_ids = list(set(ml_df["id1"].tolist() + ml_df["id2"].tolist()))
    candidates = lsh_candidate_pairs(
        ml_ids, token_data,
        num_hashes=args.lsh_hashes,
        bands=args.lsh_bands,
        k=args.shingle_k,
    )
    step2_time = time.time() - step2_start

    # Build candidate set from ML dataset pairs
    ml_pair_set = set()
    for _, row in ml_df.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        ml_pair_set.add(pair)

    # LSH candidates that are also in our ML dataset
    lsh_candidates = candidates & ml_pair_set
    logger.info("LSH candidates: %d / %d pairs (%.1f%% reduction) in %.1fs",
                len(lsh_candidates), len(ml_pair_set),
                100 * (1 - len(lsh_candidates) / max(len(ml_pair_set), 1)),
                step2_time)

    # Step 3: Load features for LSH candidates
    logger.info("=== Step 3: Feature Loading ===")
    step3_start = time.time()

    # For the pipeline, use the pre-computed features from test_features.csv
    features_path = args.intermediate_dir / "features" / "test_features.csv"
    if features_path.exists():
        feat_df_full = pd.read_csv(features_path)
        feature_cols = [c for c in feat_df_full.columns if c not in ("label", "id1", "id2")]
        logger.info("Pre-computed features: %d pairs, %d features",
                     len(feat_df_full), len(feature_cols))

        # Filter to LSH candidates only — this is the actual LSH reduction
        feat_df_full["pair_key"] = feat_df_full.apply(
            lambda r: (min(r["id1"], r["id2"]), max(r["id1"], r["id2"])), axis=1
        )
        lsh_mask = feat_df_full["pair_key"].isin(lsh_candidates)
        feat_df = feat_df_full[lsh_mask].reset_index(drop=True)
        non_candidates = feat_df_full[~lsh_mask].reset_index(drop=True)
        logger.info("LSH filtered: %d candidate pairs (from %d total, %.1f%% reduction)",
                     len(feat_df), len(feat_df_full),
                     100 * len(non_candidates) / max(len(feat_df_full), 1))
    else:
        logger.error("Run feature engineering first")
        return

    step3_time = time.time() - step3_start

    # Step 4: ML Classification
    logger.info("=== Step 4: ML Classification ===")
    step4_start = time.time()

    X = feat_df[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
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

    # Ground truth from ML-evaluable set
    ground_truth = {}
    for _, row in ml_df.iterrows():
        pair = (min(row["id1"], row["id2"]), max(row["id1"], row["id2"]))
        ground_truth[pair] = int(row["label"])

    metrics = evaluate_pipeline(predictions, ground_truth)
    total_time = time.time() - total_start

    # Add timing and LSH metrics
    metrics["type1_detected"] = len(type1_pairs)
    metrics["lsh_candidates"] = len(lsh_candidates)
    metrics["lsh_total_pairs"] = len(ml_pair_set)
    metrics["lsh_reduction_ratio"] = 1 - len(lsh_candidates) / max(len(ml_pair_set), 1)
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
    labels = [f"TP ({metrics['tp']})", f"FP ({metrics['fp']})",
              f"TN ({metrics['tn']})", f"FN ({metrics['fn']})"]
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
