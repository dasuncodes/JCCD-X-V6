"""Feature engineering: compute 27 features per pair using Zig batch computation."""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from src.bindings.features import FeaturesLib
from src.python.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    "lex_levenshtein_ratio",
    "lex_jaccard_tokens",
    "lex_dice_tokens",
    "lex_lcs_ratio",
    "lex_sequence_match_ratio",
    "struct_ast_hist_cosine",
    "struct_bigram_cosine",
    "struct_depth_diff",
    "struct_depth_ratio",
    "struct_node_count1",
    "struct_node_count2",
    "struct_node_ratio",
    "struct_node_diff",
    "quant_token_ratio",
    "quant_identifier_ratio",
    "quant_cc1",
    "quant_cc2",
    "quant_cc_ratio",
    "quant_cc_diff",
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_flat_arrays(token_data: dict) -> tuple:
    """Convert token_data dict into flat arrays for Zig batch computation."""
    ids = sorted(token_data.keys())
    id_to_idx = {fid: i for i, fid in enumerate(ids)}
    n = len(ids)

    # Token sequences: concatenate all
    token_seqs = []
    token_offsets = np.zeros(n, dtype=np.uintp)
    token_counts = np.zeros(n, dtype=np.uintp)
    offset = 0

    for i, fid in enumerate(ids):
        td = token_data[fid]
        tokens = td["token_ids"]
        token_offsets[i] = offset
        token_counts[i] = len(tokens)
        token_seqs.extend(tokens)
        offset += len(tokens)

    all_tokens = np.array(token_seqs, dtype=np.uint32)

    # AST histograms: 12 values per file
    histograms = np.zeros(n * 12, dtype=np.uint32)
    depths = np.zeros(n, dtype=np.uint16)
    node_counts = np.zeros(n, dtype=np.uint32)
    cyclomatics = np.zeros(n, dtype=np.uint32)

    for i, fid in enumerate(ids):
        ast = token_data[fid]["ast_stats"]
        h = ast["histogram"]
        histograms[i * 12:i * 12 + len(h)] = h[:12]
        depths[i] = ast["max_depth"]
        node_counts[i] = ast["node_count"]
        cyclomatics[i] = ast["cyclomatic_complexity"]

    # Bigrams: concatenate all
    bigram_seqs = []
    bigram_offsets = np.zeros(n, dtype=np.uintp)
    bigram_counts = np.zeros(n, dtype=np.uintp)
    offset = 0

    for i, fid in enumerate(ids):
        bigs = token_data[fid]["ast_stats"].get("bigrams", [])
        bigram_offsets[i] = offset
        bigram_counts[i] = len(bigs)
        bigram_seqs.extend(bigs)
        offset += len(bigs)

    all_bigrams = (
        np.array(bigram_seqs, dtype=np.uint64) if bigram_seqs
        else np.array([], dtype=np.uint64)
    )

    return {
        "ids": ids,
        "id_to_idx": id_to_idx,
        "all_tokens": all_tokens,
        "token_offsets": token_offsets,
        "token_counts": token_counts,
        "histograms": histograms,
        "depths": depths,
        "node_counts": node_counts,
        "cyclomatics": cyclomatics,
        "all_bigrams": all_bigrams,
        "bigram_offsets": bigram_offsets,
        "bigram_counts": bigram_counts,
    }


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features_batch(
    df: pd.DataFrame,
    flat: dict,
    feat_lib: FeaturesLib,
) -> pd.DataFrame:
    """Compute features for all pairs using Zig batch + Python Levenshtein."""
    id_to_idx = flat["id_to_idx"]
    logger.info("Computing features for %d pairs...", len(df))

    # Build pair index arrays
    pair_idx_a = []
    pair_idx_b = []
    valid_mask = []

    for _, row in df.iterrows():
        id1, id2 = row["id1"], row["id2"]
        if id1 in id_to_idx and id2 in id_to_idx:
            pair_idx_a.append(id_to_idx[id1])
            pair_idx_b.append(id_to_idx[id2])
            valid_mask.append(True)
        else:
            pair_idx_a.append(0)
            pair_idx_b.append(0)
            valid_mask.append(False)

    pair_idx_a = np.array(pair_idx_a, dtype=np.uintp)
    pair_idx_b = np.array(pair_idx_b, dtype=np.uintp)
    valid_mask = np.array(valid_mask)
    valid_count = valid_mask.sum()
    logger.info("Valid pairs: %d / %d", valid_count, len(df))

    if valid_count == 0:
        logger.warning("No valid pairs found")
        return pd.DataFrame()

    # Compute Levenshtein ratios in Python (rapidfuzz is fast C code)
    logger.info("Computing Levenshtein ratios...")
    levenshtein_ratios = np.zeros(len(df), dtype=np.float32)
    start = time.time()

    token_offsets = flat["token_offsets"]
    token_counts = flat["token_counts"]
    all_tokens = flat["all_tokens"]

    for i, (_, row) in enumerate(df.iterrows()):
        if not valid_mask[i]:
            continue
        ia, ib = pair_idx_a[i], pair_idx_b[i]
        tok_a = all_tokens[token_offsets[ia]:token_offsets[ia] + token_counts[ia]]
        tok_b = all_tokens[token_offsets[ib]:token_offsets[ib] + token_counts[ib]]
        s_a = " ".join(str(int(t)) for t in tok_a)
        s_b = " ".join(str(int(t)) for t in tok_b)
        levenshtein_ratios[i] = Levenshtein.normalized_similarity(s_a, s_b)
        if (i + 1) % 50000 == 0:
            logger.info("  Levenshtein: %d / %d", i + 1, len(df))

    logger.info("Levenshtein done in %.1f sec", time.time() - start)

    # Compute remaining features in Zig (parallelized)
    logger.info("Computing Zig features (parallel)...")
    start = time.time()
    zig_features = feat_lib.compute_features_batch(
        pair_idx_a, pair_idx_b,
        flat["all_tokens"], flat["token_offsets"], flat["token_counts"],
        flat["histograms"], flat["depths"], flat["node_counts"], flat["cyclomatics"],
        flat["all_bigrams"], flat["bigram_offsets"], flat["bigram_counts"],
        num_threads=0,
    )
    logger.info("Zig features done in %.1f sec", time.time() - start)

    # Build output DataFrame
    # zig_features columns: [jaccard, dice, lcs_ratio, seq_match, hist_cosine,
    #   bigram_cosine, depth_diff, depth_ratio, node_count1, node_count2,
    #   node_ratio, node_diff, token_ratio, identifier_ratio, cc1, cc2,
    #   cc_ratio, cc_diff]  (18 columns, indices 1..18 in the 25-col array)
    # We use columns 1-18 from zig_features and prepend levenshtein

    result = pd.DataFrame()
    result["lex_levenshtein_ratio"] = levenshtein_ratios
    result["lex_jaccard_tokens"] = zig_features[:, 1]
    result["lex_dice_tokens"] = zig_features[:, 2]
    result["lex_lcs_ratio"] = zig_features[:, 3]
    result["lex_sequence_match_ratio"] = zig_features[:, 4]
    result["struct_ast_hist_cosine"] = zig_features[:, 5]
    result["struct_bigram_cosine"] = zig_features[:, 6]
    result["struct_depth_diff"] = zig_features[:, 7]
    result["struct_depth_ratio"] = zig_features[:, 8]
    result["struct_node_count1"] = zig_features[:, 9]
    result["struct_node_count2"] = zig_features[:, 10]
    result["struct_node_ratio"] = zig_features[:, 11]
    result["struct_node_diff"] = zig_features[:, 12]
    result["quant_token_ratio"] = zig_features[:, 13]
    result["quant_identifier_ratio"] = zig_features[:, 14]
    result["quant_cc1"] = zig_features[:, 15]
    result["quant_cc2"] = zig_features[:, 16]
    result["quant_cc_ratio"] = zig_features[:, 17]
    result["quant_cc_diff"] = zig_features[:, 18]

    result["label"] = df["label"].values
    result["id1"] = df["id1"].values
    result["id2"] = df["id2"].values

    # Filter to valid pairs only
    result = result[valid_mask].reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Metrics and plots
# ---------------------------------------------------------------------------

def compute_feature_metrics(feat_df: pd.DataFrame) -> dict:
    feature_cols = [c for c in feat_df.columns if c not in ("label", "id1", "id2")]
    metrics = {"total_pairs": len(feat_df), "num_features": len(feature_cols)}
    for col in feature_cols:
        if col in feat_df.columns:
            vals = feat_df[col].values
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                metrics[col] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
    return metrics


def generate_feature_plots(feat_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = [c for c in feat_df.columns if c not in ("label", "id1", "id2")]

    # Correlation heatmap
    corr = feat_df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    short = [c.replace("lex_", "L_").replace("struct_", "S_").replace("quant_", "Q_")[:12]
             for c in feature_cols]
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(plots_dir / "feature_correlation.png", dpi=150)
    plt.close(fig)

    # Box plots per class
    sample_feats = ["lex_levenshtein_ratio", "struct_ast_hist_cosine", "quant_cc_ratio"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, sample_feats):
        if col not in feat_df.columns:
            continue
        pos = feat_df.loc[feat_df["label"] == 1, col].values
        neg = feat_df.loc[feat_df["label"] == 0, col].values
        ax.boxplot([neg[np.isfinite(neg)], pos[np.isfinite(pos)]],
                   tick_labels=["NON", "Type-3"])
        ax.set_title(col)
    plt.tight_layout()
    fig.savefig(plots_dir / "feature_boxplots.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Feature engineering (Zig parallel)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--intermediate-dir", type=Path, default=Path("data/intermediate"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/intermediate/features"))
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feat_lib = FeaturesLib()

    # Load token data
    token_path = args.intermediate_dir / "token_data.pkl"
    if not token_path.exists():
        logger.error("Run tokenization first")
        sys.exit(1)
    logger.info("Loading token data...")
    with open(token_path, "rb") as f:
        token_data = pickle.load(f)
    logger.info("Loaded token data for %d files", len(token_data))

    # Prepare flat arrays
    logger.info("Preparing flat arrays...")
    flat = prepare_flat_arrays(token_data)

    # Load datasets
    train_df = pd.read_csv(args.data_dir / "training_dataset.csv")
    test_df = pd.read_csv(args.data_dir / "testing_dataset.csv")

    # Compute features
    total_start = time.time()

    logger.info("=== Training features ===")
    train_features = compute_features_batch(train_df, flat, feat_lib)
    train_path = args.output_dir / "train_features.csv"
    train_features.to_csv(train_path, index=False)
    logger.info("Saved %d training features to %s", len(train_features), train_path)

    logger.info("=== Testing features ===")
    test_features = compute_features_batch(test_df, flat, feat_lib)
    test_path = args.output_dir / "test_features.csv"
    test_features.to_csv(test_path, index=False)
    logger.info("Saved %d testing features to %s", len(test_features), test_path)

    total_elapsed = time.time() - total_start
    logger.info("Total feature computation time: %.1f sec", total_elapsed)

    # Metrics
    combined = pd.concat([train_features, test_features], ignore_index=True)
    metrics = compute_feature_metrics(combined)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.eval_dir / "feature_metrics.json")

    # Plots
    plots_dir = args.eval_dir / "plots" / "features"
    generate_feature_plots(combined, plots_dir)

    print("\n=== Feature Engineering Summary ===")
    print(f"Training pairs: {len(train_features)}")
    print(f"Testing pairs: {len(test_features)}")
    n_feats = len([c for c in combined.columns if c not in ("label", "id1", "id2")])
    print(f"Features per pair: {n_feats}")
    print(f"Total time: {total_elapsed:.1f} sec")
    print("Done.")


if __name__ == "__main__":
    main()
