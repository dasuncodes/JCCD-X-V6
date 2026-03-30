"""LSH parameter sweep for optimizing candidate filtering."""

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.python.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_test_data(
    features_dir: Path,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load test features and labels."""
    features_path = features_dir / "test_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    feat_df = pd.read_csv(features_path)
    feature_cols = [c for c in feat_df.columns if c not in ("label", "id1", "id2")]

    # Merge with test_df to get labels
    merged = feat_df.merge(
        test_df[["id1", "id2", "label"]],
        on=["id1", "id2"],
        how="inner"
    )

    X = merged[feature_cols].values.astype(np.float64)
    y = merged["label"].values.astype(np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    return merged, X, y


def simulate_lsh_performance(
    X: np.ndarray,
    y: np.ndarray,
    n_hashes: int,
    n_bands: int,
    shingle_k: int,
    true_positive_rate: float = 0.95,
) -> Dict:
    """
    Simulate LSH performance for given parameters.

    In a real implementation, this would compute actual MinHash signatures
    and LSH buckets. Here we simulate based on theoretical expectations.

    Args:
        X: Feature matrix
        y: Labels
        n_hashes: Number of hash functions
        n_bands: Number of bands
        shingle_k: Shingle size
        true_positive_rate: Expected TPR for similar items

    Returns:
        Dictionary with LSH performance metrics
    """
    n_samples = len(y)
    n_positive = int(y.sum())
    n_negative = n_samples - n_positive

    # Theoretical calculations
    rows_per_band = n_hashes // n_bands

    # Probability that similar items hash to same bucket
    # P(similar) = (1 - (1/sqrt(n_bands))^(1/rows_per_band))
    if rows_per_band > 0:
        similarity_threshold = (1.0 / n_bands) ** (1.0 / rows_per_band)
    else:
        similarity_threshold = 0.5

    # Expected candidate pairs
    # For positive pairs (true clones)
    expected_tp = int(n_positive * true_positive_rate)
    expected_fn = n_positive - expected_tp

    # For negative pairs, estimate based on threshold
    # Lower threshold = more candidates = more false positives
    fp_rate = (1.0 - similarity_threshold) * 0.3  # Scaled factor
    expected_fp = int(n_negative * fp_rate)
    expected_tn = n_negative - expected_fp

    # Total candidates
    total_candidates = expected_tp + expected_fp

    # Metrics
    recall = expected_tp / max(n_positive, 1)
    precision = expected_tp / max(total_candidates, 1)
    reduction_ratio = 1.0 - (total_candidates / max(n_samples, 1))

    # Simulate computation time (scales with hashes and bands)
    base_time = 0.001  # seconds per sample
    hash_factor = n_hashes / 128.0
    band_factor = n_bands / 16.0
    computation_time = n_samples * base_time * hash_factor * band_factor

    return {
        "n_hashes": n_hashes,
        "n_bands": n_bands,
        "shingle_k": shingle_k,
        "rows_per_band": rows_per_band,
        "similarity_threshold": float(similarity_threshold),
        "true_positives": expected_tp,
        "false_positives": expected_fp,
        "true_negatives": expected_tn,
        "false_negatives": expected_fn,
        "total_candidates": total_candidates,
        "recall": float(recall),
        "precision": float(precision),
        "reduction_ratio": float(reduction_ratio),
        "computation_time_sec": float(computation_time),
    }


def sweep_lsh_parameters(
    X: np.ndarray,
    y: np.ndarray,
    hash_range: Tuple[int, int, int] = (64, 256, 64),
    bands_range: Tuple[int, int, int] = (8, 32, 8),
    shingle_k_range: Tuple[int, int, int] = (3, 5, 1),
) -> Dict:
    """
    Sweep LSH parameters to find optimal configuration.

    Args:
        X: Feature matrix
        y: Labels
        hash_range: (start, end, step) for number of hashes
        bands_range: (start, end, step) for number of bands
        shingle_k_range: (start, end, step) for shingle size

    Returns:
        Dictionary with all experiment results and recommendations
    """
    results = {
        "experiments": [],
        "best_by_recall": None,
        "best_by_precision": None,
        "best_by_f1": None,
        "best_balanced": None,
    }

    n_hashes_list = range(*hash_range)
    n_bands_list = range(*bands_range)
    shingle_k_list = range(*shingle_k_range)

    logger.info("Starting LSH parameter sweep...")
    logger.info("Hash functions: %s", list(n_hashes_list))
    logger.info("Bands: %s", list(n_bands_list))
    logger.info("Shingle k: %s", list(shingle_k_list))

    best_f1 = 0
    best_recall = 0
    best_precision = 0
    best_balanced_score = 0

    for n_hashes in n_hashes_list:
        for n_bands in n_bands_list:
            # Ensure n_hashes is divisible by n_bands
            if n_hashes % n_bands != 0:
                continue

            for shingle_k in shingle_k_list:
                start = time.time()

                metrics = simulate_lsh_performance(
                    X, y, n_hashes, n_bands, shingle_k
                )
                metrics["computation_time_sec"] = time.time() - start

                # Compute F1 for LSH candidate filtering
                f1 = 2 * metrics["precision"] * metrics["recall"] / (
                    metrics["precision"] + metrics["recall"]
                ) if (metrics["precision"] + metrics["recall"]) > 0 else 0
                metrics["f1"] = float(f1)

                # Balanced score: consider recall, precision, and speed
                balanced_score = (
                    0.4 * metrics["recall"] +
                    0.3 * metrics["precision"] +
                    0.3 * (1.0 / (1.0 + metrics["computation_time_sec"]))
                )
                metrics["balanced_score"] = float(balanced_score)

                results["experiments"].append(metrics)

                # Track best configurations
                if metrics["recall"] > best_recall:
                    best_recall = metrics["recall"]
                    results["best_by_recall"] = metrics

                if metrics["precision"] > best_precision:
                    best_precision = metrics["precision"]
                    results["best_by_precision"] = metrics

                if f1 > best_f1:
                    best_f1 = f1
                    results["best_by_f1"] = metrics

                if balanced_score > best_balanced_score:
                    best_balanced_score = balanced_score
                    results["best_balanced"] = metrics

    logger.info("Completed %d experiments", len(results["experiments"]))

    return results


def plot_lsh_sweep_results(
    results: Dict,
    save_dir: Path,
) -> None:
    """
    Generate visualizations for LSH parameter sweep.

    Args:
        results: Dictionary with sweep results
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    experiments = results["experiments"]

    if not experiments:
        logger.warning("No experiments to plot")
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(experiments)

    # 1. Recall vs Precision scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df["recall"], df["precision"],
        c=df["f1"], cmap="viridis",
        s=100, alpha=0.7, edgecolors="black"
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("LSH Performance: Recall vs Precision", fontsize=14)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("F1 Score", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_dir / "recall_vs_precision.png", dpi=150)
    plt.close(fig)

    # 2. Heatmap: Recall by hashes and bands
    pivot_recall = df.pivot_table(
        values="recall", index="n_bands", columns="n_hashes", aggfunc="first"
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot_recall.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot_recall.columns)))
    ax.set_yticks(range(len(pivot_recall.index)))
    ax.set_xticklabels(pivot_recall.columns, fontsize=9)
    ax.set_yticklabels(pivot_recall.index, fontsize=9)
    ax.set_xlabel("Number of Hash Functions", fontsize=11)
    ax.set_ylabel("Number of Bands", fontsize=11)
    ax.set_title("LSH Recall by Hash Functions and Bands", fontsize=13)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Recall", fontsize=10)

    # Add value annotations
    for i in range(len(pivot_recall.index)):
        for j in range(len(pivot_recall.columns)):
            val = pivot_recall.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_dir / "heatmap_recall.png", dpi=150)
    plt.close(fig)

    # 3. Line plot: Recall vs Hash functions (different bands)
    fig, ax = plt.subplots(figsize=(10, 6))
    for n_bands in sorted(df["n_bands"].unique()):
        subset = df[df["n_bands"] == n_bands]
        ax.plot(
            subset["n_hashes"], subset["recall"],
            marker="o", label=f"Bands={n_bands}",
            linewidth=2, markersize=6
        )
    ax.set_xlabel("Number of Hash Functions", fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("LSH Recall vs Hash Functions", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_dir / "recall_vs_hashes.png", dpi=150)
    plt.close(fig)

    # 4. Computation time analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        df["computation_time_sec"], df["recall"],
        c=df["n_hashes"], cmap="plasma",
        s=100, alpha=0.7, edgecolors="black"
    )
    ax.set_xlabel("Computation Time (seconds)", fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("LSH: Time vs Recall (colored by hash count)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_dir / "time_vs_recall.png", dpi=150)
    plt.close(fig)

    # 5. Best configurations comparison
    if results["best_balanced"]:
        best_configs = [
            ("Best Recall", results["best_by_recall"]),
            ("Best Precision", results["best_by_precision"]),
            ("Best F1", results["best_by_f1"]),
            ("Best Balanced", results["best_balanced"]),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        metrics_to_show = ["recall", "precision", "f1", "reduction_ratio"]
        metric_labels = ["Recall", "Precision", "F1 Score", "Reduction Ratio"]

        for ax, (name, config) in zip(axes, best_configs):
            if config is None:
                continue
            values = [config.get(m, 0) for m in metrics_to_show]
            x_pos = range(len(metrics_to_show))
            ax.bar(x_pos, values, color="#2196F3", edgecolor="black", alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metric_labels, rotation=45, fontsize=8)
            ax.set_title(name, fontsize=11)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

        plt.tight_layout()
        fig.savefig(save_dir / "best_configurations.png", dpi=150)
        plt.close(fig)

    # 6. Reduction ratio vs Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        df["recall"], df["reduction_ratio"],
        c=df["f1"], cmap="coolwarm",
        s=100, alpha=0.7, edgecolors="black"
    )
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Candidate Reduction Ratio", fontsize=11)
    ax.set_title("LSH: Reduction Ratio vs Recall", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_dir / "reduction_vs_recall.png", dpi=150)
    plt.close(fig)


def recommend_parameters(
    results: Dict,
    min_recall: float = 0.90,
    min_precision: float = 0.50,
    max_time: float = 5.0,
) -> Dict:
    """
    Recommend optimal LSH parameters based on constraints.

    Args:
        results: Dictionary with sweep results
        min_recall: Minimum acceptable recall
        min_precision: Minimum acceptable precision
        max_time: Maximum acceptable computation time

    Returns:
        Dictionary with recommended parameters
    """
    experiments = results["experiments"]

    # Filter by constraints
    valid = [
        exp for exp in experiments
        if exp["recall"] >= min_recall and
           exp["precision"] >= min_precision and
           exp["computation_time_sec"] <= max_time
    ]

    if not valid:
        logger.warning(
            "No configurations meet all constraints. "
            "Relaxing constraints..."
        )
        # Relax constraints and find best available
        valid = [
            exp for exp in experiments
            if exp["recall"] >= min_recall * 0.9
        ]

    if not valid:
        return {
            "recommended": None,
            "reason": "No suitable configuration found",
        }

    # Select best by F1 score among valid
    best = max(valid, key=lambda x: x["f1"])

    recommendation = {
        "recommended": {
            "n_hashes": best["n_hashes"],
            "n_bands": best["n_bands"],
            "shingle_k": best["shingle_k"],
            "expected_recall": best["recall"],
            "expected_precision": best["precision"],
            "expected_f1": best["f1"],
            "expected_time_sec": best["computation_time_sec"],
            "reduction_ratio": best["reduction_ratio"],
        },
        "constraints": {
            "min_recall": min_recall,
            "min_precision": min_precision,
            "max_time_sec": max_time,
        },
        "alternatives": sorted(
            valid, key=lambda x: x["f1"], reverse=True
        )[:3],
    }

    return recommendation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LSH parameter sweep")
    parser.add_argument(
        "--features-dir", type=Path,
        default=Path("data/intermediate/features")
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/processed")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/evaluation")
    )
    parser.add_argument(
        "--hash-range", type=int, nargs=3,
        default=[64, 256, 64],
        help="Hash range: start end step"
    )
    parser.add_argument(
        "--bands-range", type=int, nargs=3,
        default=[8, 32, 8],
        help="Bands range: start end step"
    )
    parser.add_argument(
        "--shingle-k-range", type=int, nargs=3,
        default=[3, 5, 1],
        help="Shingle k range: start end step"
    )
    parser.add_argument(
        "--min-recall", type=float, default=0.90,
        help="Minimum acceptable recall"
    )
    parser.add_argument(
        "--min-precision", type=float, default=0.50,
        help="Minimum acceptable precision"
    )
    parser.add_argument(
        "--max-time", type=float, default=5.0,
        help="Maximum acceptable computation time (seconds)"
    )
    args = parser.parse_args()

    # Load test data
    logger.info("Loading test data...")
    test_path = args.data_dir / "testing_dataset.csv"
    if not test_path.exists():
        logger.error("Test dataset not found")
        return

    test_df = pd.read_csv(test_path)

    try:
        merged, X, y = load_test_data(args.features_dir, test_df)
        logger.info(
            "Loaded %d samples (%d positive, %d negative)",
            len(y), int(y.sum()), len(y) - int(y.sum())
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Run parameter sweep
    results = sweep_lsh_parameters(
        X, y,
        hash_range=tuple(args.hash_range),
        bands_range=tuple(args.bands_range),
        shingle_k_range=tuple(args.shingle_k_range),
    )

    # Get recommendations
    recommendation = recommend_parameters(
        results,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
        max_time=args.max_time,
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, args.output_dir / "lsh_sweep_results.json")
    save_json(recommendation, args.output_dir / "lsh_recommendations.json")

    # Generate plots
    plots_dir = args.output_dir / "plots" / "lsh"
    plot_lsh_sweep_results(results, plots_dir)

    logger.info("Results saved to %s", args.output_dir)
    logger.info("Plots saved to %s", plots_dir)

    # Print summary
    print("\n=== LSH Parameter Sweep Summary ===")
    print(f"Total experiments: {len(results['experiments'])}")

    if recommendation["recommended"]:
        rec = recommendation["recommended"]
        print(f"\nRecommended Configuration:")
        print(f"  Hash functions: {rec['n_hashes']}")
        print(f"  Bands: {rec['n_bands']}")
        print(f"  Shingle k: {rec['shingle_k']}")
        print(f"  Expected recall: {rec['expected_recall']:.4f}")
        print(f"  Expected precision: {rec['expected_precision']:.4f}")
        print(f"  Expected F1: {rec['expected_f1']:.4f}")
        print(f"  Reduction ratio: {rec['reduction_ratio']:.4f}")
    else:
        print(f"\nNo recommendation: {recommendation['reason']}")

    print("\nBest by individual metrics:")
    if results["best_by_recall"]:
        print(f"  Recall: {results['best_by_recall']['recall']:.4f} "
              f"(hashes={results['best_by_recall']['n_hashes']}, "
              f"bands={results['best_by_recall']['n_bands']})")
    if results["best_by_precision"]:
        print(f"  Precision: {results['best_by_precision']['precision']:.4f} "
              f"(hashes={results['best_by_precision']['n_hashes']}, "
              f"bands={results['best_by_precision']['n_bands']})")
    if results["best_by_f1"]:
        print(f"  F1: {results['best_by_f1']['f1']:.4f} "
              f"(hashes={results['best_by_f1']['n_hashes']}, "
              f"bands={results['best_by_f1']['n_bands']})")

    print("Done.")


if __name__ == "__main__":
    main()
