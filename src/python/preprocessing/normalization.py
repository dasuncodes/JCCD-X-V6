"""Source code normalization: strip comments, whitespace, standardize formatting."""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.bindings.normalization import NormalizationLib
from src.python.utils.io import load_source_code, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_all(
    df: pd.DataFrame,
    source_dir: Path,
    norm: NormalizationLib,
    output_dir: Path,
) -> pd.DataFrame:
    """Normalize source code for all unique IDs in the dataframe."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.unique(df[["id1", "id2"]].values.ravel())
    logger.info("Normalizing %d unique source files...", len(ids))

    results = []
    processed = 0
    errors = 0

    for fid in ids:
        src = load_source_code(fid, source_dir)
        if src is None:
            errors += 1
            continue
        try:
            normalized = norm.normalize_source(src)
            out_path = output_dir / f"{fid}.java"
            out_path.write_text(normalized, encoding="utf-8")

            orig_lines = src.count("\n") + 1
            norm_lines = normalized.count("\n") + 1 if normalized else 0
            orig_tokens = len(src.split())
            norm_tokens = len(normalized.split()) if normalized else 0
            orig_chars = len(src)
            norm_chars = len(normalized)

            results.append({
                "id": fid,
                "original_lines": orig_lines,
                "normalized_lines": norm_lines,
                "original_tokens": orig_tokens,
                "normalized_tokens": norm_tokens,
                "original_chars": orig_chars,
                "normalized_chars": norm_chars,
                "lines_removed": orig_lines - norm_lines,
                "tokens_removed": orig_tokens - norm_tokens,
                "chars_removed": orig_chars - norm_chars,
            })
        except Exception as e:
            logger.error("Failed to normalize %s: %s", fid, e)
            errors += 1

        processed += 1
        if processed % 5000 == 0:
            logger.info("  Processed %d / %d", processed, len(ids))

    logger.info("Normalization complete: %d processed, %d errors", processed, errors)
    return pd.DataFrame(results)


def compute_normalization_metrics(impact_df: pd.DataFrame) -> dict:
    """Compute aggregate normalization impact metrics."""
    if impact_df.empty:
        return {}
    return {
        "total_files": len(impact_df),
        "avg_lines_before": float(impact_df["original_lines"].mean()),
        "avg_lines_after": float(impact_df["normalized_lines"].mean()),
        "avg_lines_removed": float(impact_df["lines_removed"].mean()),
        "avg_tokens_before": float(impact_df["original_tokens"].mean()),
        "avg_tokens_after": float(impact_df["normalized_tokens"].mean()),
        "avg_tokens_removed": float(impact_df["tokens_removed"].mean()),
        "avg_chars_before": float(impact_df["original_chars"].mean()),
        "avg_chars_after": float(impact_df["normalized_chars"].mean()),
        "avg_chars_removed": float(impact_df["chars_removed"].mean()),
        "total_lines_removed": int(impact_df["lines_removed"].sum()),
        "total_tokens_removed": int(impact_df["tokens_removed"].sum()),
    }


def _safe_hist(ax, data, **kwargs):
    """Plot histogram safely, falling back to bar chart if ax.hist fails."""
    try:
        ax.hist(data, **kwargs)
    except (ValueError, TypeError):
        try:
            counts, bin_edges = np.histogram(data, bins=kwargs.get("bins", 50))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            width = (bin_edges[1] - bin_edges[0]) * 0.9
            ax.bar(bin_centers, counts, width=width,
                   alpha=kwargs.get("alpha", 0.7),
                   edgecolor=kwargs.get("edgecolor", "black"),
                   label=kwargs.get("label", None))
        except Exception:
            pass


def generate_normalization_plots(impact_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title) in zip(
        axes,
        [
            ("lines_removed", "Lines Removed per File"),
            ("tokens_removed", "Tokens Removed per File"),
            ("chars_removed", "Characters Removed per File"),
        ],
    ):
        data = impact_df[col].clip(lower=0).values
        data = data[np.isfinite(data)]
        if len(data) > 0 and data.max() > data.min():
            _safe_hist(ax, data, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel(title.replace(" per File", ""))
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        mean_val = float(impact_df[col].mean())
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean={mean_val:.1f}")
        ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "normalization_impact.png", dpi=150)
    plt.close(fig)

    # Before vs after comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col_b, col_a, label in [
        (axes[0], "original_tokens", "normalized_tokens", "Token Count"),
        (axes[1], "original_lines", "normalized_lines", "Line Count"),
    ]:
        for col, lbl in [(col_b, "Before"), (col_a, "After")]:
            data = impact_df[col].values
            data = data[np.isfinite(data)]
            if len(data) > 0 and data.max() > data.min():
                _safe_hist(ax, data, bins=50, alpha=0.6, label=lbl, edgecolor="black")
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{label}: Before vs After Normalization")
        ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "before_vs_after.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Source code normalization")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/toma/id2sourcecode"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/normalized"))
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()

    norm = NormalizationLib()

    # Load training and testing datasets
    train_path = args.data_dir / "training_dataset.csv"
    test_path = args.data_dir / "testing_dataset.csv"
    if not train_path.exists() or not test_path.exists():
        logger.error("Run data preparation first (scripts/run_data_prep.sh)")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    logger.info("Combined dataset: %d rows", len(combined))

    # Normalize
    impact_df = normalize_all(combined, args.source_dir, norm, args.output_dir)

    # Save impact data
    impact_path = args.data_dir / "normalization_impact.csv"
    impact_df.to_csv(impact_path, index=False)
    logger.info("Saved normalization impact to %s", impact_path)

    # Metrics
    metrics = compute_normalization_metrics(impact_df)
    eval_dir = args.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, eval_dir / "normalization_metrics.json")
    logger.info("Saved normalization_metrics.json")

    # Plots
    plots_dir = eval_dir / "plots" / "normalization"
    generate_normalization_plots(impact_df, plots_dir)
    logger.info("Plots saved to %s", plots_dir)

    print("\n=== Normalization Summary ===")
    print(f"Files processed: {metrics.get('total_files', '?')}")
    print(f"Avg lines removed: {metrics.get('avg_lines_removed', 0):.1f}")
    print(f"Avg tokens removed: {metrics.get('avg_tokens_removed', 0):.1f}")
    print(f"Avg chars removed: {metrics.get('avg_chars_removed', 0):.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
