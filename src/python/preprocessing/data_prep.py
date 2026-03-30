"""Data preparation: load raw CSVs, validate, split into train/test datasets."""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.python.utils.io import (
    load_csv_pairs,
    load_source_code,
    save_csv,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_datasets(raw_dir: Path) -> dict[str, pd.DataFrame]:
    datasets = {}
    csv_files = {
        "type1": ("type-1.csv", False),
        "type2": ("type-2.csv", False),
        "type3": ("type-3.csv", False),
        "type4": ("type-4.csv", False),
        "type5": ("type-5.csv", False),
        "nonclone": ("nonclone.csv", True),
    }
    for key, (filename, has_header) in csv_files.items():
        path = raw_dir / filename
        if not path.exists():
            logger.warning("Missing file: %s", path)
            continue
        df = load_csv_pairs(path, has_header=has_header)
        datasets[key] = df
        logger.info("Loaded %s: %d rows from %s", key, len(df), filename)
    return datasets


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def assign_labels(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Assign labels for training and testing.

    Training set (binary):
    - label=1 for type-3 clones (type3 + type4)
    - label=0 for NON (type5 + nonclone)

    Testing set (multi-class):
    - label=1 for type-1 clones
    - label=2 for type-2 clones
    - label=3 for type-3 clones (type3 + type4)
    - label=0 for NON (type5 + nonclone)
    """
    labelled = {}
    for key, df in datasets.items():
        df = df.copy()
        # For testing: preserve actual type labels
        if key == "type1":
            df["label"] = 1  # Type-1 clones
        elif key == "type2":
            df["label"] = 2  # Type-2 clones
        elif key in ("type3", "type4"):
            df["label"] = 3  # Type-3 clones
        elif key in ("type5", "nonclone"):
            df["label"] = 0  # Non-clones
        else:
            df["label"] = -1  # Unknown
        labelled[key] = df
    return labelled


def assign_training_labels(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Assign binary labels for training set only.

    Training set (binary):
    - label=1 for type-3 clones (type3 + type4)
    - label=0 for NON (type5 + nonclone)
    - label=-1 for type1, type2 (excluded from training)
    """
    labelled = {}
    for key, df in datasets.items():
        df = df.copy()
        if key in ("type3", "type4"):
            df["label"] = 1
        elif key in ("type5", "nonclone"):
            df["label"] = 0
        else:
            df["label"] = -1  # type1, type2 — not used for ML training
        labelled[key] = df
    return labelled


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_source_files(
    datasets: dict[str, pd.DataFrame], source_dir: Path
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Remove pairs where either function ID has no .java file."""
    removal_stats = {}
    validated = {}

    for key, df in datasets.items():
        original_count = len(df)
        mask = df.apply(lambda row: (source_dir / f"{row['id1']}.java").exists()
                        and (source_dir / f"{row['id2']}.java").exists(), axis=1)
        df_valid = df[mask].reset_index(drop=True)
        removed = original_count - len(df_valid)
        removal_stats[key] = {
            "original": original_count,
            "valid": len(df_valid),
            "removed": removed,
        }
        validated[key] = df_valid
        logger.info(
            "%s: %d valid / %d original (%d removed)",
            key, len(df_valid), original_count, removed,
        )
    return validated, removal_stats


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def balance_training_set(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Undersample majority class to balance the training set."""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    min_count = min(len(pos), len(neg))
    if len(pos) == len(neg):
        logger.info("Training set already balanced: %d per class", len(pos))
        return df
    pos_sample = pos.sample(n=min_count, random_state=seed)
    neg_sample = neg.sample(n=min_count, random_state=seed)
    balanced = pd.concat([pos_sample, neg_sample], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        "Balanced training set: %d positive, %d negative -> %d total",
        len(pos_sample), len(neg_sample), len(balanced),
    )
    return balanced


def build_datasets(
    validated: dict[str, pd.DataFrame],  # For type1, type2 (multi-class labels)
    validated_train: dict[str, pd.DataFrame],  # For type3, type4, type5, nonclone (multi-class labels)
    test_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct training and testing datasets.

    Training set (binary, balanced):
    - 70% of type3+type4 (label→1) and type5+nonclone (label→0)
    - Balanced classes
    - Type-1 and Type-2 excluded

    Testing set (multi-class, original distribution):
    - 100% of type-1 (label=1)
    - 100% of type-2 (label=2)
    - 30% of type-3+type-4 (label=3)
    - 30% of type-5+nonclone (label=0)
    """
    # Split ABC dataset (type3, type4, type5, nonclone) from validated_train
    # These have multi-class labels (0 and 3)
    abc_parts = []
    for key in ("type3", "type4", "type5", "nonclone"):
        if key in validated_train:
            abc_parts.append(validated_train[key])
    abc = pd.concat(abc_parts, ignore_index=True)
    logger.info("ABC dataset: %d rows (type3+type4+type5+nonclone)", len(abc))

    # Create binary labels for stratified split (don't modify original label column)
    abc_binary = abc.copy()
    abc_binary["train_label"] = abc_binary["label"].apply(lambda x: 1 if x == 3 else 0)

    train_abc, test_abc = train_test_split(
        abc_binary, test_size=test_ratio, random_state=seed, stratify=abc_binary["train_label"]
    )

    # Drop the temporary train_label column, keep original multi-class labels
    train_abc = train_abc.drop(columns=["train_label"]).reset_index(drop=True)
    test_abc = test_abc.drop(columns=["train_label"]).reset_index(drop=True)

    # Convert training set to binary labels (0 and 1)
    train_abc["label"] = train_abc["label"].apply(lambda x: 1 if x == 3 else 0)

    # Balance the training set (binary: Type-3 vs Non-clone)
    train_abc = balance_training_set(train_abc, seed=seed)

    # Testing set: preserve multi-class labels
    # Start with type-3, type-4, type-5, nonclone (30% split, labels 0 and 3)
    testing_parts = [test_abc]

    # Add type-1 and type-2 (100%, from validated with multi-class labels 1 and 2)
    for key in ("type1", "type2"):
        if key in validated:
            testing_parts.append(validated[key])
            logger.info("Added %s to test set: %d rows", key, len(validated[key]))

    testing = pd.concat(testing_parts, ignore_index=True)

    logger.info("Training set: %d rows (balanced, binary)", len(train_abc))
    logger.info("Testing set: %d rows (multi-class: type-1/2/3/non)", len(testing))

    # Log test set distribution
    test_distribution = testing["label"].value_counts().to_dict()
    logger.info("Test set distribution: %s", {
        0: test_distribution.get(0, 0),  # Non-clones
        1: test_distribution.get(1, 0),  # Type-1
        2: test_distribution.get(2, 0),  # Type-2
        3: test_distribution.get(3, 0),  # Type-3
    })

    return train_abc, testing


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_length_stats(source_dir: Path, df: pd.DataFrame) -> dict:
    lines_list, tokens_list, chars_list = [], [], []
    ids = pd.unique(df[["id1", "id2"]].values.ravel())
    for fid in ids:
        src = load_source_code(fid, source_dir)
        if src is None:
            continue
        lines_list.append(src.count("\n") + 1)
        tokens_list.append(len(src.split()))
        chars_list.append(len(src))
    return {
        "lines": {"mean": float(np.mean(lines_list)), "std": float(np.std(lines_list)),
                   "min": int(np.min(lines_list)), "max": int(np.max(lines_list)),
                   "median": float(np.median(lines_list))},
        "tokens": {"mean": float(np.mean(tokens_list)), "std": float(np.std(tokens_list)),
                    "min": int(np.min(tokens_list)), "max": int(np.max(tokens_list)),
                    "median": float(np.median(tokens_list))},
        "chars": {"mean": float(np.mean(chars_list)), "std": float(np.std(chars_list)),
                   "min": int(np.min(chars_list)), "max": int(np.max(chars_list)),
                   "median": float(np.median(chars_list))},
        "sample_count": len(lines_list),
    }


def compute_split_metrics(
    validated: dict, train: pd.DataFrame, test: pd.DataFrame
) -> dict:
    metrics = {}
    for key, df in validated.items():
        metrics[f"{key}_size"] = len(df)
    metrics["training_size"] = len(train)
    metrics["testing_size"] = len(test)
    total = len(train) + len(test)
    if total > 0:
        metrics["train_ratio"] = round(len(train) / total, 4)
        metrics["test_ratio"] = round(len(test) / total, 4)
    # Label distribution in training set
    if "label" in train.columns:
        pos = int((train["label"] == 1).sum())
        neg = int((train["label"] == 0).sum())
        metrics["train_positive"] = pos
        metrics["train_negative"] = neg
        metrics["train_balance_ratio"] = round(pos / max(neg, 1), 4)
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_samples_per_class(counts: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = list(counts.keys())
    vals = [counts[k] for k in keys]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#607D8B"]
    ax.bar(keys, vals, color=colors[: len(keys)])
    ax.set_ylabel("Number of samples")
    ax.set_title("Samples per clone type (after validation)")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_code_length_histogram(
    lengths: list[int], title: str, xlabel: str, save_path: Path
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.axvline(np.mean(lengths), color="red", linestyle="--", label=f"Mean={np.mean(lengths):.1f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_data_prep_plots(
    validated: dict[str, pd.DataFrame], source_dir: Path, plots_dir: Path
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Samples per class
    counts = {k: len(v) for k, v in validated.items()}
    plot_samples_per_class(counts, plots_dir / "samples_per_class.png")

    # Code length histograms (sample unique IDs)
    all_ids = set()
    for df in validated.values():
        all_ids.update(df["id1"].tolist())
        all_ids.update(df["id2"].tolist())
    lines_list, tokens_list = [], []
    for fid in list(all_ids)[:5000]:  # sample for speed
        src = load_source_code(fid, source_dir)
        if src is None:
            continue
        lines_list.append(src.count("\n") + 1)
        tokens_list.append(len(src.split()))
    if lines_list:
        plot_code_length_histogram(
            lines_list, "Lines per snippet", "Lines",
            plots_dir / "lines_per_snippet.png",
        )
    if tokens_list:
        plot_code_length_histogram(
            tokens_list, "Tokens per snippet (whitespace split)", "Tokens",
            plots_dir / "tokens_per_snippet.png",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Data preparation for JCCD-X")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/toma/id2sourcecode"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/toma"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.eval_dir / "plots" / "data_prep"

    # 1. Load
    logger.info("Loading datasets...")
    datasets = load_all_datasets(args.raw_dir)

    # 2. Label for testing (multi-class)
    logger.info("Assigning labels (multi-class for testing)...")
    labelled = assign_labels(datasets)

    # 3. Validate
    logger.info("Validating source files...")
    validated, removal_stats = validate_source_files(labelled, args.source_dir)

    # 4. Re-label for training (binary) - ONLY for training set
    logger.info("Re-labeling for training (binary)...")
    # For training, we need binary labels but for testing we need multi-class
    # So we create a separate copy for training with binary labels
    validated_train = {}
    for key in ("type3", "type4", "type5", "nonclone"):
        if key in validated:
            df = validated[key].copy()
            # Keep original multi-class labels for test set
            # Training set will use binary labels via the split logic
            validated_train[key] = df

    # 5. Split
    logger.info("Building train/test splits...")
    training, testing = build_datasets(
        validated,  # For type1, type2 (multi-class)
        validated_for_train,  # For type3, type4, type5, nonclone (binary)
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # 5. Save
    save_csv(training, args.output_dir / "training_dataset.csv")
    save_csv(testing, args.output_dir / "testing_dataset.csv")
    logger.info("Saved training_dataset.csv and testing_dataset.csv")

    # 6. Metrics
    split_metrics = compute_split_metrics(validated, training, testing)
    split_metrics["removal_stats"] = removal_stats

    logger.info("Computing code length stats (training)...")
    split_metrics["training_length_stats"] = compute_length_stats(args.source_dir, training)
    logger.info("Computing code length stats (testing)...")
    split_metrics["testing_length_stats"] = compute_length_stats(args.source_dir, testing)

    save_json(split_metrics, args.eval_dir / "data_prep_metrics.json")
    logger.info("Saved data_prep_metrics.json")

    # 7. Plots
    logger.info("Generating plots...")
    generate_data_prep_plots(validated, args.source_dir, plots_dir)
    logger.info("Plots saved to %s", plots_dir)

    # Summary
    print("\n=== Data Preparation Summary ===")
    print(f"Training samples: {len(training)}")
    print(f"Testing samples:  {len(testing)}")
    if "label" in training.columns:
        print(f"Training balance: {split_metrics.get('train_positive', '?')} positive, "
              f"{split_metrics.get('train_negative', '?')} negative")
    print(f"Metrics: {args.eval_dir / 'data_prep_metrics.json'}")
    print(f"Plots:   {plots_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
