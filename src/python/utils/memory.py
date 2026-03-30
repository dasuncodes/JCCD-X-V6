"""Memory usage profiling for pipeline optimization."""

import argparse
import logging
import pickle
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.python.evaluation.plots import plot_memory_usage, plot_runtime_per_stage
from src.python.model.models import get_models
from src.python.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Context manager for memory profiling."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_mem = 0
        self.end_mem = 0
        self.peak_mem = 0
        self.start_time = 0
        self.end_time = 0

    def __enter__(self):
        tracemalloc.start()
        self.start_time = time.time()
        self.start_mem, _ = tracemalloc.get_traced_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        self.end_mem = current
        self.peak_mem = peak
        tracemalloc.stop()

    @property
    def memory_used_mb(self) -> float:
        return (self.end_mem - self.start_mem) / (1024 * 1024)

    @property
    def peak_memory_mb(self) -> float:
        return self.peak_mem / (1024 * 1024)

    @property
    def elapsed_time(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "memory_used_mb": self.memory_used_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "elapsed_time_sec": self.elapsed_time,
        }


def profile_data_loading(
    features_dir: Path,
    data_dir: Path,
) -> Dict:
    """
    Profile memory usage of data loading operations.

    Args:
        features_dir: Directory with feature files
        data_dir: Directory with dataset files

    Returns:
        Dictionary with profiling results
    """
    results = {"stages": []}

    # Profile training data loading
    with MemoryProfiler("Load training features") as profiler:
        train_df = pd.read_csv(features_dir / "train_features.csv")
    results["stages"].append(profiler.to_dict())
    logger.info("Training features: %.2f MB, %.2f sec",
                profiler.memory_used_mb, profiler.elapsed_time)

    # Profile test data loading
    with MemoryProfiler("Load test features") as profiler:
        test_df = pd.read_csv(features_dir / "test_features.csv")
    results["stages"].append(profiler.to_dict())
    logger.info("Test features: %.2f MB, %.2f sec",
                profiler.memory_used_mb, profiler.elapsed_time)

    # Profile dataset loading
    with MemoryProfiler("Load training dataset") as profiler:
        dataset_df = pd.read_csv(data_dir / "training_dataset.csv")
    results["stages"].append(profiler.to_dict())
    logger.info("Training dataset: %.2f MB, %.2f sec",
                profiler.memory_used_mb, profiler.elapsed_time)

    # Profile token data loading (if exists)
    token_path = features_dir.parent / "token_data.pkl"
    if token_path.exists():
        with MemoryProfiler("Load token data") as profiler:
            with open(token_path, "rb") as f:
                token_data = pickle.load(f)
        results["stages"].append(profiler.to_dict())
        logger.info("Token data: %.2f MB, %.2f sec",
                    profiler.memory_used_mb, profiler.elapsed_time)

    # Summary
    results["total_memory_mb"] = sum(s["memory_used_mb"] for s in results["stages"])
    results["total_time_sec"] = sum(s["elapsed_time_sec"] for s in results["stages"])
    results["peak_memory_mb"] = max(s["peak_memory_mb"] for s in results["stages"])

    return results


def profile_feature_processing(
    features_dir: Path,
    cv: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Profile memory usage of feature processing.

    Args:
        features_dir: Directory with feature files
        cv: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary with profiling results
    """
    results = {"stages": []}

    # Load data
    train_df = pd.read_csv(features_dir / "train_features.csv")
    feature_cols = [c for c in train_df.columns if c not in ("label", "id1", "id2")]
    X = train_df[feature_cols].values.astype(np.float64)
    y = train_df["label"].values.astype(np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    # Profile feature normalization
    with MemoryProfiler("Feature normalization (NaN handling)") as profiler:
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    results["stages"].append(profiler.to_dict())

    # Profile model training (per fold)
    model = get_models()["xgboost"]
    fold_memories = []

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        with MemoryProfiler(f"Model training (fold {fold_idx + 1})") as profiler:
            X_train, y_train = X[train_idx], y[train_idx]
            model.fit(X_train, y_train)
        fold_memories.append(profiler.to_dict())

    results["stages"].extend(fold_memories)

    # Profile prediction
    with MemoryProfiler("Model prediction") as profiler:
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
    results["stages"].append(profiler.to_dict())

    # Summary
    results["total_memory_mb"] = sum(s["memory_used_mb"] for s in results["stages"])
    results["total_time_sec"] = sum(s["elapsed_time_sec"] for s in results["stages"])
    results["peak_memory_mb"] = max(s["peak_memory_mb"] for s in results["stages"])
    results["avg_fold_memory_mb"] = np.mean([f["memory_used_mb"] for f in fold_memories])

    return results


def profile_lsh_computation(
    n_samples: int = 10000,
    n_features: int = 20,
    n_hashes: int = 128,
    n_bands: int = 16,
) -> Dict:
    """
    Profile memory usage of LSH computation.

    Args:
        n_samples: Number of samples to simulate
        n_features: Number of features
        n_hashes: Number of hash functions
        n_bands: Number of bands

    Returns:
        Dictionary with profiling results
    """
    results = {"stages": []}

    # Simulate shingle data
    np.random.seed(42)
    shingles = np.random.randint(0, 1000000, size=(n_samples * 100)).astype(np.uint64)

    # Profile shingle storage
    with MemoryProfiler("Shingle storage") as profiler:
        _ = shingles.nbytes
    results["stages"].append(profiler.to_dict())

    # Profile MinHash signature computation
    with MemoryProfiler("MinHash signature computation") as profiler:
        signatures = np.zeros((n_samples, n_hashes), dtype=np.uint64)
        for i in range(n_samples):
            signatures[i] = np.random.randint(0, 2**64, n_hashes, dtype=np.uint64)
    results["stages"].append(profiler.to_dict())
    logger.info("MinHash signatures: %.2f MB, %.2f sec",
                profiler.memory_used_mb, profiler.elapsed_time)

    # Profile LSH bucket assignment
    with MemoryProfiler("LSH bucket assignment") as profiler:
        rows_per_band = n_hashes // n_bands
        buckets = {}
        for i in range(n_samples):
            for b in range(n_bands):
                bucket_key = (b, hash(tuple(signatures[i, b*rows_per_band:(b+1)*rows_per_band])))
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(i)
    results["stages"].append(profiler.to_dict())
    logger.info("LSH buckets: %.2f MB, %.2f sec",
                profiler.memory_used_mb, profiler.elapsed_time)

    # Profile candidate pair generation
    with MemoryProfiler("Candidate pair generation") as profiler:
        candidates = set()
        for bucket_ids in buckets.values():
            if len(bucket_ids) > 500:
                continue
            for i in range(len(bucket_ids)):
                for j in range(i + 1, len(bucket_ids)):
                    candidates.add((min(bucket_ids[i], bucket_ids[j]),
                                   max(bucket_ids[i], bucket_ids[j])))
    results["stages"].append(profiler.to_dict())
    logger.info("Candidate pairs: %d, %.2f MB",
                len(candidates), profiler.memory_used_mb)

    # Summary
    results["total_memory_mb"] = sum(s["memory_used_mb"] for s in results["stages"])
    results["total_time_sec"] = sum(s["elapsed_time_sec"] for s in results["stages"])
    results["peak_memory_mb"] = max(s["peak_memory_mb"] for s in results["stages"])
    results["n_candidates"] = len(candidates)

    return results


def profile_full_pipeline(
    features_dir: Path,
    data_dir: Path,
    model_dir: Path,
    cv: int = 5,
) -> Dict:
    """
    Profile memory usage of full pipeline.

    Args:
        features_dir: Directory with feature files
        data_dir: Directory with dataset files
        model_dir: Directory with saved models
        cv: Number of CV folds

    Returns:
        Dictionary with profiling results
    """
    results = {
        "stages": [],
        "summary": {},
    }

    total_memory = 0
    peak_memory = 0
    total_time = 0

    # Stage 1: Data loading
    logger.info("=== Stage 1: Data Loading ===")
    with MemoryProfiler("Data loading") as profiler:
        train_df = pd.read_csv(features_dir / "train_features.csv")
        test_df = pd.read_csv(features_dir / "test_features.csv")
    results["stages"].append(profiler.to_dict())
    total_memory += profiler.memory_used_mb
    total_time += profiler.elapsed_time
    peak_memory = max(peak_memory, profiler.peak_memory_mb)

    # Stage 2: Feature preparation
    logger.info("=== Stage 2: Feature Preparation ===")
    with MemoryProfiler("Feature preparation") as profiler:
        feature_cols = [c for c in train_df.columns if c not in ("label", "id1", "id2")]
        X_train = train_df[feature_cols].values.astype(np.float64)
        y_train = train_df["label"].values.astype(np.int32)
        X_test = test_df[feature_cols].values.astype(np.float64)
        y_test = test_df["label"].values.astype(np.int32)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=0.0)
    results["stages"].append(profiler.to_dict())
    total_memory += profiler.memory_used_mb
    total_time += profiler.elapsed_time
    peak_memory = max(peak_memory, profiler.peak_memory_mb)

    # Stage 3: Model training
    logger.info("=== Stage 3: Model Training ===")
    with MemoryProfiler("Model training") as profiler:
        model = get_models()["xgboost"]
        model.fit(X_train, y_train)
    results["stages"].append(profiler.to_dict())
    total_memory += profiler.memory_used_mb
    total_time += profiler.elapsed_time
    peak_memory = max(peak_memory, profiler.peak_memory_mb)

    # Stage 4: Model evaluation
    logger.info("=== Stage 4: Model Evaluation ===")
    with MemoryProfiler("Model evaluation") as profiler:
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
    results["stages"].append(profiler.to_dict())
    total_memory += profiler.memory_used_mb
    total_time += profiler.elapsed_time
    peak_memory = max(peak_memory, profiler.peak_memory_mb)

    # Stage 5: Model saving
    logger.info("=== Stage 5: Model Saving ===")
    with MemoryProfiler("Model saving") as profiler:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "profiled_model.joblib")
    results["stages"].append(profiler.to_dict())
    total_memory += profiler.memory_used_mb
    total_time += profiler.elapsed_time
    peak_memory = max(peak_memory, profiler.peak_memory_mb)

    # Summary
    results["summary"] = {
        "total_memory_mb": total_memory,
        "peak_memory_mb": peak_memory,
        "total_time_sec": total_time,
        "n_stages": len(results["stages"]),
    }

    return results


def generate_memory_report(
    results: Dict,
    save_dir: Path,
) -> None:
    """
    Generate memory profiling report with visualizations.

    Args:
        results: Dictionary with profiling results
        save_dir: Directory to save reports
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Memory usage by stage
    if "stages" in results:
        stages = [s["name"] for s in results["stages"]]
        memories = [s["memory_used_mb"] for s in results["stages"]]
        times = [s["elapsed_time_sec"] for s in results["stages"]]

        # Memory bar chart
        plot_memory_usage(stages, memories, save_dir / "memory_by_stage.png")

        # Runtime bar chart
        plot_runtime_per_stage(stages, times, save_dir / "runtime_by_stage.png")

        # Memory vs Time scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(memories, times, s=100, alpha=0.7, edgecolors="black")

        for i, stage in enumerate(stages):
            short_name = stage[:20] + "..." if len(stage) > 20 else stage
            ax.annotate(short_name, (memories[i], times[i]), fontsize=8)

        ax.set_xlabel("Memory Used (MB)", fontsize=11)
        ax.set_ylabel("Time (seconds)", fontsize=11)
        ax.set_title("Memory vs Time by Stage", fontsize=13)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_dir / "memory_vs_time.png", dpi=150)
        plt.close(fig)


def get_optimization_recommendations(
    results: Dict,
) -> Dict:
    """
    Generate optimization recommendations based on profiling results.

    Args:
        results: Dictionary with profiling results

    Returns:
        Dictionary with recommendations
    """
    recommendations = []

    if "stages" in results:
        # Find memory-heavy stages
        stages = results["stages"]
        max_mem_stage = max(stages, key=lambda s: s["memory_used_mb"])
        max_time_stage = max(stages, key=lambda s: s["elapsed_time_sec"])

        if max_mem_stage["memory_used_mb"] > 100:
            recommendations.append({
                "stage": max_mem_stage["name"],
                "issue": "High memory usage",
                "value_mb": max_mem_stage["memory_used_mb"],
                "suggestion": "Consider processing in batches or using generators",
            })

        if max_time_stage["elapsed_time_sec"] > 10:
            recommendations.append({
                "stage": max_time_stage["name"],
                "issue": "Slow execution",
                "value_sec": max_time_stage["elapsed_time_sec"],
                "suggestion": "Consider parallelization or algorithm optimization",
            })

    # General recommendations
    if results.get("summary", {}).get("peak_memory_mb", 0) > 500:
        recommendations.append({
            "issue": "High peak memory",
            "value_mb": results["summary"]["peak_memory_mb"],
            "suggestion": "Clear unused variables, use gc.collect(), process in chunks",
        })

    return {
        "recommendations": recommendations,
        "n_recommendations": len(recommendations),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Memory usage profiling")
    parser.add_argument(
        "--features-dir", type=Path,
        default=Path("data/intermediate/features")
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/processed")
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("artifacts/models")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/evaluation")
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=== Memory Usage Profiling ===")

    all_results = {
        "data_loading": {},
        "feature_processing": {},
        "lsh_computation": {},
        "full_pipeline": {},
        "recommendations": {},
    }

    # 1. Profile data loading
    logger.info("\n=== Profiling Data Loading ===")
    if (args.features_dir / "train_features.csv").exists():
        all_results["data_loading"] = profile_data_loading(
            args.features_dir, args.data_dir
        )
    else:
        logger.warning("Features not found, skipping data loading profile")

    # 2. Profile feature processing
    logger.info("\n=== Profiling Feature Processing ===")
    if (args.features_dir / "train_features.csv").exists():
        all_results["feature_processing"] = profile_feature_processing(
            args.features_dir, cv=args.cv, seed=args.seed
        )
    else:
        logger.warning("Features not found, skipping feature processing profile")

    # 3. Profile LSH computation (simulated)
    logger.info("\n=== Profiling LSH Computation (Simulated) ===")
    all_results["lsh_computation"] = profile_lsh_computation(
        n_samples=10000,
        n_hashes=128,
        n_bands=16,
    )

    # 4. Profile full pipeline
    logger.info("\n=== Profiling Full Pipeline ===")
    if (args.features_dir / "train_features.csv").exists():
        all_results["full_pipeline"] = profile_full_pipeline(
            args.features_dir, args.data_dir, args.model_dir, cv=args.cv
        )
    else:
        logger.warning("Features not found, skipping full pipeline profile")

    # 5. Generate recommendations
    logger.info("\n=== Generating Recommendations ===")
    if all_results["full_pipeline"]:
        all_results["recommendations"] = get_optimization_recommendations(
            all_results["full_pipeline"]
        )

    # 6. Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(all_results, args.output_dir / "memory_profile.json")

    # 7. Generate reports
    plots_dir = args.output_dir / "plots" / "memory"

    if all_results["data_loading"]:
        generate_memory_report(all_results["data_loading"], plots_dir / "data_loading")

    if all_results["feature_processing"]:
        generate_memory_report(all_results["feature_processing"], plots_dir / "feature_processing")

    if all_results["lsh_computation"]:
        generate_memory_report(all_results["lsh_computation"], plots_dir / "lsh_computation")

    if all_results["full_pipeline"]:
        generate_memory_report(all_results["full_pipeline"], plots_dir / "full_pipeline")

    logger.info("Results saved to %s", args.output_dir)
    logger.info("Plots saved to %s", plots_dir)

    # Print summary
    print("\n=== Memory Profiling Summary ===")

    if all_results["full_pipeline"]:
        summary = all_results["full_pipeline"]["summary"]
        print(f"\nFull Pipeline:")
        print(f"  Total memory: {summary['total_memory_mb']:.2f} MB")
        print(f"  Peak memory: {summary['peak_memory_mb']:.2f} MB")
        print(f"  Total time: {summary['total_time_sec']:.2f} sec")

    if all_results["recommendations"]:
        print(f"\nRecommendations: {all_results['recommendations']['n_recommendations']}")
        for rec in all_results["recommendations"]["recommendations"]:
            print(f"  - {rec['issue']}: {rec['suggestion']}")

    print("Done.")


if __name__ == "__main__":
    main()
