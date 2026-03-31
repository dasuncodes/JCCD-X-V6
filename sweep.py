#!/usr/bin/env python3
"""Quick LSH parameter sweep using subprocess."""
import subprocess
import re
import json
import sys


def run_pipeline(hashes, bands, threshold=0.35, sim_threshold=0.0):
    cmd = [
        "poetry",
        "run",
        "python",
        "-m",
        "src.python.pipeline.full_pipeline",
        "--lsh-hashes",
        str(hashes),
        "--lsh-bands",
        str(bands),
        "--threshold",
        str(threshold),
        "--sim-threshold",
        str(sim_threshold),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    # parse last summary lines
    lines = output.split("\n")
    summary_started = False
    metrics = {}
    for line in lines:
        if line.strip() == "=== Pipeline Summary ===":
            summary_started = True
        if summary_started and line.strip().startswith("LSH reduction:"):
            # LSH reduction: 51.8%
            match = re.search(r"LSH reduction:\s*([\d.]+)%", line)
            if match:
                metrics["lsh_reduction"] = float(match.group(1))
        if summary_started and line.strip().startswith("Recall:"):
            match = re.search(r"Recall:\s*([\d.]+)", line)
            if match:
                metrics["recall"] = float(match.group(1))
        if summary_started and line.strip().startswith("Precision:"):
            match = re.search(r"Precision:\s*([\d.]+)", line)
            if match:
                metrics["precision"] = float(match.group(1))
        if summary_started and line.strip().startswith("Total time:"):
            match = re.search(r"Total time:\s*([\d.]+)s", line)
            if match:
                metrics["time"] = float(match.group(1))
    return metrics, output


def main():
    hashes_list = [32, 36, 40, 44, 48]
    bands_list = [8, 9, 10, 11, 12, 13]
    results = []
    for h in hashes_list:
        for b in bands_list:
            if h % b != 0:
                continue  # skip non-divisible for now
            metrics, output = run_pipeline(h, b)
            if not metrics:
                print(f"Failed to parse output for hashes={h}, bands={b}")
                continue
            metrics["hashes"] = h
            metrics["bands"] = b
            results.append(metrics)
            print(
                f"  hashes={h}, bands={b}: reduction={metrics.get('lsh_reduction',0):.1f}%, recall={metrics.get('recall',0):.4f}, precision={metrics.get('precision',0):.4f}"
            )
    # save results
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to sweep_results.json")
    # filter for recall >=0.90 and reduction >=50
    good = [r for r in results if r.get("recall", 0) >= 0.90 and r.get("lsh_reduction", 0) >= 50.0]
    print(f"Found {len(good)} configurations meeting targets")
    for g in good:
        print(
            f"  hashes={g['hashes']}, bands={g['bands']}, reduction={g['lsh_reduction']:.1f}%, recall={g['recall']:.4f}, precision={g['precision']:.4f}"
        )


if __name__ == "__main__":
    main()
