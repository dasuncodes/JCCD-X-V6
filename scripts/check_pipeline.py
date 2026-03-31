#!/usr/bin/env python3
"""Diagnostic script to check pipeline prerequisites and fix feature mismatch."""

import json
import sys
from pathlib import Path

def check_pipeline_prerequisites():
    """Check if all required files exist for the pipeline to work."""

    print("=" * 70)
    print("Pipeline Prerequisites Check")
    print("=" * 70)

    issues = []
    warnings = []

    # Check model files
    print("\n1. Checking model files...")
    model_dir = Path("artifacts/models")

    model_files = {
        "best_model.joblib": "Trained ML model",
        "best_model_name.joblib": "Model name identifier",
    }

    for fname, desc in model_files.items():
        fpath = model_dir / fname
        if fpath.exists():
            print(f"   ✓ {fname}: {desc}")
        else:
            print(f"   ✗ {fname}: MISSING - {desc}")
            issues.append(f"Missing model file: {fname}")

    # Check selected features
    selected_features_path = model_dir / "selected_features.json"
    if selected_features_path.exists():
        with open(selected_features_path, 'r') as f:
            selected_features = json.load(f)
        print(f"   ✓ selected_features.json: {len(selected_features)} features")
    else:
        print(f"   ⚠ selected_features.json: MISSING - Will use all 26 features")
        warnings.append("No selected_features.json - pipeline will use all features")

    # Check RFE model
    rfe_model_path = model_dir / "best_model_rfe.joblib"
    if rfe_model_path.exists():
        print(f"   ✓ best_model_rfe.joblib: RFE-optimized model exists")
    else:
        print(f"   ℹ best_model_rfe.jobfile: Not found (optional)")

    # Check data files
    print("\n2. Checking data files...")
    data_dirs = {
        "data/processed": "Processed datasets",
        "data/intermediate": "Intermediate features and tokens",
    }

    for dpath, desc in data_dirs.items():
        if Path(dpath).exists():
            print(f"   ✓ {dpath}/: {desc}")
        else:
            print(f"   ✗ {dpath}/: MISSING - {desc}")
            issues.append(f"Missing data directory: {dpath}")

    # Check specific required files
    required_files = {
        "data/processed/testing_dataset.csv": "Test dataset with labels",
        "data/intermediate/token_data.pkl": "Token data for LSH",
        "data/intermediate/features/test_features.csv": "Test features",
    }

    for fpath, desc in required_files.items():
        if Path(fpath).exists():
            print(f"   ✓ {fpath}: {desc}")
        else:
            print(f"   ✗ {fpath}: MISSING - {desc}")
            issues.append(f"Missing required file: {fpath}")

    # Check normalized source files
    print("\n3. Checking normalized source files...")
    norm_dir = Path("data/processed/normalized")
    if norm_dir.exists():
        java_files = list(norm_dir.glob("*.java"))
        print(f"   ✓ {norm_dir}/: {len(java_files)} normalized Java files")
    else:
        print(f"   ✗ {norm_dir}/: MISSING - Type-1/2 detection will fail")
        issues.append("Missing normalized source files")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if issues:
        print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\n   The pipeline CANNOT run without fixing these issues.")
        print("   You need to run the full data preparation pipeline:")
        print("   bash scripts/run_all.sh")
        return False
    elif warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
        print("\n   The pipeline can run but may have suboptimal performance.")
        return True
    else:
        print("\n✓ All prerequisites are met!")
        print("\n   The pipeline should run successfully.")
        return True


def check_feature_mismatch():
    """Check if there's a feature mismatch between training and inference."""

    print("\n" + "=" * 70)
    print("Feature Mismatch Analysis")
    print("=" * 70)

    model_dir = Path("artifacts/models")
    intermediate_dir = Path("data/intermediate/features")

    # Check if model was trained with RFE
    selected_features_path = model_dir / "selected_features.json"
    test_features_path = intermediate_dir / "test_features.csv"

    if not selected_features_path.exists():
        print("\nℹ️  Model was trained with ALL 26 features (no RFE selection)")
        print("   This is fine if test features also have all 26 features.")
        return

    if not test_features_path.exists():
        print("\n⚠️  Cannot check feature mismatch - test_features.csv missing")
        return

    with open(selected_features_path, 'r') as f:
        selected_features = set(json.load(f))

    import pandas as pd
    feat_df = pd.read_csv(test_features_path)
    test_features = set(feat_df.columns) - {"id1", "id2", "label"}

    print(f"\nModel trained with: {len(selected_features)} features")
    print(f"Test features available: {len(test_features)} features")

    missing = selected_features - test_features
    extra = test_features - selected_features

    if missing:
        print(f"\n❌ CRITICAL: {len(missing)} features in model but NOT in test data:")
        for f in sorted(missing):
            print(f"   - {f}")
        print("\n   This will cause the ML classifier to fail!")
        print("   Solution: Re-run feature engineering to generate all required features.")
    else:
        print(f"\n✓ Feature sets match!")
        if extra:
            print(f"   Note: {len(extra)} extra features in test data (will be ignored)")


def suggest_fixes():
    """Suggest fixes based on detected issues."""

    print("\n" + "=" * 70)
    print("Suggested Fixes")
    print("=" * 70)

    print("""
If you see MISSING files, you need to run the full pipeline:

    bash scripts/run_all.sh

This will:
1. Prepare the dataset (split train/test)
2. Normalize source code
3. Generate tokens and AST data
4. Compute features for all pairs
5. Train the ML model
6. Run the full pipeline evaluation


If you see FEATURE MISMATCH:

The model was trained with different features than what's available.
Solutions:

1. Re-train with current features:
   bash scripts/run_training.sh

2. Or re-generate features to match model:
   bash scripts/run_features.sh


If ML classifier predicts ALL as non-clone (Precision/Recall = 0):

1. Check probability distribution:
   - Look for "Probability stats" in pipeline output
   - If max probability < threshold (0.30), lower the threshold:
     bash scripts/run_pipeline.sh --threshold 0.1

2. Feature mismatch (most common):
   - Model expects features that aren't in test data
   - Re-run feature engineering or retrain model

3. Model needs retraining:
   - If features changed, retrain with:
     bash scripts/run_training.sh
""")


if __name__ == "__main__":
    ok = check_pipeline_prerequisites()
    check_feature_mismatch()
    suggest_fixes()

    sys.exit(0 if ok else 1)
