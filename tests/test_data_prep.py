"""Tests for data preparation module."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.python.preprocessing.data_prep import (
    assign_labels,
    build_datasets,
    compute_split_metrics,
    load_all_datasets,
    validate_source_files,
)
from src.python.utils.io import load_csv_pairs, load_source_code, save_csv, save_json


@pytest.fixture
def raw_dir():
    return Path("data/raw/toma")


@pytest.fixture
def source_dir():
    return Path("data/raw/toma/id2sourcecode")


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path


class TestLoadCSV:
    def test_load_type1(self, raw_dir):
        df = load_csv_pairs(raw_dir / "type-1.csv", has_header=False)
        assert len(df) > 0
        assert "id1" in df.columns
        assert "id2" in df.columns
        assert df["id1"].dtype == object  # string

    def test_load_nonclone(self, raw_dir):
        df = load_csv_pairs(raw_dir / "nonclone.csv", has_header=True)
        assert len(df) > 0
        assert "id1" in df.columns
        assert "id2" in df.columns

    def test_load_all(self, raw_dir):
        datasets = load_all_datasets(raw_dir)
        assert "type1" in datasets
        assert "type2" in datasets
        assert "type3" in datasets
        assert "type4" in datasets
        assert "type5" in datasets
        assert "nonclone" in datasets


class TestLabelling:
    def test_type3_label(self, raw_dir):
        datasets = load_all_datasets(raw_dir)
        labelled = assign_labels(datasets)
        assert (labelled["type3"]["label"] == 1).all()
        assert (labelled["type4"]["label"] == 1).all()
        assert (labelled["type5"]["label"] == 0).all()
        assert (labelled["nonclone"]["label"] == 0).all()
        assert (labelled["type1"]["label"] == -1).all()
        assert (labelled["type2"]["label"] == -1).all()


class TestValidation:
    def test_validation_removes_missing(self, raw_dir, source_dir):
        datasets = load_all_datasets(raw_dir)
        labelled = assign_labels(datasets)
        validated, stats = validate_source_files(labelled, source_dir)
        for key in validated:
            assert stats[key]["valid"] <= stats[key]["original"]
            assert stats[key]["valid"] == len(validated[key])
            assert stats[key]["removed"] == stats[key]["original"] - stats[key]["valid"]

    def test_all_valid_pairs_have_source(self, raw_dir, source_dir):
        datasets = load_all_datasets(raw_dir)
        labelled = assign_labels(datasets)
        validated, _ = validate_source_files(labelled, source_dir)
        for key, df in validated.items():
            for _, row in df.head(10).iterrows():
                assert (source_dir / f"{row['id1']}.java").exists()
                assert (source_dir / f"{row['id2']}.java").exists()


class TestSourceCode:
    def test_load_existing(self, source_dir):
        # Pick a known ID from type-1
        df = load_csv_pairs("data/raw/toma/type-1.csv")
        fid = df.iloc[0]["id1"]
        src = load_source_code(fid, source_dir)
        assert src is not None
        assert len(src) > 0

    def test_load_missing(self, source_dir):
        src = load_source_code("NONEXISTENT_99999", source_dir)
        assert src is None


class TestSplit:
    def test_split_ratios(self, raw_dir, source_dir):
        datasets = load_all_datasets(raw_dir)
        labelled = assign_labels(datasets)
        validated, _ = validate_source_files(labelled, source_dir)
        train, test = build_datasets(validated, test_ratio=0.3, seed=42)
        # Training set should be balanced (equal pos/neg)
        pos = (train["label"] == 1).sum()
        neg = (train["label"] == 0).sum()
        assert pos == neg
        # Training set should be non-empty and reasonable size
        assert len(train) > 0
        assert len(test) > 0

    def test_train_has_no_type1_type2(self, raw_dir, source_dir):
        datasets = load_all_datasets(raw_dir)
        labelled = assign_labels(datasets)
        validated, _ = validate_source_files(labelled, source_dir)
        train, _ = build_datasets(validated, test_ratio=0.3, seed=42)
        assert set(train["label"].unique()).issubset({0, 1})

    def test_test_has_type1_type2(self, raw_dir, source_dir):
        datasets = load_all_datasets(raw_dir)
        labelled = assign_labels(datasets)
        validated, _ = validate_source_files(labelled, source_dir)
        _, test = build_datasets(validated, test_ratio=0.3, seed=42)
        assert -1 in test["label"].values


class TestMetrics:
    def test_split_metrics(self, raw_dir, source_dir):
        datasets = load_all_datasets(raw_dir)
        labelled = assign_labels(datasets)
        validated, _ = validate_source_files(labelled, source_dir)
        train, test = build_datasets(validated, test_ratio=0.3, seed=42)
        metrics = compute_split_metrics(validated, train, test)
        assert "training_size" in metrics
        assert "testing_size" in metrics
        assert "train_positive" in metrics
        assert "train_negative" in metrics
        assert metrics["training_size"] == len(train)
        assert metrics["testing_size"] == len(test)


class TestIO:
    def test_save_load_csv(self, tmp_output):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_output / "test.csv"
        save_csv(df, path)
        loaded = pd.read_csv(path)
        assert len(loaded) == 2

    def test_save_json(self, tmp_output):
        data = {"key": "value", "num": 42}
        path = tmp_output / "test.json"
        save_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data
