.PHONY: setup build-zig data-prep preprocess features train pipeline ablation test clean help

PYTHON := .venv/bin/python
POETRY := poetry

help:
	@echo "JCCD-X-V6 Build System"
	@echo "====================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup          - Create venv, install Python deps, build Zig libs"
	@echo "  build-zig      - Build Zig shared libraries"
	@echo "  data-prep      - Run data preparation (split, clean, validate)"
	@echo "  preprocess     - Run source code normalization & tokenization"
	@echo "  features       - Run feature engineering (lexical, structural, quantitative)"
	@echo "  train          - Run ML model training with 5-fold CV"
	@echo "  pipeline       - Run full detection pipeline on test set"
	@echo "  ablation       - Run ablation studies (pipeline + feature + runtime)"
	@echo "  test           - Run all tests (Python + Zig)"
	@echo "  test-python    - Run Python tests only"
	@echo "  test-zig       - Run Zig tests only"
	@echo "  lint           - Run Python linting (ruff)"
	@echo "  clean          - Remove generated artifacts"
	@echo "  help           - Show this help message"

setup:
	@echo "=== Setting up Python environment ==="
	$(POETRY) install
	@echo ""
	@echo "=== Building Zig shared libraries ==="
	$(MAKE) build-zig
	@echo ""
	@echo "=== Setup complete ==="

build-zig:
	@echo "=== Building Zig libraries ==="
	zig build -Doptimize=ReleaseFast
	@echo "=== Zig build complete ==="

data-prep:
	@echo "=== Running data preparation ==="
	$(POETRY) run python -m src.python.preprocessing.data_prep
	@echo "=== Data preparation complete ==="

preprocess:
	@echo "=== Running preprocessing & normalization ==="
	$(POETRY) run python -m src.python.preprocessing.normalization
	@echo "=== Preprocessing complete ==="

features:
	@echo "=== Running feature engineering ==="
	$(POETRY) run python -m src.python.feature_engineering.features
	@echo "=== Feature engineering complete ==="

train:
	@echo "=== Running model training ==="
	$(POETRY) run python -m src.python.model.train
	@echo "=== Training complete ==="

pipeline:
	@echo "=== Running full pipeline ==="
	$(POETRY) run python -m src.python.pipeline.full_pipeline
	@echo "=== Pipeline complete ==="

ablation:
	@echo "=== Running ablation studies ==="
	$(POETRY) run python -m src.python.evaluation.ablation
	@echo "=== Ablation studies complete ==="

test: test-python test-zig

test-python:
	@echo "=== Running Python tests ==="
	$(POETRY) run pytest tests/ -v
	@echo "=== Python tests complete ==="

test-zig:
	@echo "=== Running Zig tests ==="
	zig build test
	@echo "=== Zig tests complete ==="

lint:
	@echo "=== Running linter ==="
	$(POETRY) run ruff check src/python/ tests/
	@echo "=== Linting complete ==="

clean:
	@echo "=== Cleaning generated artifacts ==="
	rm -rf data/processed/* data/intermediate/* artifacts/*
	rm -rf zig-cache zig-out
	rm -rf .pytest_cache __pycache__ .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "=== Clean complete ==="
