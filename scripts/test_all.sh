#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Zig Tests ==="
zig build test
echo ""

echo "=== Running Python Tests ==="
poetry run pytest tests/ -v
echo ""

echo "=== Running Lint ==="
poetry run ruff check src/python/ tests/
echo ""

echo "=== All Tests Complete ==="
