#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  JCCD-X-V6 Test Suite"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Run Python tests
echo "=== Running Python Tests ==="
poetry run pytest tests/ -v --tb=short
echo ""

# Run Zig tests
echo "=== Running Zig Tests ==="
zig build test
echo ""

# Run linter
echo "=== Running Python Linter ==="
poetry run ruff check src/python/ tests/
echo ""

echo "=========================================="
echo "  All Tests Complete!"
echo "=========================================="
echo ""
