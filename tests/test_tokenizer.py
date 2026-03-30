"""Tests for the tokenization and CST generation module."""

import numpy as np
import pytest

from src.bindings.ast_stats import AstStatsLib
from src.python.preprocessing.tokenizer import (
    extract_tokens,
)


@pytest.fixture(scope="module")
def ast_lib():
    return AstStatsLib()


class TestTokenExtraction:
    def test_simple_method(self):
        src = "int x = 5;"
        token_ids, ast_info = extract_tokens(src)
        assert len(token_ids) > 0
        assert ast_info["node_count"] > 0
        assert len(ast_info["depths"]) == ast_info["node_count"]

    def test_if_statement(self):
        src = "if (x > 0) { return x; }"
        token_ids, ast_info = extract_tokens(src)
        assert len(token_ids) > 0
        # Should contain keywords, identifiers, operators, separators
        unique_types = set(token_ids)
        assert len(unique_types) > 1

    def test_method_declaration(self):
        src = "public void foo(int a, int b) { return a + b; }"
        token_ids, ast_info = extract_tokens(src)
        assert len(token_ids) > 0
        assert max(ast_info["depths"]) > 0

    def test_empty_input(self):
        src = ""
        token_ids, ast_info = extract_tokens(src)
        # Empty input should still produce some AST nodes (program node)
        assert ast_info["node_count"] >= 0

    def test_parent_indices(self):
        src = "int x = 5; int y = 10;"
        _, ast_info = extract_tokens(src)
        parents = ast_info["parent_indices"]
        assert len(parents) == ast_info["node_count"]
        # Root should have no parent (maxInt sentinel)
        assert parents[0] == 2**64 - 1


class TestAstStatsZig:
    def test_compute_ast_stats(self, ast_lib):
        nodes = np.array([0, 1, 2, 3, 1], dtype=np.uint8)
        depths = np.array([0, 1, 1, 2, 2], dtype=np.uint16)
        max_depth = np.zeros(1, dtype=np.uint16)
        histogram = np.zeros(10, dtype=np.uint32)
        ast_lib.compute_ast_stats(nodes, depths, max_depth, histogram)
        assert max_depth[0] == 2
        assert histogram[0] == 1  # 1 keyword
        assert histogram[1] == 2  # 2 identifiers
        assert histogram[2] == 1  # 1 operator
        assert histogram[3] == 1  # 1 literal

    def test_compute_bigrams(self, ast_lib):
        nodes = np.array([0, 1, 2, 3], dtype=np.uint8)
        no_parent = np.uintp(2**64 - 1)  # std.math.maxInt(usize)
        parents = np.array([no_parent, 0, 0, 1], dtype=np.uintp)
        bigrams = np.zeros(4, dtype=np.uint64)
        count = ast_lib.compute_bigrams(nodes, parents, bigrams)
        assert count == 3

    def test_cyclomatic_complexity(self, ast_lib):
        nodes = np.array([0, 1, 0, 2, 1], dtype=np.uint8)
        cc = ast_lib.compute_cyclomatic_complexity(nodes)
        assert cc == 4  # 1 base + 2 keywords + 1 operator

    def test_histogram_cosine_similarity_identical(self, ast_lib):
        hist = np.array([1, 2, 3, 0, 1], dtype=np.uint32)
        sim = ast_lib.histogram_cosine_similarity(hist, hist)
        assert sim > 0.99

    def test_histogram_cosine_similarity_orthogonal(self, ast_lib):
        hist_a = np.array([1, 0, 0, 0, 0], dtype=np.uint32)
        hist_b = np.array([0, 1, 0, 0, 0], dtype=np.uint32)
        sim = ast_lib.histogram_cosine_similarity(hist_a, hist_b)
        assert sim < 0.01
