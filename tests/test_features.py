"""Tests for feature engineering module."""

import pytest
from rapidfuzz.distance import Levenshtein

from src.bindings.features import FEATURE_COUNT, FeaturesLib


@pytest.fixture(scope="module")
def feat_lib():
    return FeaturesLib()


class TestFeaturesLib:
    def test_feature_count(self):
        assert FEATURE_COUNT == 25

    def test_batch_identical(self, feat_lib):
        import numpy as np
        # Two identical token sequences
        tokens = np.array([0, 1, 2, 3, 1, 2, 0], dtype=np.uint32)
        offsets = np.array([0, 0], dtype=np.uintp)
        counts = np.array([7, 7], dtype=np.uintp)
        histograms = np.array([1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        depths = np.array([5, 5], dtype=np.uint16)
        node_counts = np.array([10, 10], dtype=np.uint32)
        cyclomatics = np.array([3, 3], dtype=np.uint32)
        bigrams = np.array([100, 200], dtype=np.uint64)
        boffsets = np.array([0, 0], dtype=np.uintp)
        bcounts = np.array([2, 2], dtype=np.uintp)

        pair_a = np.array([0], dtype=np.uintp)
        pair_b = np.array([1], dtype=np.uintp)

        result = feat_lib.compute_features_batch(
            pair_a, pair_b,
            tokens, offsets, counts,
            histograms, depths, node_counts, cyclomatics,
            bigrams, boffsets, bcounts,
        )
        assert result.shape == (1, FEATURE_COUNT)
        # Identical pair: Jaccard should be ~1.0
        assert result[0, 1] > 0.99
        # Identical pair: Dice should be ~1.0
        assert result[0, 2] > 0.99
        # Identical pair: hist cosine should be ~1.0
        assert result[0, 5] > 0.99


class TestLevenshtein:
    def test_identical(self):
        r = Levenshtein.normalized_similarity("1 2 3", "1 2 3")
        assert r == 1.0

    def test_different(self):
        r = Levenshtein.normalized_similarity("1 2 3", "4 5 6")
        assert r < 0.5
