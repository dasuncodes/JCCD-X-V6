"""Python bindings for the Zig feature computation library."""

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).parent.parent.parent / "zig-out" / "lib"
FEATURE_COUNT = 25


class FeaturesLib:
    def __init__(self):
        self._lib = ctypes.CDLL(str(_LIB_PATH / "libfeatures.so"))
        self._setup_functions()

    def _setup_functions(self):
        self._lib.compute_features_batch.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),   # pair_idx_a
            ctypes.POINTER(ctypes.c_size_t),   # pair_idx_b
            ctypes.c_size_t,                    # pair_count
            ctypes.POINTER(ctypes.c_uint32),   # tokens
            ctypes.POINTER(ctypes.c_size_t),   # token_offsets
            ctypes.POINTER(ctypes.c_size_t),   # token_counts
            ctypes.POINTER(ctypes.c_uint32),   # histograms
            ctypes.POINTER(ctypes.c_uint16),   # depths
            ctypes.POINTER(ctypes.c_uint32),   # node_counts
            ctypes.POINTER(ctypes.c_uint32),   # cyclomatics
            ctypes.POINTER(ctypes.c_uint64),   # bigrams
            ctypes.POINTER(ctypes.c_size_t),   # bigram_offsets
            ctypes.POINTER(ctypes.c_size_t),   # bigram_counts
            ctypes.POINTER(ctypes.c_float),    # out_features
            ctypes.c_size_t,                    # num_threads
        ]

    def compute_features_batch(
        self,
        pair_idx_a: np.ndarray,
        pair_idx_b: np.ndarray,
        all_tokens: np.ndarray,
        token_offsets: np.ndarray,
        token_counts: np.ndarray,
        histograms: np.ndarray,
        depths: np.ndarray,
        node_counts: np.ndarray,
        cyclomatics: np.ndarray,
        all_bigrams: np.ndarray,
        bigram_offsets: np.ndarray,
        bigram_counts: np.ndarray,
        num_threads: int = 0,
    ) -> np.ndarray:
        pair_count = len(pair_idx_a)
        out = np.zeros(pair_count * FEATURE_COUNT, dtype=np.float32)

        pair_idx_a = np.ascontiguousarray(pair_idx_a, dtype=np.uintp)
        pair_idx_b = np.ascontiguousarray(pair_idx_b, dtype=np.uintp)
        all_tokens = np.ascontiguousarray(all_tokens, dtype=np.uint32)
        token_offsets = np.ascontiguousarray(token_offsets, dtype=np.uintp)
        token_counts = np.ascontiguousarray(token_counts, dtype=np.uintp)
        histograms = np.ascontiguousarray(histograms, dtype=np.uint32)
        depths = np.ascontiguousarray(depths, dtype=np.uint16)
        node_counts = np.ascontiguousarray(node_counts, dtype=np.uint32)
        cyclomatics = np.ascontiguousarray(cyclomatics, dtype=np.uint32)
        all_bigrams = np.ascontiguousarray(all_bigrams, dtype=np.uint64)
        bigram_offsets = np.ascontiguousarray(bigram_offsets, dtype=np.uintp)
        bigram_counts = np.ascontiguousarray(bigram_counts, dtype=np.uintp)
        out = np.ascontiguousarray(out, dtype=np.float32)

        self._lib.compute_features_batch(
            pair_idx_a.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            pair_idx_b.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            pair_count,
            all_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            token_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            token_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            histograms.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            depths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            node_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            cyclomatics.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            all_bigrams.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            bigram_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            bigram_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_threads,
        )

        return out.reshape(pair_count, FEATURE_COUNT)
