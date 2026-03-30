"""Python bindings for the Zig AST statistics library."""

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).parent.parent.parent / "zig-out" / "lib"


class AstStatsLib:
    def __init__(self):
        self._lib = ctypes.CDLL(str(_LIB_PATH / "libast_stats.so"))
        self._setup_functions()

    def _setup_functions(self):
        self._lib.compute_ast_stats.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),    # nodes
            ctypes.POINTER(ctypes.c_uint16),   # depths
            ctypes.c_size_t,                    # node_count
            ctypes.POINTER(ctypes.c_uint16),   # out_max_depth
            ctypes.POINTER(ctypes.c_uint32),   # out_histogram
            ctypes.c_size_t,                    # histogram_size
        ]

        self._lib.compute_bigrams.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),    # nodes
            ctypes.POINTER(ctypes.c_size_t),   # parent_indices
            ctypes.c_size_t,                    # node_count
            ctypes.POINTER(ctypes.c_uint64),   # out_bigrams
        ]
        self._lib.compute_bigrams.restype = ctypes.c_size_t

        self._lib.compute_cyclomatic_complexity.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),    # node_types
            ctypes.c_size_t,                    # node_count
        ]
        self._lib.compute_cyclomatic_complexity.restype = ctypes.c_uint32

        self._lib.histogram_cosine_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),   # hist_a
            ctypes.POINTER(ctypes.c_uint32),   # hist_b
            ctypes.c_size_t,                    # size
        ]
        self._lib.histogram_cosine_similarity.restype = ctypes.c_uint32

    def compute_ast_stats(
        self, nodes: np.ndarray, depths: np.ndarray,
        out_max_depth: np.ndarray, out_histogram: np.ndarray,
    ) -> None:
        nodes = np.ascontiguousarray(nodes, dtype=np.uint8)
        depths = np.ascontiguousarray(depths, dtype=np.uint16)
        out_max_depth = np.ascontiguousarray(out_max_depth, dtype=np.uint16)
        out_histogram = np.ascontiguousarray(out_histogram, dtype=np.uint32)
        self._lib.compute_ast_stats(
            nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            depths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            len(nodes),
            out_max_depth.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            out_histogram.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(out_histogram),
        )

    def compute_bigrams(
        self, nodes: np.ndarray, parents: np.ndarray, out_bigrams: np.ndarray
    ) -> int:
        nodes = np.ascontiguousarray(nodes, dtype=np.uint8)
        parents = np.ascontiguousarray(parents, dtype=np.uintp)
        out_bigrams = np.ascontiguousarray(out_bigrams, dtype=np.uint64)
        return self._lib.compute_bigrams(
            nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            parents.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            len(nodes),
            out_bigrams.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )

    def compute_cyclomatic_complexity(self, node_types: np.ndarray) -> int:
        node_types = np.ascontiguousarray(node_types, dtype=np.uint8)
        return self._lib.compute_cyclomatic_complexity(
            node_types.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            len(node_types),
        )

    def histogram_cosine_similarity(
        self, hist_a: np.ndarray, hist_b: np.ndarray
    ) -> float:
        hist_a = np.ascontiguousarray(hist_a, dtype=np.uint32)
        hist_b = np.ascontiguousarray(hist_b, dtype=np.uint32)
        assert len(hist_a) == len(hist_b)
        result = self._lib.histogram_cosine_similarity(
            hist_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            hist_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(hist_a),
        )
        return result / 10000.0
