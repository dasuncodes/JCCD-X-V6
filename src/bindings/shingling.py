"""Python bindings for the Zig shingling library."""

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).parent.parent.parent / "zig-out" / "lib"


class ShinglingLib:
    def __init__(self):
        self._lib = ctypes.CDLL(str(_LIB_PATH / "libshingling.so"))
        self._setup_functions()

    def _setup_functions(self):
        self._lib.compute_shingles.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),   # tokens
            ctypes.c_size_t,                    # token_count
            ctypes.c_size_t,                    # k
            ctypes.POINTER(ctypes.c_uint64),   # out
        ]
        self._lib.compute_shingles.restype = ctypes.c_size_t

        self._lib.compute_shingles_batch.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),   # tokens
            ctypes.POINTER(ctypes.c_size_t),   # token_counts
            ctypes.c_size_t,                    # snippet_count
            ctypes.c_size_t,                    # k
            ctypes.POINTER(ctypes.c_uint64),   # out_hashes
            ctypes.c_size_t,                    # out_hashes_cap
            ctypes.POINTER(ctypes.c_size_t),   # out_offsets
        ]
        self._lib.compute_shingles_batch.restype = ctypes.c_size_t

    def compute_shingles(self, tokens: np.ndarray, k: int) -> np.ndarray:
        tokens = np.ascontiguousarray(tokens, dtype=np.uint32)
        max_shingles = max(len(tokens) - k + 1, 0)
        out = np.zeros(max_shingles, dtype=np.uint64)
        count = self._lib.compute_shingles(
            tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(tokens), k,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )
        return out[:count]

    def compute_shingles_batch(
        self, all_tokens: np.ndarray, token_counts: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute shingles for multiple snippets in batch."""
        all_tokens = np.ascontiguousarray(all_tokens, dtype=np.uint32)
        token_counts = np.ascontiguousarray(token_counts, dtype=np.uintp)

        # Estimate output capacity
        total_shingles = sum(max(0, int(tc) - k + 1) for tc in token_counts if int(tc) >= k)
        out_hashes = np.zeros(total_shingles, dtype=np.uint64)
        out_offsets = np.zeros(len(token_counts), dtype=np.uintp)

        written = self._lib.compute_shingles_batch(
            all_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            token_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            len(token_counts), k,
            out_hashes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            total_shingles,
            out_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        )
        return out_hashes[:written], out_offsets
