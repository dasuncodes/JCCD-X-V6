"""Python bindings for the Zig MinHash/LSH library."""

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).parent.parent.parent / "zig-out" / "lib"


class MinHashLib:
    def __init__(self):
        self._lib = ctypes.CDLL(str(_LIB_PATH / "libminhash.so"))
        self._setup_functions()

    def _setup_functions(self):
        self._lib.minhash_signature.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),   # shingles
            ctypes.c_size_t,                    # shingle_count
            ctypes.c_size_t,                    # num_hashes
            ctypes.c_uint64,                    # seed
            ctypes.POINTER(ctypes.c_uint64),   # out_signature
        ]

        self._lib.minhash_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),   # sig_a
            ctypes.POINTER(ctypes.c_uint64),   # sig_b
            ctypes.c_size_t,                    # num_hashes
        ]
        self._lib.minhash_similarity.restype = ctypes.c_uint32

        self._lib.lsh_buckets.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),   # signature
            ctypes.c_size_t,                    # sig_len
            ctypes.c_size_t,                    # bands
            ctypes.c_size_t,                    # rows_per_band
            ctypes.POINTER(ctypes.c_uint64),   # out_buckets
        ]

    def minhash_signature(
        self, shingles: np.ndarray, num_hashes: int, seed: int = 42
    ) -> np.ndarray:
        shingles = np.ascontiguousarray(shingles, dtype=np.uint64)
        sig = np.zeros(num_hashes, dtype=np.uint64)
        self._lib.minhash_signature(
            shingles.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(shingles), num_hashes, seed,
            sig.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )
        return sig

    def minhash_similarity(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        sig_a = np.ascontiguousarray(sig_a, dtype=np.uint64)
        sig_b = np.ascontiguousarray(sig_b, dtype=np.uint64)
        return self._lib.minhash_similarity(
            sig_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            sig_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(sig_a),
        ) / 10000.0

    def lsh_buckets(
        self, signature: np.ndarray, bands: int, rows_per_band: int
    ) -> np.ndarray:
        signature = np.ascontiguousarray(signature, dtype=np.uint64)
        buckets = np.zeros(bands, dtype=np.uint64)
        self._lib.lsh_buckets(
            signature.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(signature), bands, rows_per_band,
            buckets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )
        return buckets
