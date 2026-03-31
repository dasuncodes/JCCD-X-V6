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
            ctypes.POINTER(ctypes.c_uint64),  # shingles
            ctypes.c_size_t,  # shingle_count
            ctypes.c_size_t,  # num_hashes
            ctypes.c_uint64,  # seed
            ctypes.POINTER(ctypes.c_uint64),  # out_signature
        ]

        self._lib.minhash_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),  # sig_a
            ctypes.POINTER(ctypes.c_uint64),  # sig_b
            ctypes.c_size_t,  # num_hashes
        ]
        self._lib.minhash_similarity.restype = ctypes.c_uint32

        self._lib.lsh_buckets.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),  # signature
            ctypes.c_size_t,  # sig_len
            ctypes.c_size_t,  # bands
            ctypes.c_size_t,  # rows_per_band
            ctypes.POINTER(ctypes.c_uint64),  # out_buckets
        ]
        self._lib.minhash_signature_batch.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),  # shingle_offsets
            ctypes.POINTER(ctypes.c_size_t),  # shingle_counts
            ctypes.POINTER(ctypes.c_uint64),  # shingles
            ctypes.c_size_t,  # file_count
            ctypes.c_size_t,  # num_hashes
            ctypes.c_uint64,  # seed
            ctypes.POINTER(ctypes.c_uint64),  # out_signatures
        ]
        self._lib.lsh_buckets_batch.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),  # signatures
            ctypes.c_size_t,  # file_count
            ctypes.c_size_t,  # sig_len
            ctypes.c_size_t,  # bands
            ctypes.c_size_t,  # rows_per_band
            ctypes.POINTER(ctypes.c_uint64),  # out_buckets
        ]

    def minhash_signature(
        self, shingles: np.ndarray, num_hashes: int, seed: int = 42
    ) -> np.ndarray:
        shingles = np.ascontiguousarray(shingles, dtype=np.uint64)
        sig = np.zeros(num_hashes, dtype=np.uint64)
        self._lib.minhash_signature(
            shingles.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(shingles),
            num_hashes,
            seed,
            sig.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )
        return sig

    def minhash_similarity(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        sig_a = np.ascontiguousarray(sig_a, dtype=np.uint64)
        sig_b = np.ascontiguousarray(sig_b, dtype=np.uint64)
        return (
            self._lib.minhash_similarity(
                sig_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                sig_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                len(sig_a),
            )
            / 10000.0
        )

    def lsh_buckets(self, signature: np.ndarray, bands: int, rows_per_band: int) -> np.ndarray:
        signature = np.ascontiguousarray(signature, dtype=np.uint64)
        buckets = np.zeros(bands, dtype=np.uint64)
        self._lib.lsh_buckets(
            signature.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(signature),
            bands,
            rows_per_band,
            buckets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )
        return buckets

    def minhash_signature_batch(
        self,
        shingles_list: list[np.ndarray],
        num_hashes: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Compute MinHash signatures for a list of shingle arrays.

        Args:
            shingles_list: List of 1-D uint64 arrays, each representing shingles for a file.
            num_hashes: Number of hash functions.
            seed: Random seed.

        Returns:
            2-D array of shape (file_count, num_hashes) with signatures.
        """
        file_count = len(shingles_list)
        # Concatenate all shingles into one flat array
        total_len = sum(len(s) for s in shingles_list)
        all_shingles = np.zeros(total_len, dtype=np.uint64)
        offsets = np.zeros(file_count, dtype=np.uint64)
        counts = np.zeros(file_count, dtype=np.uint64)
        pos = 0
        for i, s in enumerate(shingles_list):
            s = np.ascontiguousarray(s, dtype=np.uint64)
            length = len(s)
            all_shingles[pos : pos + length] = s
            offsets[i] = pos
            counts[i] = length
            pos += length
        out = np.zeros(file_count * num_hashes, dtype=np.uint64)
        self._lib.minhash_signature_batch(
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            all_shingles.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            file_count,
            num_hashes,
            seed,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )
        return out.reshape(file_count, num_hashes)

    def minhash_signature_batch_flat(
        self,
        flat_shingles: np.ndarray,
        offsets: np.ndarray,
        counts: np.ndarray,
        num_hashes: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Compute MinHash signatures using flat shingle array and offsets.

        Args:
            flat_shingles: 1-D uint64 array containing all shingles concatenated.
            offsets: 1-D uint64 array of start indices for each file.
            counts: 1-D uint64 array of shingle counts for each file.
            num_hashes: Number of hash functions.
            seed: Random seed.

        Returns:
            2-D array of shape (file_count, num_hashes) with signatures.
        """
        flat_shingles = np.ascontiguousarray(flat_shingles, dtype=np.uint64)
        offsets = np.ascontiguousarray(offsets, dtype=np.uintp)
        counts = np.ascontiguousarray(counts, dtype=np.uintp)
        file_count = len(offsets)
        out = np.zeros(file_count * num_hashes, dtype=np.uint64)
        self._lib.minhash_signature_batch(
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            flat_shingles.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            file_count,
            num_hashes,
            seed,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )
        return out.reshape(file_count, num_hashes)

    def lsh_buckets_batch(
        self,
        signatures: np.ndarray,
        bands: int,
        rows_per_band: int,
    ) -> np.ndarray:
        """Compute LSH bucket IDs for multiple signatures.

        Args:
            signatures: 2-D array of shape (file_count, sig_len) with MinHash signatures.
            bands: Number of bands.
            rows_per_band: Rows per band.

        Returns:
            2-D array of shape (file_count, bands) with bucket IDs.
        """
        signatures = np.ascontiguousarray(signatures, dtype=np.uint64)
        file_count, sig_len = signatures.shape
        out = np.zeros(file_count * bands, dtype=np.uint64)
        self._lib.lsh_buckets_batch(
            signatures.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            file_count,
            sig_len,
            bands,
            rows_per_band,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )
        return out.reshape(file_count, bands)
