"""Python bindings for the Zig normalization library."""

import ctypes
from pathlib import Path

_LIB_PATH = Path(__file__).parent.parent.parent / "zig-out" / "lib"


class NormalizationLib:
    def __init__(self):
        self._lib = ctypes.CDLL(str(_LIB_PATH / "libnormalization.so"))
        self._setup_functions()

    def _setup_functions(self):
        self._lib.remove_comments.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
        ]
        self._lib.remove_comments.restype = ctypes.c_size_t

        self._lib.normalize_whitespace.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
        ]
        self._lib.normalize_whitespace.restype = ctypes.c_size_t

        self._lib.normalize_source.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
        ]
        self._lib.normalize_source.restype = ctypes.c_size_t

        self._lib.compute_norm_impact.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
        ]

    def remove_comments(self, source: str) -> str:
        src_bytes = source.encode("utf-8")
        buf_size = max(len(src_bytes) * 2, 4096)
        out_buf = (ctypes.c_ubyte * buf_size)()
        result_len = self._lib.remove_comments(
            (ctypes.c_ubyte * len(src_bytes)).from_buffer_copy(src_bytes),
            len(src_bytes),
            out_buf,
            buf_size,
        )
        return bytes(out_buf[:result_len]).decode("utf-8", errors="replace")

    def normalize_whitespace(self, source: str) -> str:
        src_bytes = source.encode("utf-8")
        buf_size = max(len(src_bytes) * 2, 4096)
        out_buf = (ctypes.c_ubyte * buf_size)()
        result_len = self._lib.normalize_whitespace(
            (ctypes.c_ubyte * len(src_bytes)).from_buffer_copy(src_bytes),
            len(src_bytes),
            out_buf,
            buf_size,
        )
        return bytes(out_buf[:result_len]).decode("utf-8", errors="replace")

    def normalize_source(self, source: str) -> str:
        src_bytes = source.encode("utf-8")
        buf_size = max(len(src_bytes) * 2, 4096)
        out_buf = (ctypes.c_ubyte * buf_size)()
        result_len = self._lib.normalize_source(
            (ctypes.c_ubyte * len(src_bytes)).from_buffer_copy(src_bytes),
            len(src_bytes),
            out_buf,
            buf_size,
        )
        return bytes(out_buf[:result_len]).decode("utf-8", errors="replace")

    def compute_norm_impact(self, original: str, normalized: str) -> dict:
        orig_bytes = original.encode("utf-8")
        norm_bytes = normalized.encode("utf-8")
        metrics = (ctypes.c_uint32 * 4)()
        self._lib.compute_norm_impact(
            (ctypes.c_ubyte * len(orig_bytes)).from_buffer_copy(orig_bytes),
            len(orig_bytes),
            (ctypes.c_ubyte * len(norm_bytes)).from_buffer_copy(norm_bytes),
            len(norm_bytes),
            metrics,
        )
        return {
            "original_lines": metrics[0],
            "normalized_lines": metrics[1],
            "blank_lines_removed": metrics[2],
            "comments_removed_lines": metrics[3],
        }
