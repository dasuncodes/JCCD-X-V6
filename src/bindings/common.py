"""Shared bindings utilities for loading Zig libraries."""

from pathlib import Path

_LIB_PATH = Path(__file__).parent.parent.parent / "zig-out" / "lib"


def get_lib_path(name: str) -> Path:
    """Get the path to a Zig shared library by name."""
    path = _LIB_PATH / f"lib{name}.so"
    if not path.exists():
        raise FileNotFoundError(f"Zig library not found: {path}. Run 'zig build -Doptimize=ReleaseFast'")
    return path
