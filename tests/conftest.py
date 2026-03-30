"""Shared test configuration."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.python.*` can be imported
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
