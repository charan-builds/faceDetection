"""
Ensure the repository root is on sys.path when tool modules are loaded.

Used by tools/*.py scripts so ``python tools/<script>.py`` works without
``pip install -e .``. Prefer editable install or ``python -m tools.<script>``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT_STR = str(_REPO_ROOT)

if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)
