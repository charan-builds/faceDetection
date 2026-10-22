"""
Repository path helpers for tools/ CLI scripts.

These functions do not import the ``src`` package so tools can resolve paths
when executed as modules from the project root:

    python -m tools.collect_pad_dataset
"""

from __future__ import annotations

from pathlib import Path


def resolve_project_root() -> Path:
    """
    Return the Face Lock AI repository root directory.

    Assumes this file lives at ``<root>/tools/project_paths.py``.
    """
    return Path(__file__).resolve().parent.parent


def default_collection_root(collection_id: str = "face_lock_v1") -> Path:
    """Default PAD dataset directory under eval/datasets/."""
    return resolve_project_root() / "eval" / "datasets" / collection_id
