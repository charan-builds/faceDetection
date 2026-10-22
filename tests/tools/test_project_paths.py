"""Tests for tools.project_paths."""

from __future__ import annotations

from pathlib import Path

from tools.project_paths import default_collection_root, resolve_project_root


def test_resolve_project_root_points_at_repo() -> None:
    root = resolve_project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "config.py").is_file()
    assert (root / "tools" / "collect_pad_dataset.py").is_file()


def test_default_collection_root() -> None:
    root = resolve_project_root()
    assert default_collection_root() == root / "eval" / "datasets" / "face_lock_v1"
