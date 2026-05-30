"""Tests for eval.harness.clip_loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2  # noqa: E402

from eval.harness.clip_loader import (
    ClipLoadError,
    discover_frame_paths,
    frames_from_arrays,
    load_clip_frames,
    load_presentation,
    load_still_frame,
    resolve_sample_path,
)


def _write_frame(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


@pytest.fixture
def collection_root(tmp_path: Path) -> Path:
    session = tmp_path / "bona_fide" / "subj_001" / "session_01"
    for index in range(1, 4):
        _write_frame(session / f"frame_{index:04d}.jpg", 200)
    still = tmp_path / "stills" / "still_01.jpg"
    _write_frame(still, 180)
    return tmp_path


def test_discover_frame_paths_sorted(collection_root: Path) -> None:
    session = collection_root / "bona_fide" / "subj_001" / "session_01"
    paths = discover_frame_paths(session)
    assert len(paths) == 3
    assert paths[0].name == "frame_0001.jpg"


def test_load_clip_frames(collection_root: Path) -> None:
    loaded = load_clip_frames(
        collection_root,
        "bona_fide/subj_001/session_01",
    )
    assert loaded.presentation_type == "clip"
    assert loaded.frame_count == 3
    assert len(loaded.source_paths) == 3


def test_load_still_frame(collection_root: Path) -> None:
    loaded = load_still_frame(collection_root, "stills/still_01.jpg")
    assert loaded.presentation_type == "still"
    assert loaded.frame_count == 1


def test_load_presentation_dispatches(collection_root: Path) -> None:
    clip = load_presentation(
        collection_root,
        "bona_fide/subj_001/session_01",
        "clip",
    )
    still = load_presentation(collection_root, "stills/still_01.jpg", "still")
    assert clip.frame_count == 3
    assert still.frame_count == 1


def test_resolve_sample_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ClipLoadError, match="escapes collection root"):
        resolve_sample_path(tmp_path, "../outside")


def test_missing_session_raises(tmp_path: Path) -> None:
    with pytest.raises(ClipLoadError, match="not found"):
        load_clip_frames(tmp_path, "missing/session")


def test_frames_from_arrays() -> None:
    arrays = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    loaded = frames_from_arrays(arrays)
    assert loaded.frame_count == 2
    assert loaded.source_paths == ()


def test_frames_from_arrays_empty_raises() -> None:
    with pytest.raises(ClipLoadError, match="empty"):
        frames_from_arrays([])


def test_max_frames_cap(collection_root: Path) -> None:
    loaded = load_clip_frames(
        collection_root,
        "bona_fide/subj_001/session_01",
        max_frames=2,
    )
    assert loaded.frame_count == 2
