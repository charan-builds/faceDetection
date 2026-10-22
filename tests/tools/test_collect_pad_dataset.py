"""Tests for tools.collect_pad_dataset (non-camera logic)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from tools.collect_pad_dataset import (
    CollectionConfig,
    CollectionValidationError,
    DEFAULT_CAPTURE_FPS,
    build_presentation_path,
    effective_capture_fps,
    frame_filename,
    next_sequence_dir,
    sample_interval_seconds,
    subject_id_to_folder_name,
    validate_frame_count,
    validate_presentation_type,
    validate_subject_id,
    write_meta_json,
)


@pytest.fixture
def collection_root(tmp_path: Path) -> Path:
    root = tmp_path / "face_lock_v1"
    root.mkdir()
    return root


@pytest.fixture
def config(collection_root: Path) -> CollectionConfig:
    return CollectionConfig(
        collection_root=collection_root,
        camera_index=0,
        min_frames=12,
        max_frames=16,
        capture_fps=DEFAULT_CAPTURE_FPS,
        collector_id="test_collector",
        write_meta=True,
        jpeg_quality=90,
    )


class TestValidation:
    def test_validate_subject_id_ok(self) -> None:
        assert validate_subject_id("subj_001") == "subj_001"
        assert validate_subject_id(" SUBJ_042 ") == "subj_042"

    def test_validate_subject_id_bad(self) -> None:
        with pytest.raises(CollectionValidationError, match="subj_NNN"):
            validate_subject_id("subject_001")
        with pytest.raises(CollectionValidationError):
            validate_subject_id("subj_1")

    def test_validate_presentation_type_ok(self) -> None:
        assert validate_presentation_type("pa-01") == "PA-01"
        assert validate_presentation_type("bf") == "BF"

    def test_validate_presentation_type_bad(self) -> None:
        with pytest.raises(CollectionValidationError):
            validate_presentation_type("PA-99")

    def test_subject_id_to_folder_name(self) -> None:
        assert subject_id_to_folder_name("subj_007") == "subject_007"


class TestPaths:
    def test_build_bona_fide_path(self, collection_root: Path) -> None:
        info = build_presentation_path(collection_root, "subj_001", "BF")
        assert info.presentation_dir.name == "session_01"
        assert info.relative_path == "bona_fide/subject_001/session_01"

    def test_build_attack_path(self, collection_root: Path) -> None:
        info = build_presentation_path(collection_root, "subj_002", "PA-03")
        assert "PA-03_screen_video" in info.relative_path
        assert info.presentation_dir.name == "attempt_01"

    def test_next_sequence_increments(self, collection_root: Path) -> None:
        parent = collection_root / "bona_fide" / "subject_003"
        first, name1 = next_sequence_dir(parent, "session")
        first.mkdir(parents=True)
        second, name2 = next_sequence_dir(parent, "session")
        assert name1 == "session_01"
        assert name2 == "session_02"
        assert first != second

    def test_path_escape_rejected(self, collection_root: Path) -> None:
        outside = collection_root.parent.parent / "outside"
        with pytest.raises(CollectionValidationError, match="escapes"):
            from tools.collect_pad_dataset import validate_presentation_dir

            validate_presentation_dir(collection_root, outside)


class TestFramesAndMeta:
    def test_frame_filename(self) -> None:
        assert frame_filename(1) == "frame_0001.jpg"
        assert frame_filename(16) == "frame_0016.jpg"

    def test_frame_filename_invalid(self) -> None:
        with pytest.raises(CollectionValidationError):
            frame_filename(0)

    def test_write_meta_json(self, collection_root: Path, config: CollectionConfig) -> None:
        presentation = collection_root / "bona_fide" / "subject_001" / "session_01"
        presentation.mkdir(parents=True)
        path = write_meta_json(
            presentation,
            collector_id=config.collector_id,
            frame_count=12,
            camera_index=0,
            subject_id="subj_001",
            presentation_type="BF",
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["collector_id"] == "test_collector"
        assert data["frame_count"] == 12
        assert data["camera_index"] == 0
        assert data["pai_id"] == "BF"
        assert "captured_at_utc" in data

    def test_save_frame_roundtrip(self, tmp_path: Path, config: CollectionConfig) -> None:
        from tools.collect_pad_dataset import save_frame

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        target = tmp_path / "frame_0001.jpg"
        save_frame(image, target, config.jpeg_quality)
        assert target.is_file()


class TestCaptureTiming:
    def test_sample_interval_seconds_default_fps(self) -> None:
        assert sample_interval_seconds(DEFAULT_CAPTURE_FPS) == pytest.approx(0.2)

    def test_sample_interval_seconds_rejects_non_positive(self) -> None:
        with pytest.raises(CollectionValidationError, match="capture_fps"):
            sample_interval_seconds(0.0)

    def test_effective_capture_fps(self) -> None:
        assert effective_capture_fps(15, 3.0) == pytest.approx(5.0)
        assert effective_capture_fps(0, 3.0) == 0.0
        assert effective_capture_fps(10, 0.0) == 0.0


class TestFrameCountWarnings:
    def test_validate_frame_count_warns_low(self, config: CollectionConfig, monkeypatch: pytest.MonkeyPatch) -> None:
        messages = []
        from tools.collect_pad_dataset import LOGGER
        monkeypatch.setattr(LOGGER, "warning", lambda msg, *args: messages.append(msg % args))
        validate_frame_count(6, config)
        assert any("protocol minimum" in m for m in messages)
