"""Tests for tools.generate_manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.collect_pad_dataset import frame_filename
from tools.generate_manifest import (
    CollectionStatistics,
    ManifestGenerationError,
    build_manifest_document,
    count_frames,
    folder_name_to_subject_id,
    generate_manifest,
    make_sample_id,
    scan_collection,
    write_manifest,
)


def _write_presentation(
    root: Path,
    rel_path: str,
    frame_count: int,
) -> None:
    presentation = root / rel_path
    presentation.mkdir(parents=True, exist_ok=True)
    for index in range(1, frame_count + 1):
        (presentation / frame_filename(index)).write_bytes(b"fake-jpeg")


@pytest.fixture
def mini_collection(tmp_path: Path) -> Path:
    """Small valid collection tree."""
    root = tmp_path / "face_lock_v1"
    _write_presentation(root, "bona_fide/subject_001/session_01", 12)
    _write_presentation(root, "bona_fide/subject_001/session_02", 10)
    _write_presentation(root, "attacks/PA-01_print/subject_002/attempt_01", 14)
    _write_presentation(root, "attacks/PA-02_screen_static/subject_003/attempt_01", 8)
    _write_presentation(root, "attacks/PA-03_screen_video/subject_003/attempt_02", 16)
    return root


class TestHelpers:
    def test_folder_name_to_subject_id(self) -> None:
        assert folder_name_to_subject_id("subject_007") == "subj_007"
        assert folder_name_to_subject_id("bad_name") is None

    def test_make_sample_id(self) -> None:
        assert make_sample_id("BF", "subj_001", "session_03") == "BF_subj001_s03"
        assert make_sample_id("PA-01", "subj_007", "attempt_02") == "PA01_subj007_a02"

    def test_count_frames_with_gap(self, tmp_path: Path) -> None:
        presentation = tmp_path / "clip"
        presentation.mkdir()
        (presentation / "frame_0001.jpg").write_bytes(b"x")
        (presentation / "frame_0003.jpg").write_bytes(b"x")
        count, missing = count_frames(presentation)
        assert count == 2
        assert 2 in missing


class TestScanCollection:
    def test_scan_finds_presentations(self, mini_collection: Path) -> None:
        result = scan_collection(mini_collection)
        assert len(result.samples) == 5
        assert result.statistics.bf == 2
        assert result.statistics.pa01 == 1
        assert result.statistics.pa02 == 1
        assert result.statistics.pa03 == 1
        assert result.statistics.pa04 == 0

    def test_scan_sample_fields(self, mini_collection: Path) -> None:
        result = scan_collection(mini_collection)
        bf = next(sample for sample in result.samples if sample.pai_id == "BF")
        assert bf.ground_truth == "bona_fide"
        assert bf.subject_id == "subj_001"
        assert bf.path == "bona_fide/subject_001/session_01"
        assert bf.presentation_type == "clip"
        assert bf.frame_count == 12

    def test_scan_warns_invalid_subject_folder(self, tmp_path: Path) -> None:
        root = tmp_path / "face_lock_v1"
        _write_presentation(root, "bona_fide/invalid_subject/session_01", 12)
        result = scan_collection(root)
        assert any("Invalid subject folder" in warning.message for warning in result.warnings)
        assert len(result.samples) == 0

    def test_scan_warns_missing_frames(self, tmp_path: Path) -> None:
        root = tmp_path / "face_lock_v1"
        empty = root / "bona_fide/subject_001/session_01"
        empty.mkdir(parents=True)
        result = scan_collection(root)
        assert any("No frame_" in warning.message for warning in result.warnings)

    def test_scan_warns_low_frame_count(self, mini_collection: Path) -> None:
        result = scan_collection(mini_collection, min_frames_warn=12)
        assert any(
            "frame count 10" in warning.message.lower() or "Frame count 10" in warning.message
            for warning in result.warnings
        )


class TestManifestWrite:
    def test_build_manifest_document(self, mini_collection: Path) -> None:
        result = scan_collection(mini_collection)
        document = build_manifest_document(
            collection_id="face_lock_v1",
            samples=result.samples,
        )
        assert document["manifest_version"] == "1.0.0"
        assert document["collection_id"] == "face_lock_v1"
        assert len(document["samples"]) == 5
        assert document["protocol_ids"] == ["PAD-ISO-LAB-v1"]

    def test_generate_manifest_writes_file(self, mini_collection: Path) -> None:
        output = mini_collection / "manifest.json"
        document, result = generate_manifest(mini_collection, output)
        assert output.is_file()
        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert loaded["collection_id"] == "face_lock_v1"
        assert len(loaded["samples"]) == len(document["samples"])

    def test_generate_manifest_dry_run(self, mini_collection: Path) -> None:
        output = mini_collection / "manifest.json"
        generate_manifest(mini_collection, output, dry_run=True)
        assert not output.exists()

    def test_strict_mode_fails_on_errors(self, tmp_path: Path) -> None:
        root = tmp_path / "face_lock_v1"
        _write_presentation(root, "bona_fide/bad_subject/session_01", 12)
        with pytest.raises(ManifestGenerationError, match="Strict mode"):
            generate_manifest(root, strict=True)

    def test_empty_collection_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "face_lock_v1"
        root.mkdir()
        with pytest.raises(ManifestGenerationError, match="No valid samples"):
            generate_manifest(root)

    def test_statistics_unique_subjects(self, mini_collection: Path) -> None:
        stats = scan_collection(mini_collection).statistics
        assert len(stats.subjects_bf) == 1
        assert len(stats.subjects_pa03) == 1
        assert stats.as_dict()["unique_subjects"]["BF"] == 1
