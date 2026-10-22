"""Tests for tools.qc_dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.collect_pad_dataset import frame_filename
from tools.qc_dataset import (
    compute_decision,
    discover_presentations,
    load_manifest,
    run_qc,
    run_qc_and_write,
    write_qc_report,
)


def _write_frames(presentation: Path, count: int, *, fill: int = 180) -> None:
    presentation.mkdir(parents=True, exist_ok=True)
    pytest.importorskip("cv2")
    import cv2

    image = np.full((64, 64, 3), fill, dtype=np.uint8)
    for index in range(1, count + 1):
        assert cv2.imwrite(str(presentation / frame_filename(index)), image)


@pytest.fixture
def good_collection(tmp_path: Path) -> Path:
    root = tmp_path / "face_lock_v1"
    _write_frames(root / "bona_fide/subject_001/session_01", 12, fill=180)
    _write_frames(root / "attacks/PA-01_print/subject_002/attempt_01", 12, fill=181)
    manifest = {
        "manifest_version": "1.0.0",
        "collection_id": "face_lock_v1",
        "samples": [
            {
                "sample_id": "BF_subj001_s01",
                "pai_id": "BF",
                "ground_truth": "bona_fide",
                "subject_id": "subj_001",
                "path": "bona_fide/subject_001/session_01",
                "presentation_type": "clip",
                "frame_count": 12,
            },
            {
                "sample_id": "PA01_subj002_a01",
                "pai_id": "PA-01",
                "ground_truth": "attack",
                "subject_id": "subj_002",
                "path": "attacks/PA-01_print/subject_002/attempt_01",
                "presentation_type": "clip",
                "frame_count": 12,
            },
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class TestDecision:
    def test_compute_decision_pass(self) -> None:
        assert compute_decision([], []) == "PASS"

    def test_compute_decision_warning(self) -> None:
        from tools.qc_dataset import QCIssue

        warnings = [
            QCIssue("x", "warning", "p", "m"),
        ]
        assert compute_decision([], warnings) == "WARNING"

    def test_compute_decision_fail(self) -> None:
        from tools.qc_dataset import QCIssue

        errors = [QCIssue("x", "error", "p", "m")]
        assert compute_decision(errors, []) == "FAIL"


class TestStructure:
    def test_missing_category_folder_warns(self, tmp_path: Path) -> None:
        root = tmp_path / "face_lock_v1"
        root.mkdir()
        _write_frames(root / "bona_fide/subject_001/session_01", 12)
        report = run_qc(root, check_quality=False)
        assert any(issue.code == "missing_category_folder" for issue in report.warnings)

    def test_invalid_subject_folder_errors(self, tmp_path: Path) -> None:
        root = tmp_path / "face_lock_v1"
        _write_frames(root / "bona_fide/bad_name/session_01", 12)
        report = run_qc(root, check_quality=False)
        assert report.decision == "FAIL"
        assert any(issue.code == "invalid_subject_folder" for issue in report.errors)

    def test_invalid_sequence_folder_errors(self, tmp_path: Path) -> None:
        root = tmp_path / "face_lock_v1"
        _write_frames(root / "bona_fide/subject_001/not_a_session", 12)
        report = run_qc(root, check_quality=False)
        assert any(issue.code == "invalid_sequence_folder" for issue in report.errors)


class TestFrames:
    def test_frame_count_below_protocol(self, tmp_path: Path) -> None:
        root = tmp_path / "face_lock_v1"
        _write_frames(root / "bona_fide/subject_001/session_01", 6)
        report = run_qc(root, check_quality=False)
        assert any(issue.code == "frame_count_below_protocol" for issue in report.errors)

    def test_non_sequential_frames(self, tmp_path: Path) -> None:
        root = tmp_path / "face_lock_v1"
        presentation = root / "bona_fide/subject_001/session_01"
        presentation.mkdir(parents=True)
        (presentation / "frame_0001.jpg").write_bytes(b"x")
        (presentation / "frame_0003.jpg").write_bytes(b"x")
        report = run_qc(root, check_quality=False)
        assert any(issue.code == "non_sequential_frames" for issue in report.errors)

    def test_corrupt_jpeg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            from deepface import DeepFace
            monkeypatch.setattr(DeepFace, "extract_faces", lambda *args, **kwargs: [{"confidence": 0.99}])
        except ImportError:
            pass
        cv2 = pytest.importorskip("cv2")
        root = tmp_path / "face_lock_v1"
        presentation = root / "bona_fide/subject_001/session_01"
        presentation.mkdir(parents=True)
        for index in range(1, 10):
            path = presentation / frame_filename(index)
            if index == 5:
                path.write_bytes(b"not-a-jpeg")
            else:
                cv2.imwrite(str(path), np.full((32, 32, 3), 128, dtype=np.uint8))
        report = run_qc(root, check_quality=True)
        assert any(issue.code == "corrupt_jpeg" for issue in report.errors)


class TestManifest:
    def test_duplicate_sample_id(self, good_collection: Path) -> None:
        manifest_path = good_collection / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        duplicate = dict(data["samples"][0])
        data["samples"].append(duplicate)
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        report = run_qc(good_collection, check_quality=False)
        assert any(issue.code == "duplicate_sample_id" for issue in report.errors)

    def test_duplicate_path(self, good_collection: Path) -> None:
        manifest_path = good_collection / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        copy = dict(data["samples"][1])
        copy["sample_id"] = "OTHER_id"
        data["samples"].append(copy)
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        report = run_qc(good_collection, check_quality=False)
        assert any(issue.code == "duplicate_path" for issue in report.errors)

    def test_invalid_pai_and_ground_truth(self, good_collection: Path) -> None:
        manifest_path = good_collection / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Invalid PAI ID
        data["samples"][0]["pai_id"] = "PA-99"
        # Valid PAI ID but mismatched ground truth
        data["samples"][1]["pai_id"] = "PA-01"
        data["samples"][1]["ground_truth"] = "bona_fide"
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        report = run_qc(good_collection, check_quality=False)
        assert any(issue.code == "invalid_pai_id" for issue in report.errors)
        assert any(issue.code == "ground_truth_pai_mismatch" for issue in report.errors)

    def test_manifest_path_missing(self, good_collection: Path) -> None:
        manifest_path = good_collection / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["samples"][0]["path"] = "bona_fide/subject_999/session_99"
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        report = run_qc(good_collection, check_quality=False)
        assert any(issue.code == "manifest_path_missing" for issue in report.errors)


class TestQualityMetrics:
    def test_low_brightness_warning(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            from deepface import DeepFace
            monkeypatch.setattr(DeepFace, "extract_faces", lambda *args, **kwargs: [{"confidence": 0.99}])
        except ImportError:
            pass
        root = tmp_path / "face_lock_v1"
        _write_frames(root / "bona_fide/subject_001/session_01", 12, fill=10)
        report = run_qc(root, check_quality=True)
        assert any(issue.code == "low_brightness" for issue in report.warnings)

    def test_excessive_blur_warning(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            from deepface import DeepFace
            monkeypatch.setattr(DeepFace, "extract_faces", lambda *args, **kwargs: [{"confidence": 0.99}])
        except ImportError:
            pass
        cv2 = pytest.importorskip("cv2")
        root = tmp_path / "face_lock_v1"
        presentation = root / "bona_fide/subject_001/session_01"
        presentation.mkdir(parents=True)
        uniform = np.full((64, 64, 3), 128, dtype=np.uint8)
        for index in range(1, 13):
            cv2.imwrite(str(presentation / frame_filename(index)), uniform)
        report = run_qc(root, check_quality=True)
        assert any(issue.code == "excessive_blur" for issue in report.warnings)

    def test_inconsistent_image_size(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            from deepface import DeepFace
            monkeypatch.setattr(DeepFace, "extract_faces", lambda *args, **kwargs: [{"confidence": 0.99}])
        except ImportError:
            pass
        cv2 = pytest.importorskip("cv2")
        root = tmp_path / "face_lock_v1"
        presentation = root / "bona_fide/subject_001/session_01"
        presentation.mkdir(parents=True)
        for index in range(1, 13):
            size = 64 if index < 6 else 48
            image = np.full((size, size, 3), 180, dtype=np.uint8)
            cv2.imwrite(str(presentation / frame_filename(index)), image)
        report = run_qc(root, check_quality=True)
        assert any(issue.code == "inconsistent_image_size" for issue in report.warnings)

    def test_zero_faces_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            from deepface import DeepFace
            monkeypatch.setattr(DeepFace, "extract_faces", lambda *args, **kwargs: [])
        except ImportError:
            pass
        root = tmp_path / "face_lock_v1"
        _write_frames(root / "bona_fide/subject_001/session_01", 12, fill=180)
        report = run_qc(root, check_quality=True)
        assert any(issue.code == "no_face_detected" for issue in report.errors)

    def test_one_face_pass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            from deepface import DeepFace
            monkeypatch.setattr(DeepFace, "extract_faces", lambda *args, **kwargs: [{"confidence": 0.99}])
        except ImportError:
            pass
        root = tmp_path / "face_lock_v1"
        _write_frames(root / "bona_fide/subject_001/session_01", 12, fill=180)
        report = run_qc(root, check_quality=True)
        assert not any(issue.code == "no_face_detected" for issue in report.errors)
        assert not any(issue.code == "multiple_faces_detected" for issue in report.errors)

    def test_multiple_faces_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            from deepface import DeepFace
            monkeypatch.setattr(DeepFace, "extract_faces", lambda *args, **kwargs: [{"confidence": 0.99}, {"confidence": 0.95}])
        except ImportError:
            pass
        root = tmp_path / "face_lock_v1"
        _write_frames(root / "bona_fide/subject_001/session_01", 12, fill=180)
        report = run_qc(root, check_quality=True)
        assert any(issue.code == "multiple_faces_detected" for issue in report.errors)


class TestReportOutput:
    def test_good_collection_pass_or_warning(self, good_collection: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            from deepface import DeepFace
            monkeypatch.setattr(DeepFace, "extract_faces", lambda *args, **kwargs: [{"confidence": 0.99}])
        except ImportError:
            pass
        report = run_qc(good_collection, check_quality=True)
        assert report.decision in ("PASS", "WARNING")
        assert report.summary["category_counts"]["BF"] == 1

    def test_write_report_file(self, good_collection: Path) -> None:
        output = good_collection / "dataset_qc_report.json"
        report = run_qc_and_write(
            good_collection,
            output,
            check_quality=False,
            dry_run=False,
        )
        assert output.is_file()
        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert loaded["decision"] == report.decision
        assert "errors" in loaded
        assert "warnings" in loaded

    def test_discover_presentations(self, good_collection: Path) -> None:
        errors: list = []
        warnings: list = []
        presentations, subject_sets = discover_presentations(
            good_collection, errors, warnings
        )
        assert len(presentations) == 2
        assert "subj_001" in subject_sets["BF"]

    def test_load_manifest(self, good_collection: Path) -> None:
        manifest = load_manifest(good_collection)
        assert manifest is not None
        assert len(manifest["samples"]) == 2
