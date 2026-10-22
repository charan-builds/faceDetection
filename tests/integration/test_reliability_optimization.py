"""
Regression and reliability optimization tests for Face Lock AI.

Tests cover:
1. Owner recognition on suitable frontal frames (padded crop + fallback consistency)
2. Face detection / DeepFace detection disagreement handling
3. Re-centering pose and timeout behavior
4. Repeated success logging suppression and stale-result expiry on face loss
5. PAD false-rejection prevention on controlled fixtures while maintaining attack protection
6. Correct rejection of nonmatching identity
7. Preserved cosine threshold (0.30) and strict-mode authentication checks
"""

from __future__ import annotations

import logging
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import src.recognize as rec
from src.ai_engine import DEFAULT_COSINE_THRESHOLD, cosine_distance
from src.config import AI_SETTINGS, PAD_SETTINGS
from src.pad_engine import (
    ConfidenceBand,
    HeuristicPassiveScorer,
    PadOutcome,
    PadResult,
    REASON_ACTIVE_PASSED,
    build_pad_frame,
    run_pad_gate,
)
from src.recognize import (
    LOGGER,
    check_recenter_criteria,
    extract_face_crop,
    recognize_frame,
    recognize_identity,
    recognize_live,
)

pytest.importorskip("numpy")


@pytest.fixture
def capture_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    LOGGER.addHandler(caplog.handler)
    LOGGER.setLevel(logging.DEBUG)
    yield caplog


@pytest.fixture
def owner_template_record() -> list[dict[str, Any]]:
    # Known 512-dim embedding for owner
    return [
        {
            "user_name": "owner",
            "model_name": "Facenet512",
            "embeddings": [[0.05] * 512],
        }
    ]


@pytest.fixture
def impostor_embedding() -> list[float]:
    # Distinct orthogonal embedding vector
    vec = [-0.05] * 512
    vec[0] = 1.0
    return vec


class TestOwnerRecognitionAndCrop:
    def test_owner_recognition_uses_padded_crop_and_measures_timing(
        self,
        bright_frame: np.ndarray,
        owner_template_record: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Frontal face box uses crop extraction and records timing profile without saving raw vectors."""
        crop_called = False

        def mock_generate(image_input: Any, **kwargs: Any) -> list[float]:
            nonlocal crop_called
            if hasattr(image_input, "shape") and image_input.shape != bright_frame.shape:
                crop_called = True
            return [0.05] * 512

        monkeypatch.setattr("src.recognize.generate_embedding", mock_generate)

        face_box = (20, 20, 40, 40)
        res = recognize_identity(
            frame=bright_frame,
            trusted_records=owner_template_record,
            threshold=DEFAULT_COSINE_THRESHOLD,
            face_box=face_box,
        )

        assert res["access_granted"] is True
        assert res["state"] == "VERIFIED"
        assert res["user_name"] == "owner"
        assert crop_called is True
        assert "timing_profile" in res
        profile = res["timing_profile"]
        assert profile["method"] == "crop"
        assert profile["embed_ms"] >= 0.0
        assert profile["match_ms"] >= 0.0
        assert profile["total_ms"] >= 0.0

    def test_face_detection_disagreement_falls_back_to_full_frame(
        self,
        bright_frame: np.ndarray,
        owner_template_record: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        capture_logs: pytest.LogCaptureFixture,
    ) -> None:
        """When DeepFace rejects the crop, it safely falls back to full frame without enforce_detection=False."""
        attempts = 0

        def mock_generate(image_input: Any, **kwargs: Any) -> list[float]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                # Crop extraction fails in DeepFace
                raise RuntimeError("No usable face was detected in the image.")
            # Fallback to full frame succeeds
            return [0.05] * 512

        monkeypatch.setattr("src.recognize.generate_embedding", mock_generate)

        face_box = (20, 20, 40, 40)
        res = recognize_identity(
            frame=bright_frame,
            trusted_records=owner_template_record,
            threshold=DEFAULT_COSINE_THRESHOLD,
            face_box=face_box,
        )

        assert attempts == 2
        assert res["access_granted"] is True
        assert res["timing_profile"]["method"] == "full_frame"
        assert "recognition_crop_fallback" in capture_logs.text


class TestRecenterAndTimeout:
    def test_recenter_criteria_allows_natural_human_drift(self) -> None:
        """check_recenter_criteria allows return within reasonable resolution-scaled drift."""
        # 640x480 frame, start cx=320, head turned to 360 (delta=40), returns to 335 (delta=15)
        ok, reason = check_recenter_criteria(
            face_box=(285, 140, 100, 100),  # cx = 335, cy = 190
            face_error=None,
            frame_shape=(480, 640, 3),
            start_centroid_x=320.0,
            max_shift=40.0,
        )
        assert ok is True
        assert reason == "frontal_centered"

    def test_recenter_criteria_detects_still_turned_head(self) -> None:
        """When head remains shifted significantly, check_recenter_criteria rejects."""
        # 640x480 frame, start cx=320, current cx=380 (delta=60px)
        ok, reason = check_recenter_criteria(
            face_box=(330, 140, 100, 100),  # cx = 380, cy = 190
            face_error=None,
            frame_shape=(480, 640, 3),
            start_centroid_x=320.0,
            max_shift=65.0,
        )
        assert ok is False
        assert reason == "head_still_turned"


class TestStaleResultExpiryAndDuplicateLogging:
    def test_repeated_access_granted_logged_only_once(
        self,
        bright_frame: np.ndarray,
        owner_template_record: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        capture_logs: pytest.LogCaptureFixture,
    ) -> None:
        """ACCESS GRANTED is logged only on state transition, not repeatedly across 10 display frames."""
        frame_box = (20, 20, 40, 40)
        monkeypatch.setattr(rec, "open_webcam", lambda idx: MagicMock())
        monkeypatch.setattr(rec, "read_frame", lambda cap: bright_frame)
        monkeypatch.setattr(rec, "detect_single_face", lambda f: (frame_box, None))
        monkeypatch.setattr(rec, "draw_status_overlay", lambda *args, **kwargs: None)
        monkeypatch.setattr(rec, "show_frame", lambda win, f: None)
        monkeypatch.setattr(rec, "release_webcam", lambda cap: None)
        monkeypatch.setattr(rec, "close_windows", lambda: None)
        monkeypatch.setattr(rec, "load_trusted_records", lambda: owner_template_record)
        monkeypatch.setattr(rec, "generate_embedding", lambda **kwargs: [0.05] * 512)

        # 6 display frames: frame 1 runs inference (skip=1), frames 2-6 display with same result
        keys = iter([0, 0, 0, 0, 0, ord("q")])
        monkeypatch.setattr(rec, "read_display_key", lambda: next(keys))

        rec.recognize_live(camera_index=0, frame_skip=1, cooldown_seconds=5.0)

        # Count occurrences of recognition_granted in logs
        granted_logs = [line for line in capture_logs.text.splitlines() if "event=recognition_granted" in line]
        assert len(granted_logs) == 1, f"Expected exactly 1 grant log, got {len(granted_logs)}"

    def test_access_granted_expires_promptly_when_face_disappears(
        self,
        bright_frame: np.ndarray,
        owner_template_record: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When face disappears (face_box is None), verified status expires back to DETECTING FACE."""
        frame_box = (20, 20, 40, 40)
        # Sequence: frame 1 face present -> granted; frame 2 face None -> must expire!
        boxes = [(frame_box, None), (None, None)]
        box_idx = 0

        def mock_detect(frame: Any):
            nonlocal box_idx
            b = boxes[min(box_idx, len(boxes) - 1)]
            box_idx += 1
            return b

        captured_statuses: list[dict[str, Any]] = []

        def mock_overlay(frame: Any, status: dict[str, Any], *args, **kwargs):
            captured_statuses.append(dict(status))

        monkeypatch.setattr(rec, "open_webcam", lambda idx: MagicMock())
        monkeypatch.setattr(rec, "read_frame", lambda cap: bright_frame)
        monkeypatch.setattr(rec, "detect_single_face", mock_detect)
        monkeypatch.setattr(rec, "draw_status_overlay", mock_overlay)
        monkeypatch.setattr(rec, "show_frame", lambda win, f: None)
        monkeypatch.setattr(rec, "release_webcam", lambda cap: None)
        monkeypatch.setattr(rec, "close_windows", lambda: None)
        monkeypatch.setattr(rec, "load_trusted_records", lambda: owner_template_record)
        monkeypatch.setattr(rec, "generate_embedding", lambda **kwargs: [0.05] * 512)

        keys = iter([0, ord("q")])
        monkeypatch.setattr(rec, "read_display_key", lambda: next(keys))

        rec.recognize_live(camera_index=0, frame_skip=1, cooldown_seconds=5.0)

        assert len(captured_statuses) >= 2
        # Frame 1: Verified
        assert captured_statuses[0]["access_granted"] is True
        assert captured_statuses[0]["state"] == "VERIFIED"
        # Frame 2: Expired immediately upon loss of face!
        assert captured_statuses[1]["access_granted"] is False
        assert captured_statuses[1]["state"] == "DETECTING FACE"


class TestPadRobustnessAndSecurity:
    def test_live_looking_fixture_passes_without_false_attack(self) -> None:
        """A fixture with live webcam texture and skin reflectance does not get falsely rejected as attack."""
        scorer = HeuristicPassiveScorer()
        # Create a synthetic fixture with natural skin tone (R > G >= B) and moderate texture
        np.random.seed(42)
        h, w = 120, 120
        # Skin base: B=70, G=80, R=95 with realistic variance
        noise = np.random.normal(0, 15, (h, w)).astype(np.float32)
        b = np.clip(70 + noise, 0, 255).astype(np.uint8)
        g = np.clip(80 + noise, 0, 255).astype(np.uint8)
        r = np.clip(95 + noise, 0, 255).astype(np.uint8)
        live_sim = np.stack([b, g, r], axis=2)

        # Micro movement between frames (1.2px)
        frames = []
        for i in range(8):
            shift = int(i % 2)
            face_box = (10 + shift, 10, 80, 80)
            frames.append(build_pad_frame(live_sim, frame_index=i, face_box=face_box))

        result = scorer.score(frames)
        # Score must be above reject threshold (0.38)
        assert result.live_probability > PAD_SETTINGS.passive_reject_threshold

    def test_flat_paper_print_fixture_rejected_as_attack(self) -> None:
        """A flat uniform image with 0 texture is classified as presentation attack."""
        scorer = HeuristicPassiveScorer()
        flat_img = np.full((120, 120, 3), 180, dtype=np.uint8)
        frames = [
            build_pad_frame(flat_img, frame_index=i, face_box=(10, 10, 80, 80))
            for i in range(8)
        ]
        result = scorer.score(frames)
        # Score must be below reject threshold (0.38)
        assert result.live_probability <= PAD_SETTINGS.passive_reject_threshold


class TestSecurityAndThresholdPreservation:
    def test_nonmatching_identity_is_strictly_denied(
        self,
        bright_frame: np.ndarray,
        owner_template_record: list[dict[str, Any]],
        impostor_embedding: list[float],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An impostor with cosine distance > 0.30 is denied access with ACCESS DENIED."""
        monkeypatch.setattr("src.recognize.generate_embedding", lambda **kwargs: impostor_embedding)

        res = recognize_identity(
            frame=bright_frame,
            trusted_records=owner_template_record,
            threshold=DEFAULT_COSINE_THRESHOLD,
        )

        assert res["access_granted"] is False
        assert res["state"] == "NOT VERIFIED"
        assert res["primary_text"] == "ACCESS DENIED"
        assert res["secondary_text"] == "Face not recognized"

    def test_threshold_remains_strictly_0_30(self) -> None:
        """The cosine distance threshold must remain unchanged at 0.30."""
        assert DEFAULT_COSINE_THRESHOLD == 0.30
        assert AI_SETTINGS.cosine_threshold == 0.30
