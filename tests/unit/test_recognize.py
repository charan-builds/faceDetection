"""
Unit tests for src.recognize.

Focuses on testing load_trusted_records() without mocking load_trusted_records itself,
validating 1:1 owner verification, case-insensitivity, and fail-closed error handling.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import src.config as config_module
import src.recognize as recognize_module
from src.recognize import (
    build_denied_result,
    classify_recognition_error,
    estimate_brightness,
    format_distance,
    is_bad_lighting,
    load_trusted_records,
    should_run_inference,
)


@pytest.mark.unit
class TestLoadTrustedRecords:
    """
    Focused tests for load_trusted_records() calling the real function
    and the real clean_name() import.
    """

    def test_load_trusted_records_success_exact_match(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A valid owner record matching RECOGNITION_SETTINGS.owner_name is loaded."""
        mock_records = [
            {
                "user_name": "owner",
                "embeddings": [[0.1] * 512],
                "model_name": "Facenet512",
            }
        ]
        monkeypatch.setattr(
            "src.recognize.load_all_user_embeddings",
            lambda: mock_records,
        )

        records = load_trusted_records()

        assert len(records) == 1
        assert records[0]["user_name"] == "owner"

    def test_load_trusted_records_case_insensitive_matching(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Matching is case-insensitive via clean_name:
        Config owner 'Alice' matches stored record 'alice' and 'ALICE'.
        """
        rec_settings = replace(config_module.RECOGNITION_SETTINGS, owner_name="Alice")
        monkeypatch.setattr(config_module, "RECOGNITION_SETTINGS", rec_settings)
        monkeypatch.setattr(recognize_module, "RECOGNITION_SETTINGS", rec_settings)

        mock_records = [
            {"user_name": "ALICE", "embeddings": [[0.1] * 512]},
            {"user_name": "bob", "embeddings": [[0.2] * 512]},
        ]
        monkeypatch.setattr(
            "src.recognize.load_all_user_embeddings",
            lambda: mock_records,
        )

        records = load_trusted_records()

        assert len(records) == 1
        assert records[0]["user_name"] == "ALICE"

    def test_load_trusted_records_ignores_non_owner_identities(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        1:1 verification policy: records belonging to other users (impostors)
        are excluded even if validly stored in encodings/.
        """
        rec_settings = replace(config_module.RECOGNITION_SETTINGS, owner_name="charlie")
        monkeypatch.setattr(config_module, "RECOGNITION_SETTINGS", rec_settings)
        monkeypatch.setattr(recognize_module, "RECOGNITION_SETTINGS", rec_settings)

        mock_records = [
            {"user_name": "alice", "embeddings": [[0.1] * 512]},
            {"user_name": "bob", "embeddings": [[0.2] * 512]},
        ]
        monkeypatch.setattr(
            "src.recognize.load_all_user_embeddings",
            lambda: mock_records,
        )

        records = load_trusted_records()

        assert len(records) == 0

    def test_load_trusted_records_empty_store_returns_empty_list(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When encodings/ has no records, load_trusted_records() returns []."""
        monkeypatch.setattr(
            "src.recognize.load_all_user_embeddings",
            lambda: [],
        )

        records = load_trusted_records()

        assert records == []

    def test_load_trusted_records_handles_malformed_records(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Records with missing or empty user_name do not cause crashes."""
        mock_records = [
            {"embeddings": [[0.1] * 512]},  # Missing user_name
            {"user_name": "", "embeddings": [[0.1] * 512]},  # Empty string
            {"user_name": None, "embeddings": [[0.1] * 512]},  # None
            {"user_name": "owner", "embeddings": [[0.1] * 512]},  # Valid owner
        ]
        monkeypatch.setattr(
            "src.recognize.load_all_user_embeddings",
            lambda: mock_records,
        )

        records = load_trusted_records()

        assert len(records) == 1
        assert records[0]["user_name"] == "owner"

    def test_load_trusted_records_fail_closed_on_storage_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        If load_all_user_embeddings() raises an unhandled error (e.g. disk I/O,
        permissions, corruption), load_trusted_records catches it, logs it,
        and returns an empty list (fail-closed).
        """
        def _raise_error() -> list[dict[str, Any]]:
            raise OSError("Encodings directory read failure")

        monkeypatch.setattr(
            "src.recognize.load_all_user_embeddings",
            _raise_error,
        )

        records = load_trusted_records()

        assert records == []


@pytest.mark.unit
class TestRecognizeHelpers:
    """Unit tests for standalone helper functions in src.recognize."""

    def test_estimate_brightness(self) -> None:
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert estimate_brightness(dark_frame) == 0.0

        bright_frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        assert estimate_brightness(bright_frame) == 200.0

    def test_is_bad_lighting(self) -> None:
        dark_frame = np.full((100, 100, 3), 30, dtype=np.uint8)
        assert is_bad_lighting(dark_frame, threshold=45.0) is True

        bright_frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        assert is_bad_lighting(bright_frame, threshold=45.0) is False

    def test_format_distance(self) -> None:
        assert format_distance(None) == "n/a"
        assert format_distance(0.12345) == "0.123"

    def test_build_denied_result(self) -> None:
        result = build_denied_result("Bad lighting", "Move to bright area")
        assert result["access_granted"] is False
        assert result["primary_text"] == "ACCESS DENIED"
        assert result["secondary_text"] == "Bad lighting"
        assert result["detail_text"] == "Move to bright area"

    def test_classify_recognition_error(self) -> None:
        err_multiple = RuntimeError("Expected exactly one face, but found 2.")
        res_multiple = classify_recognition_error(err_multiple)
        assert res_multiple["secondary_text"] == "Multiple faces detected"

        err_no_face = RuntimeError("No usable face detected.")
        res_no_face = classify_recognition_error(err_no_face)
        assert res_no_face["secondary_text"] == "No face detected"

    def test_should_run_inference_cadence(self) -> None:
        # Frame skip 10: frame 10 should run if cooldown elapsed
        assert should_run_inference(frame_number=10, next_allowed_time=0.0, frame_skip=10) is True
        # Frame 9 should not run
        assert should_run_inference(frame_number=9, next_allowed_time=0.0, frame_skip=10) is False
        # Cooldown in the future should not run even on frame 10
        assert should_run_inference(frame_number=10, next_allowed_time=9999999999.0, frame_skip=10) is False

    def test_run_pad_check_passes_face_box_to_pad_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.recognize import run_pad_check
        frame = np.full((100, 100, 3), 150, dtype=np.uint8)
        mock_session = MagicMock()
        mock_session.evaluate_final.return_value = MagicMock()

        # When face_box is passed explicitly
        run_pad_check(frame, frame_number=1, pad_session=mock_session, face_box=(10, 20, 30, 40))
        assert mock_session.add_frame.called
        pad_frame = mock_session.add_frame.call_args[0][0]
        assert pad_frame.face_box == (10, 20, 30, 40)

        # When face_box is None, it calls detect_single_face
        mock_session.reset_mock()
        monkeypatch.setattr("src.recognize.detect_single_face", lambda f: ((5, 10, 15, 20), None))
        run_pad_check(frame, frame_number=2, pad_session=mock_session)
        assert mock_session.add_frame.called
        pad_frame = mock_session.add_frame.call_args[0][0]
        assert pad_frame.face_box == (5, 10, 15, 20)



@pytest.mark.unit
class TestRecognizeStrictMode:
    def test_recognize_live_fails_closed_when_key_missing_in_strict_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """recognize_live() raises RuntimeError immediately if strict mode is on and key is missing."""
        from src.recognize import recognize_live
        monkeypatch.setenv("FACE_LOCK_STRICT_MODE", "1")
        monkeypatch.delenv("FACE_LOCK_SIGNING_KEY", raising=False)

        with pytest.raises(RuntimeError, match="Strict mode is enabled"):
            recognize_live()


@pytest.mark.unit
class TestCheckRecenterCriteria:
    """Unit tests for check_recenter_criteria helper."""

    def test_none_face_box_returns_false(self) -> None:
        from src.recognize import check_recenter_criteria
        ok, reason = check_recenter_criteria(None, "no_face", (480, 640, 3))
        assert ok is False
        assert reason == "no_face"

    def test_face_error_propagated(self) -> None:
        from src.recognize import check_recenter_criteria
        ok, reason = check_recenter_criteria((100, 100, 80, 80), "multiple_faces", (480, 640, 3))
        assert ok is False
        assert reason == "multiple_faces"

    def test_aspect_ratio_out_of_bounds(self) -> None:
        from src.recognize import check_recenter_criteria
        # Extremely wide box (w=160, h=40, ratio=4.0)
        ok, reason = check_recenter_criteria((100, 100, 160, 40), None, (480, 640, 3))
        assert ok is False
        assert reason == "aspect_ratio_out_of_bounds"

    def test_face_size_out_of_bounds(self) -> None:
        from src.recognize import check_recenter_criteria
        # Tiny face (h=30 on 480 frame -> 0.0625 < 0.15)
        ok, reason = check_recenter_criteria((100, 100, 30, 30), None, (480, 640, 3))
        assert ok is False
        assert reason == "face_size_out_of_bounds"

    def test_off_center_in_frame(self) -> None:
        from src.recognize import check_recenter_criteria
        # Face box shoved into the top-left edge: x=5, y=5, w=80, h=80 -> cx=45 (0.07 of 640)
        ok, reason = check_recenter_criteria((5, 5, 80, 80), None, (480, 640, 3))
        assert ok is False
        assert reason == "off_center_in_frame"

    def test_head_still_turned_detected(self) -> None:
        from src.recognize import check_recenter_criteria
        # Centered in frame (x=280, w=100, h=100 -> cx=330 on 640)
        # But challenge start was at cx=400, delta=70px >> 12px
        ok, reason = check_recenter_criteria(
            (280, 150, 100, 100),
            None,
            (480, 640, 3),
            start_centroid_x=400.0,
            max_shift=50.0,
        )
        assert ok is False
        assert reason == "head_still_turned"

    def test_frontal_and_centered_success_with_baseline(self) -> None:
        from src.recognize import check_recenter_criteria
        # Start at 320, current cx = 323 (x=273, w=100, h=100 -> cx=323, cy=200)
        ok, reason = check_recenter_criteria(
            (273, 150, 100, 100),
            None,
            (480, 640, 3),
            start_centroid_x=320.0,
            max_shift=30.0,
        )
        assert ok is True
        assert reason == "frontal_centered"

    def test_frontal_and_centered_success_without_baseline(self) -> None:
        from src.recognize import check_recenter_criteria
        # Normal centered face, start_centroid_x is None
        ok, reason = check_recenter_criteria(
            (270, 150, 100, 100),
            None,
            (480, 640, 3),
            start_centroid_x=None,
        )
        assert ok is True
        assert reason == "frontal_centered"


@pytest.mark.unit
class TestRecognizeFrameActiveChallenge:
    """Unit tests for active PAD challenge handling in recognize_frame."""

    def test_recognize_frame_defers_identity_on_active_challenge_passed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from unittest.mock import MagicMock
        from src.pad_engine import PadOutcome, PadResult, REASON_ACTIVE_PASSED
        from src.recognize import recognize_frame

        mock_generate = MagicMock()
        monkeypatch.setattr("src.recognize.generate_embedding", mock_generate)

        active_passed_result = PadResult(
            outcome=PadOutcome.LIVE,
            reason_code=REASON_ACTIVE_PASSED,
            user_message_key="verifying_liveness",
            model_id="heuristic_rgb_v1",
            model_version="1.0.0",
            frames_analyzed=10,
            latency_ms=1.5,
            active_passed=True,
        )

        frame = np.full((100, 100, 3), 150, dtype=np.uint8)
        trusted_records = [{"user_name": "owner", "embeddings": [[0.1] * 512]}]

        result = recognize_frame(
            frame=frame,
            trusted_records=trusted_records,
            threshold=0.30,
            frame_number=10,
            pad_result=active_passed_result,
        )

        # Must NOT call embedding generation on challenge completion frame!
        mock_generate.assert_not_called()
        assert result["access_granted"] is False
        assert result["state"] == "LIVENESS VERIFIED — LOOK AT CAMERA"
        assert result["primary_text"] == "LIVENESS VERIFIED — LOOK AT CAMERA"
        assert result["liveness_passed"] is True
        assert result["pad_reason_code"] == REASON_ACTIVE_PASSED
