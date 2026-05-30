"""
Integration tests for the live recognition workflow in src.recognize.

These tests wire recognize.py with real find_best_match / should_run_inference logic
while mocking capture, display, DeepFace (generate_embedding), and the webcam.

No real camera, no OpenCV windows, and no DeepFace inference.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.recognize import (
    LOGGER,
    recognize_frame,
    recognize_live,
    should_run_inference,
)

from tests.integration.conftest import EMBEDDING_ALICE, EMBEDDING_STRANGER


pytest.importorskip("numpy")


@pytest.fixture
def capture_recognize_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Attach caplog to the recognize module logger (propagate is False)."""
    LOGGER.addHandler(caplog.handler)
    LOGGER.setLevel(logging.DEBUG)
    yield caplog


@pytest.mark.integration
class TestRecognizeFrameWorkflow:
    """recognize_frame() with mocked generate_embedding and real comparison logic."""

    def test_successful_recognition_grants_access(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
    ) -> None:
        monkeypatch.setattr(
            "src.recognize.generate_embedding",
            lambda **kwargs: EMBEDDING_ALICE,
        )

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=trusted_records_alice,
            threshold=0.30,
        )

        assert result["access_granted"] is True
        assert result["primary_text"] == "ACCESS GRANTED"
        assert result["user_name"] == "alice"
        assert result["cosine_distance"] == pytest.approx(0.0)

    def test_failed_recognition_when_face_does_not_match(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
        capture_recognize_logs: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            "src.recognize.generate_embedding",
            lambda **kwargs: EMBEDDING_STRANGER,
        )

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=trusted_records_alice,
            threshold=0.30,
        )

        assert result["access_granted"] is False
        assert result["primary_text"] == "ACCESS DENIED"
        assert "Closest user: alice" in result["secondary_text"]
        assert "event=recognition_denied" in capture_recognize_logs.text

    def test_no_registered_users_denies_without_calling_embedding(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
    ) -> None:
        mock_generate = MagicMock(return_value=EMBEDDING_ALICE)
        monkeypatch.setattr("src.recognize.generate_embedding", mock_generate)

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=[],
            threshold=0.30,
        )

        mock_generate.assert_not_called()
        assert result["secondary_text"] == "No registered users"

    def test_no_face_detected_maps_to_friendly_denied_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
        capture_recognize_logs: pytest.LogCaptureFixture,
    ) -> None:
        def raise_no_face(**kwargs: Any) -> list[float]:
            raise RuntimeError("No usable face was detected in the image.")

        monkeypatch.setattr("src.recognize.generate_embedding", raise_no_face)

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=trusted_records_alice,
            threshold=0.30,
        )

        assert result["access_granted"] is False
        assert result["secondary_text"] == "No face detected"
        assert "event=recognition_inference_failed" in capture_recognize_logs.text

    def test_multiple_faces_maps_to_multiple_faces_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
    ) -> None:
        def raise_multiple(**kwargs: Any) -> list[float]:
            raise RuntimeError("Expected exactly one face, but found 2.")

        monkeypatch.setattr("src.recognize.generate_embedding", raise_multiple)

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=trusted_records_alice,
            threshold=0.30,
        )

        assert result["secondary_text"] == "Multiple faces detected"

    def test_bad_lighting_denies_before_embedding(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dark_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
        capture_recognize_logs: pytest.LogCaptureFixture,
    ) -> None:
        mock_generate = MagicMock(return_value=EMBEDDING_ALICE)
        monkeypatch.setattr("src.recognize.generate_embedding", mock_generate)

        result = recognize_frame(
            frame=dark_frame,
            trusted_records=trusted_records_alice,
            threshold=0.30,
        )

        mock_generate.assert_not_called()
        assert result["secondary_text"] == "Bad lighting"
        assert "event=recognition_low_lighting" in capture_recognize_logs.text

    def test_threshold_rejection_denies_closest_user(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
    ) -> None:
        """Orthogonal embedding is closest to alice but still above a strict threshold."""
        monkeypatch.setattr(
            "src.recognize.generate_embedding",
            lambda **kwargs: EMBEDDING_STRANGER,
        )

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=trusted_records_alice,
            threshold=0.10,
        )

        assert result["access_granted"] is False
        assert result["cosine_distance"] == pytest.approx(1.0)

    def test_embedding_generation_failure_returns_generic_denied_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
    ) -> None:
        def raise_unknown(**kwargs: Any) -> list[float]:
            raise RuntimeError("GPU unavailable")

        monkeypatch.setattr("src.recognize.generate_embedding", raise_unknown)

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=trusted_records_alice,
            threshold=0.30,
        )

        assert result["secondary_text"] == "Recognition failed"
        assert "GPU unavailable" not in result["secondary_text"]


@pytest.mark.integration
class TestSchedulingLogic:
    """Frame skip and cooldown gates used by recognize_live()."""

    def test_frame_skip_only_runs_on_multiples(self) -> None:
        assert should_run_inference(frame_number=9, next_allowed_time=0.0, frame_skip=10) is False
        assert should_run_inference(frame_number=10, next_allowed_time=0.0, frame_skip=10) is True

    def test_cooldown_blocks_inference_until_time_advances(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = {"now": 100.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

        assert should_run_inference(frame_number=10, next_allowed_time=105.0, frame_skip=1) is False

        clock["now"] = 106.0
        assert should_run_inference(frame_number=11, next_allowed_time=105.0, frame_skip=1) is True

    def test_frame_skip_zero_is_treated_as_one(self) -> None:
        assert should_run_inference(frame_number=1, next_allowed_time=0.0, frame_skip=0) is True


@pytest.mark.integration
class TestRecognizeLiveLoop:
    """
    recognize_live() coordinates capture, AI, display, and logging with mocks.
    """

    @pytest.fixture
    def mock_live_dependencies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Patch I/O and AI boundaries for a short synthetic live loop."""
        state: dict[str, Any] = {
            "frame_reads": 0,
            "embedding_calls": 0,
            "max_frames": 6,
        }

        def fake_open_webcam(camera_index: int) -> str:
            return "fake-cap"

        def fake_read_frame(cap: Any) -> np.ndarray:
            state["frame_reads"] += 1
            return bright_frame

        def fake_generate_embedding(**kwargs: Any) -> list[float]:
            state["embedding_calls"] += 1
            return EMBEDDING_ALICE

        keys = [0, 0, 0, 0, 0, ord("q")]

        def fake_read_display_key(delay_ms: int = 1) -> int:
            index = min(state["frame_reads"] - 1, len(keys) - 1)
            return keys[index]

        monkeypatch.setattr("src.recognize.load_trusted_records", lambda: trusted_records_alice)
        monkeypatch.setattr("src.recognize.open_webcam", fake_open_webcam)
        monkeypatch.setattr("src.recognize.read_frame", fake_read_frame)
        monkeypatch.setattr("src.recognize.release_webcam", lambda cap: None)
        monkeypatch.setattr("src.recognize.close_windows", lambda: None)
        monkeypatch.setattr("src.recognize.show_frame", lambda *args, **kwargs: None)
        monkeypatch.setattr("src.recognize.draw_status_overlay", lambda *args, **kwargs: None)
        monkeypatch.setattr("src.recognize.generate_embedding", fake_generate_embedding)
        monkeypatch.setattr("src.recognize.read_display_key", fake_read_display_key)

        return state

    def test_recognize_live_returns_granted_status(
        self, mock_live_dependencies: dict[str, Any], capture_recognize_logs: pytest.LogCaptureFixture
    ) -> None:
        result = recognize_live(frame_skip=2, cooldown_seconds=0.0)

        assert result["access_granted"] is True
        assert result["user_name"] == "alice"
        assert "event=recognition_started" in capture_recognize_logs.text
        assert "event=recognition_granted" in capture_recognize_logs.text
        assert "event=recognition_stopped_by_user" in capture_recognize_logs.text

    def test_recognize_live_respects_frame_skip(
        self, monkeypatch: pytest.MonkeyPatch, mock_live_dependencies: dict[str, Any]
    ) -> None:
        state = mock_live_dependencies

        def fake_read_display_key(delay_ms: int = 1) -> int:
            # Quit after 8 frames so multiples of 4 (frames 4 and 8) can run.
            if state["frame_reads"] >= 8:
                return ord("q")
            return 0

        monkeypatch.setattr("src.recognize.read_display_key", fake_read_display_key)

        recognize_live(frame_skip=4, cooldown_seconds=0.0)

        assert state["embedding_calls"] == 2

    def test_recognize_live_respects_cooldown(
        self, monkeypatch: pytest.MonkeyPatch, mock_live_dependencies: dict[str, Any]
    ) -> None:
        state = mock_live_dependencies
        clock = {"now": 0.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

        def advance_clock_on_embedding(**kwargs: Any) -> list[float]:
            state["embedding_calls"] += 1
            clock["now"] += 1.0
            return EMBEDDING_ALICE

        monkeypatch.setattr("src.recognize.generate_embedding", advance_clock_on_embedding)

        recognize_live(frame_skip=1, cooldown_seconds=5.0)

        # Cooldown is longer than the short loop; only the first eligible frame runs AI.
        assert state["embedding_calls"] == 1

    def test_recognize_live_handles_camera_runtime_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        trusted_records_alice: list[dict[str, Any]],
        capture_recognize_logs: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr("src.recognize.load_trusted_records", lambda: trusted_records_alice)
        monkeypatch.setattr("src.recognize.open_webcam", lambda idx: "fake-cap")

        def raise_camera_error(cap: Any) -> np.ndarray:
            raise RuntimeError("Webcam disconnected")

        monkeypatch.setattr("src.recognize.read_frame", raise_camera_error)
        monkeypatch.setattr("src.recognize.release_webcam", lambda cap: None)
        monkeypatch.setattr("src.recognize.close_windows", lambda: None)
        monkeypatch.setattr("src.recognize.show_frame", lambda *a, **k: None)
        monkeypatch.setattr("src.recognize.draw_status_overlay", lambda *a, **k: None)

        result = recognize_live()

        assert result["secondary_text"] == "Camera error"
        assert "event=recognition_runtime_error" in capture_recognize_logs.text

    def test_recognize_live_with_no_registered_users_stays_denied(
        self, monkeypatch: pytest.MonkeyPatch, bright_frame: np.ndarray
    ) -> None:
        monkeypatch.setattr("src.recognize.load_trusted_records", lambda: [])
        monkeypatch.setattr("src.recognize.open_webcam", lambda idx: "fake-cap")
        monkeypatch.setattr("src.recognize.read_frame", lambda cap: bright_frame)
        monkeypatch.setattr("src.recognize.release_webcam", lambda cap: None)
        monkeypatch.setattr("src.recognize.close_windows", lambda: None)
        monkeypatch.setattr("src.recognize.show_frame", lambda *a, **k: None)
        monkeypatch.setattr("src.recognize.draw_status_overlay", lambda *a, **k: None)
        monkeypatch.setattr("src.recognize.read_display_key", lambda delay_ms=1: ord("q"))

        mock_generate = MagicMock(return_value=EMBEDDING_ALICE)
        monkeypatch.setattr("src.recognize.generate_embedding", mock_generate)

        result = recognize_live(frame_skip=1, cooldown_seconds=0.0)

        mock_generate.assert_not_called()
        assert result["primary_text"] == "ACCESS DENIED"
        assert result["secondary_text"] == "No registered users"


@pytest.mark.integration
class TestRecognizePadIntegration:
    """Recognition workflow with PAD enabled."""

    def test_pad_attack_blocks_embedding(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
    ) -> None:
        from dataclasses import replace

        import src.config as config_module
        import src.recognize as recognize_module

        pad_on = replace(
            config_module.PAD_SETTINGS,
            enabled=True,
            mode="passive_only",
            passive_model_id="mock_always_attack",
            min_frames_passive=2,
        )
        monkeypatch.setattr(config_module, "PAD_SETTINGS", pad_on)
        monkeypatch.setattr(recognize_module, "PAD_SETTINGS", pad_on)

        mock_generate = MagicMock(return_value=EMBEDDING_ALICE)
        monkeypatch.setattr("src.recognize.generate_embedding", mock_generate)

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=trusted_records_alice,
            threshold=0.30,
            frame_number=2,
        )

        mock_generate.assert_not_called()
        assert result["access_granted"] is False
        assert result["liveness_passed"] is False
        assert "Liveness" in result["secondary_text"]

    def test_pad_live_allows_embedding(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
    ) -> None:
        from dataclasses import replace

        import src.config as config_module
        import src.recognize as recognize_module

        pad_on = replace(
            config_module.PAD_SETTINGS,
            enabled=True,
            mode="passive_only",
            passive_model_id="mock_always_live",
            min_frames_passive=2,
        )
        monkeypatch.setattr(config_module, "PAD_SETTINGS", pad_on)
        monkeypatch.setattr(recognize_module, "PAD_SETTINGS", pad_on)

        monkeypatch.setattr(
            "src.recognize.generate_embedding",
            lambda **kwargs: EMBEDDING_ALICE,
        )

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=trusted_records_alice,
            threshold=0.30,
            frame_number=2,
        )

        assert result["access_granted"] is True
        assert result["liveness_passed"] is True
