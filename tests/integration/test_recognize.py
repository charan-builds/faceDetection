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
        # 1:1 owner-verification model: the UI does not expose which registered
        # user was closest (privacy / anti-enumeration). The closest_user is
        # still available in the structured log for audit purposes.
        assert result["secondary_text"] == "Face not recognized"
        assert "event=recognition_denied" in capture_recognize_logs.text
        assert "closest_user=alice" in capture_recognize_logs.text

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
class TestOwnerVerificationWorkflow:
    """1:1 Single-User Laptop Security tests (Owner vs Impostor)."""

    def test_owner_with_pad_pass_is_allowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
    ) -> None:
        from dataclasses import replace
        import src.config as config_module
        import src.recognize as recognize_module

        # Enforce that alice is the owner
        rec_on = replace(config_module.RECOGNITION_SETTINGS, owner_name="alice")
        monkeypatch.setattr(config_module, "RECOGNITION_SETTINGS", rec_on)
        monkeypatch.setattr(recognize_module, "RECOGNITION_SETTINGS", rec_on)

        monkeypatch.setattr(
            "src.recognize.generate_embedding",
            lambda **kwargs: EMBEDDING_ALICE,
        )

        # load_trusted_records filters for the owner "alice"
        owner_records = [r for r in trusted_records_alice if r["user_name"] == "alice"]

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=owner_records,
            threshold=0.30,
        )

        assert result["access_granted"] is True
        assert result["user_name"] == "alice"

    def test_impostor_with_pad_pass_is_denied_even_if_registered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
    ) -> None:
        """
        If 'bob' tries to authenticate on 'alice's laptop, he is DENIED,
        even if bob is somehow registered in the encodings folder.
        """
        from dataclasses import replace
        import src.config as config_module
        import src.recognize as recognize_module

        # Enforce that alice is the owner
        rec_on = replace(config_module.RECOGNITION_SETTINGS, owner_name="alice")
        monkeypatch.setattr(config_module, "RECOGNITION_SETTINGS", rec_on)
        monkeypatch.setattr(recognize_module, "RECOGNITION_SETTINGS", rec_on)

        monkeypatch.setattr(
            "src.recognize.generate_embedding",
            lambda **kwargs: EMBEDDING_STRANGER,
        )

        # Bob is in trusted_records (the actual test just uses alice records, but we simulate bob matching a registered face)
        bob_record = {"user_name": "bob", "embeddings": [EMBEDDING_STRANGER]}
        all_records = trusted_records_alice + [bob_record]

        # recognize_live filters the records BEFORE recognize_frame is called
        filtered_records = [r for r in all_records if r["user_name"] == "alice"]

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=filtered_records,
            threshold=0.30,
        )

        assert result["access_granted"] is False
        assert result["primary_text"] == "ACCESS DENIED"


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

    def test_recognize_live_cooldown_not_starved_by_intermediate_frames(
        self, monkeypatch: pytest.MonkeyPatch, mock_live_dependencies: dict[str, Any]
    ) -> None:
        """
        Regression test: intermediate non-selected frames (where frame_number % frame_skip != 0)
        must NOT update next_allowed_time. If they did, inference would be starved perpetually.
        """
        state = mock_live_dependencies
        clock = {"now": 0.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

        def advance_clock_on_read(cap: Any) -> np.ndarray:
            clock["now"] += 0.033
            state["frame_reads"] += 1
            return np.full((48, 64, 3), 200, dtype=np.uint8)

        monkeypatch.setattr("src.recognize.read_frame", advance_clock_on_read)

        def quit_after_12_frames(delay_ms: int = 1) -> int:
            if state["frame_reads"] >= 12:
                return ord("q")
            return 0

        monkeypatch.setattr("src.recognize.read_display_key", quit_after_12_frames)

        # frame_skip=5, cooldown=1.0s.
        # Frames 1..4 advance clock to ~0.132s.
        # Frame 5 arrives at ~0.165s -> multiple of 5, cooldown (0.0) satisfied.
        # Inference runs! embedding_calls becomes 1.
        # next_allowed_time is then updated to 0.165 + 1.0 = 1.165s.
        # Frames 6..12 advance clock to ~0.396s, so frame 10 skips due to cooldown.
        recognize_live(frame_skip=5, cooldown_seconds=1.0)

        assert state["embedding_calls"] == 1

    def test_recognize_live_passes_face_box_to_pad_session(
        self, monkeypatch: pytest.MonkeyPatch, mock_live_dependencies: dict[str, Any]
    ) -> None:
        """
        Regression test: detect_single_face box is passed into build_pad_frame for PAD tracking.
        """
        from dataclasses import replace
        import src.config as config_module
        import src.recognize as recognize_module

        pad_on = replace(
            config_module.PAD_SETTINGS,
            enabled=True,
            mode="passive_only",
            passive_model_id="mock_always_live",
            min_frames_passive=1,
        )
        monkeypatch.setattr(config_module, "PAD_SETTINGS", pad_on)
        monkeypatch.setattr(recognize_module, "PAD_SETTINGS", pad_on)

        expected_box = (15, 25, 35, 45)
        monkeypatch.setattr("src.recognize.detect_single_face", lambda f: (expected_box, None))

        recorded_pad_frames: list[Any] = []
        real_create_session = recognize_module.create_pad_session

        def spying_create_pad_session(settings: Any = None) -> Any:
            sess = real_create_session(settings)
            orig_add = sess.add_frame

            def spying_add_frame(pad_frame: Any) -> None:
                recorded_pad_frames.append(pad_frame)
                orig_add(pad_frame)

            sess.add_frame = spying_add_frame
            return sess

        monkeypatch.setattr("src.recognize.create_pad_session", spying_create_pad_session)

        recognize_live(frame_skip=1, cooldown_seconds=0.0)

        assert len(recorded_pad_frames) > 0
        assert all(pf.face_box == expected_box for pf in recorded_pad_frames)

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
        import src.pad_engine as pad_engine_module
        import src.recognize as recognize_module

        pad_on = replace(
            config_module.PAD_SETTINGS,
            enabled=True,
            mode="passive_only",
            passive_model_id="mock_always_attack",
            min_frames_passive=2,
        )
        monkeypatch.setattr(config_module, "PAD_SETTINGS", pad_on)
        monkeypatch.setattr(pad_engine_module, "PAD_SETTINGS", pad_on)
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
        import src.pad_engine as pad_engine_module
        import src.recognize as recognize_module

        pad_on = replace(
            config_module.PAD_SETTINGS,
            enabled=True,
            mode="passive_only",
            passive_model_id="mock_always_live",
            min_frames_passive=2,
        )
        monkeypatch.setattr(config_module, "PAD_SETTINGS", pad_on)
        monkeypatch.setattr(pad_engine_module, "PAD_SETTINGS", pad_on)
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

    def test_stateless_single_frame_call_denies_access_under_default_pad(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bright_frame: np.ndarray,
        trusted_records_alice: list[dict[str, Any]],
    ) -> None:
        """
        Under default PAD settings (min_frames_passive=8), a single-image call to
        recognize_frame without a pad_session falls back to check_single_image_pad.
        Because a single frame cannot provide temporal evidence, it yields INCONCLUSIVE
        ('too_few_frames') and must deny access even if the embedding is an exact match.
        """
        from src.config import PADSettings
        import src.config as config_module
        import src.pad_engine as pad_engine_module
        import src.recognize as recognize_module

        # Ensure default PAD settings (enabled=True, min_frames_passive=8) are active
        default_pad = PADSettings()
        monkeypatch.setattr(config_module, "PAD_SETTINGS", default_pad)
        monkeypatch.setattr(pad_engine_module, "PAD_SETTINGS", default_pad)
        monkeypatch.setattr(recognize_module, "PAD_SETTINGS", default_pad)

        mock_generate = MagicMock(return_value=EMBEDDING_ALICE)
        monkeypatch.setattr("src.recognize.generate_embedding", mock_generate)

        result = recognize_frame(
            frame=bright_frame,
            trusted_records=trusted_records_alice,
            threshold=0.30,
            pad_session=None,
        )

        mock_generate.assert_not_called()
        assert result["access_granted"] is False
        assert result["liveness_passed"] is False
        assert result["pad_outcome"] == "INCONCLUSIVE"
        assert result["pad_reason_code"] == "too_few_frames"


@pytest.mark.integration
class TestLiveFaceBoxAndExplicitStates:
    """Targeted integration tests for face bounding box propagation and explicit UI states."""

    def test_owner_matching_within_threshold_sets_verified_state(
        self,
        trusted_records_alice: list[dict[str, Any]],
        bright_frame: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Enrolled owner matching <= 0.30 yields state=VERIFIED and access_granted=True."""
        from src.recognize import recognize_identity

        # Exact match (distance 0.0)
        monkeypatch.setattr(
            "src.recognize.generate_embedding",
            lambda **kwargs: EMBEDDING_ALICE,
        )

        result = recognize_identity(
            frame=bright_frame,
            trusted_records=trusted_records_alice,
            threshold=0.30,
        )

        assert result["access_granted"] is True
        assert result["state"] == "VERIFIED"
        assert result["primary_text"] == "ACCESS GRANTED"
        assert result["user_name"] == "alice"
        assert result["cosine_distance"] <= 0.30
        assert result["liveness_passed"] is True

    def test_non_owner_rejection_sets_not_verified_state(
        self,
        trusted_records_alice: list[dict[str, Any]],
        bright_frame: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-matching face (> 0.30) yields state=NOT VERIFIED and access_granted=False."""
        from src.recognize import recognize_identity

        # Far candidate (distance ~ 1.0)
        far_embedding = [-x for x in EMBEDDING_ALICE]
        monkeypatch.setattr(
            "src.recognize.generate_embedding",
            lambda **kwargs: far_embedding,
        )

        result = recognize_identity(
            frame=bright_frame,
            trusted_records=trusted_records_alice,
            threshold=0.30,
        )

        assert result["access_granted"] is False
        assert result["state"] == "NOT VERIFIED"
        assert result["primary_text"] == "ACCESS DENIED"
        assert "not recognized" in result["secondary_text"].lower() or "identity mismatch" in result["secondary_text"].lower()

    def test_face_box_passed_to_draw_status_overlay_in_recognize_live(
        self,
        bright_frame: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Live recognition loop passes detected face_box into draw_status_overlay."""
        from src.pad_engine import PadOutcome, PadResult
        import src.recognize as rec

        test_box = (25, 30, 80, 80)
        captured_overlay_kwargs: list[dict[str, Any]] = []

        def mock_detect(frame: Any) -> tuple[tuple[int, int, int, int] | None, str | None]:
            return test_box, None

        def mock_overlay(frame: Any, status: dict[str, Any], frame_skip: int, cooldown_seconds: float, face_box: Any = None) -> None:
            captured_overlay_kwargs.append({"face_box": face_box, "status": status})

        # Quit after 1 frame
        key_sequence = iter([ord("q")])

        monkeypatch.setattr(rec, "open_webcam", lambda idx: MagicMock())
        monkeypatch.setattr(rec, "read_frame", lambda cap: bright_frame)
        monkeypatch.setattr(rec, "detect_single_face", mock_detect)
        monkeypatch.setattr(rec, "draw_status_overlay", mock_overlay)
        monkeypatch.setattr(rec, "show_frame", lambda win, f: None)
        monkeypatch.setattr(rec, "read_display_key", lambda: next(key_sequence))
        monkeypatch.setattr(rec, "release_webcam", lambda cap: None)
        monkeypatch.setattr(rec, "close_windows", lambda: None)
        monkeypatch.setattr(rec, "load_trusted_records", lambda: [{"user_name": "owner", "embeddings": [[0.1] * 512]}])

        rec.recognize_live(camera_index=0, frame_skip=1, cooldown_seconds=0.0)

        assert len(captured_overlay_kwargs) > 0
        assert captured_overlay_kwargs[0]["face_box"] == test_box


@pytest.mark.integration
class TestRecognizeLiveRecenter:
    """
    Focused integration tests for Fix 1:
    Verifying that active challenge completion defers identity extraction,
    displays 'LIVENESS VERIFIED — LOOK AT CAMERA', waits for a newly detected,
    single, centered face, and handles timeout and transient no-face frames properly.
    """

    def test_recognize_live_active_passed_defers_identity_until_recentered(
        self,
        bright_frame: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dataclasses import replace
        import src.config as config_module
        from src.pad_engine import PadOutcome, PadResult, REASON_ACTIVE_PASSED
        import src.recognize as rec

        # Explicitly enable PAD for this test (overriding the autouse disabled fixture)
        pad_on = replace(
            config_module.PAD_SETTINGS,
            enabled=True,
            mode="active_required",
        )
        monkeypatch.setattr(config_module, "PAD_SETTINGS", pad_on)
        monkeypatch.setattr(rec, "PAD_SETTINGS", pad_on)

        # Frame sequence:
        # Frame 1: Challenge completion frame (head turned: cx=20 vs start=32, delta=12)
        # Frame 2: Centered frame (cx=32, start=32, delta=0)
        turned_box = (10, 14, 20, 20)    # cx = 20
        centered_box = (22, 14, 20, 20)  # cx = 32

        boxes = [turned_box, centered_box]
        box_idx = 0

        def mock_detect(frame: Any) -> tuple[tuple[int, int, int, int] | None, str | None]:
            nonlocal box_idx
            box = boxes[min(box_idx, len(boxes) - 1)]
            box_idx += 1
            return box, None

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

        mock_session = MagicMock()
        mock_session.session_id = "test-session-123"
        mock_session.evaluate_final.return_value = active_passed_result
        mock_session.active_start_centroid_x.return_value = 32.0
        mock_session.active_max_delta_x.return_value = 15.0

        captured_statuses: list[dict[str, Any]] = []
        captured_overlay_boxes: list[Any] = []

        def mock_overlay(frame: Any, status: dict[str, Any], frame_skip: int, cooldown_seconds: float, face_box: Any = None) -> None:
            captured_statuses.append(dict(status))
            captured_overlay_boxes.append(face_box)

        embedding_calls_on_frame: list[int] = []
        frame_read_count = 0

        def fake_read_frame(cap: Any) -> np.ndarray:
            nonlocal frame_read_count
            frame_read_count += 1
            return bright_frame

        def mock_generate(**kwargs: Any) -> list[float]:
            embedding_calls_on_frame.append(frame_read_count)
            return [0.1] * 512

        # Quit after 2 frames
        key_sequence = iter([0, ord("q")])

        monkeypatch.setattr(rec, "open_webcam", lambda idx: MagicMock())
        monkeypatch.setattr(rec, "read_frame", fake_read_frame)
        monkeypatch.setattr(rec, "detect_single_face", mock_detect)
        monkeypatch.setattr(rec, "draw_status_overlay", mock_overlay)
        monkeypatch.setattr(rec, "show_frame", lambda win, f: None)
        monkeypatch.setattr(rec, "read_display_key", lambda: next(key_sequence))
        monkeypatch.setattr(rec, "release_webcam", lambda cap: None)
        monkeypatch.setattr(rec, "close_windows", lambda: None)
        monkeypatch.setattr(rec, "create_pad_session", lambda settings: mock_session)
        monkeypatch.setattr(rec, "generate_embedding", mock_generate)
        monkeypatch.setattr(rec, "load_trusted_records", lambda: [{"user_name": "owner", "embeddings": [[0.1] * 512]}])

        result = rec.recognize_live(camera_index=0, frame_skip=1, cooldown_seconds=0.0)

        # 1. On Frame 1 (challenge completion frame):
        # Must show LIVENESS VERIFIED — LOOK AT CAMERA and NOT call embedding extraction!
        assert len(captured_statuses) >= 2
        status_frame_1 = captured_statuses[0]
        assert status_frame_1["state"] == "LIVENESS VERIFIED — LOOK AT CAMERA"
        assert status_frame_1["primary_text"] == "LIVENESS VERIFIED — LOOK AT CAMERA"
        assert status_frame_1["liveness_passed"] is True
        assert status_frame_1["access_granted"] is False

        # 2. Embedding extraction was NOT called on frame 1:
        assert 1 not in embedding_calls_on_frame

        # 3. On Frame 2 (newly detected centered frame):
        # Embedding extraction was called on frame 2 and access granted!
        assert embedding_calls_on_frame == [2]
        assert result["access_granted"] is True
        assert result["user_name"] == "owner"

        # 4. Face box was tracked throughout both frames
        assert captured_overlay_boxes[0] == turned_box
        assert captured_overlay_boxes[1] == centered_box

    def test_recognize_live_recenter_timeout_resets_and_retries(
        self,
        bright_frame: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dataclasses import replace
        import src.config as config_module
        from src.pad_engine import PadOutcome, PadResult, REASON_ACTIVE_PASSED
        import src.recognize as rec

        # Explicitly enable PAD for this test
        pad_on = replace(
            config_module.PAD_SETTINGS,
            enabled=True,
            mode="active_required",
        )
        monkeypatch.setattr(config_module, "PAD_SETTINGS", pad_on)
        monkeypatch.setattr(rec, "PAD_SETTINGS", pad_on)

        # Face stays turned: cx = 20 vs start = 32
        turned_box = (10, 14, 20, 20)

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

        mock_session = MagicMock()
        mock_session.session_id = "test-session-timeout"
        mock_session.evaluate_final.return_value = active_passed_result
        mock_session.active_start_centroid_x.return_value = 32.0
        mock_session.active_max_delta_x.return_value = 15.0

        mock_generate = MagicMock()
        clock = {"now": 0.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

        def advance_clock_on_read(cap: Any) -> np.ndarray:
            # Advance clock past recenter timeout (3.0s) on frame 2
            if clock["now"] == 0.0:
                clock["now"] = 0.5
            else:
                clock["now"] = 5.0
            return bright_frame

        key_sequence = iter([0, ord("q")])

        monkeypatch.setattr(rec, "open_webcam", lambda idx: MagicMock())
        monkeypatch.setattr(rec, "read_frame", advance_clock_on_read)
        monkeypatch.setattr(rec, "detect_single_face", lambda f: (turned_box, None))
        monkeypatch.setattr(rec, "draw_status_overlay", lambda *args, **kwargs: None)
        monkeypatch.setattr(rec, "show_frame", lambda win, f: None)
        monkeypatch.setattr(rec, "read_display_key", lambda: next(key_sequence))
        monkeypatch.setattr(rec, "release_webcam", lambda cap: None)
        monkeypatch.setattr(rec, "close_windows", lambda: None)
        monkeypatch.setattr(rec, "create_pad_session", lambda settings: mock_session)
        monkeypatch.setattr(rec, "generate_embedding", mock_generate)
        monkeypatch.setattr(rec, "load_trusted_records", lambda: [{"user_name": "owner", "embeddings": [[0.1] * 512]}])

        result = rec.recognize_live(camera_index=0, frame_skip=1, cooldown_seconds=0.0)

        # Embedding must never have been called!
        mock_generate.assert_not_called()
        assert result["access_granted"] is False
        assert result["state"] == "TRY AGAIN"
        assert result["secondary_text"] == "Re-centering timed out"
        assert mock_session.reset.called

    def test_recognize_live_recenter_handles_transient_no_face(
        self,
        bright_frame: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dataclasses import replace
        import src.config as config_module
        from src.pad_engine import PadOutcome, PadResult, REASON_ACTIVE_PASSED
        import src.recognize as rec

        # Explicitly enable PAD for this test
        pad_on = replace(
            config_module.PAD_SETTINGS,
            enabled=True,
            mode="active_required",
        )
        monkeypatch.setattr(config_module, "PAD_SETTINGS", pad_on)
        monkeypatch.setattr(rec, "PAD_SETTINGS", pad_on)

        # Frame 1: Challenge completion frame (turned)
        # Frame 2: Transient face detection loss (None, "no_face")
        # Frame 3: Centered face detected (cx = 32)
        detections = [
            ((10, 14, 20, 20), None),
            (None, "no_face"),
            ((22, 14, 20, 20), None),
        ]
        det_idx = 0

        def mock_detect(frame: Any) -> tuple[tuple[int, int, int, int] | None, str | None]:
            nonlocal det_idx
            res = detections[min(det_idx, len(detections) - 1)]
            det_idx += 1
            return res

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

        mock_session = MagicMock()
        mock_session.session_id = "test-session-noface"
        mock_session.evaluate_final.return_value = active_passed_result
        mock_session.active_start_centroid_x.return_value = 32.0
        mock_session.active_max_delta_x.return_value = 15.0

        captured_statuses: list[dict[str, Any]] = []

        def mock_overlay(frame: Any, status: dict[str, Any], frame_skip: int, cooldown_seconds: float, face_box: Any = None) -> None:
            captured_statuses.append(dict(status))

        # Quit after 3 frames
        key_sequence = iter([0, 0, ord("q")])

        monkeypatch.setattr(rec, "open_webcam", lambda idx: MagicMock())
        monkeypatch.setattr(rec, "read_frame", lambda cap: bright_frame)
        monkeypatch.setattr(rec, "detect_single_face", mock_detect)
        monkeypatch.setattr(rec, "draw_status_overlay", mock_overlay)
        monkeypatch.setattr(rec, "show_frame", lambda win, f: None)
        monkeypatch.setattr(rec, "read_display_key", lambda: next(key_sequence))
        monkeypatch.setattr(rec, "release_webcam", lambda cap: None)
        monkeypatch.setattr(rec, "close_windows", lambda: None)
        monkeypatch.setattr(rec, "create_pad_session", lambda settings: mock_session)
        monkeypatch.setattr(rec, "generate_embedding", lambda **kw: [0.1] * 512)
        monkeypatch.setattr(rec, "load_trusted_records", lambda: [{"user_name": "owner", "embeddings": [[0.1] * 512]}])

        result = rec.recognize_live(camera_index=0, frame_skip=1, cooldown_seconds=0.0)

        assert len(captured_statuses) >= 3
        # Frame 2 maintained the re-centering prompt with no-face detail
        assert captured_statuses[1]["state"] == "LIVENESS VERIFIED — LOOK AT CAMERA"
        assert captured_statuses[1]["secondary_text"] == "Center face in front of camera"

        # Frame 3 successfully verified
        assert result["access_granted"] is True
        assert result["user_name"] == "owner"
