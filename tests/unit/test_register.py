"""
Unit tests for src.register — enrollment-path coverage.

Strategy
--------
- cv2, open_webcam, read_frame, save_frame, imshow, waitKey: all mocked.
- generate_embedding / save_user_embeddings: patched at the src.register call-site.
- check_single_image_pad: patched to return LIVE or ATTACK as needed.
- Filesystem: uses tmp_path; DATA_DIR is re-pointed via monkeypatch so no real
  data/ or encodings/ are created.
- No DeepFace downloads, no real webcam, runs in milliseconds.

Scenarios
---------
1.  Successful registration (happy path)
2.  Duplicate user (user folder already exists — still succeeds, overwrite allowed)
3.  Camera unavailable (open_webcam raises RuntimeError)
4.  Capture failure (read_frame raises RuntimeError mid-loop)
5.  PAD failure on every captured frame (all frames rejected -> not enough embeddings)
6.  Embedding failure (generate_embedding raises for every image)
7.  User cancellation (pressing 'q' during capture)
8.  Save failure (save_user_embeddings raises IOError)
9.  Invalid / empty user name supplied as argument
10. Too few valid embeddings after partial failures
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import cv2
import numpy as np
import pytest

from src.pad_engine import (
    ChallengeState,
    ChallengeType,
    PadChallenge,
    PadOutcome,
    PadResult,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _fake_frame() -> np.ndarray:
    """Return a minimal 8x8 BGR NumPy array — stands in for a webcam frame."""
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _make_pad_result(outcome: PadOutcome) -> PadResult:
    """Build a minimal PadResult with the given outcome."""
    return PadResult(
        outcome=outcome,
        reason_code="test_reason",
        user_message_key="test_msg",
        model_id="mock",
        model_version="0.0",
        frames_analyzed=1,
        latency_ms=0.0,
    )


LIVE_PAD_RESULT = _make_pad_result(PadOutcome.LIVE)
ATTACK_PAD_RESULT = _make_pad_result(PadOutcome.ATTACK)

# Fake embedding vector (must be non-empty list of finite floats, 512 dimensions)
FAKE_EMBEDDING: list[float] = [0.1] * 512

# Fake encoding path returned by save_user_embeddings
FAKE_ENCODING_PATH = Path("/fake/encodings/alice.pkl")


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Redirect DATA_DIR used by register.get_user_data_folder to a temp location
    so no real data/ directories are created during tests.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr("src.register.DATA_DIR", data_dir)
    return data_dir


# ---------------------------------------------------------------------------
# Context manager that wires up all the low-level I/O mocks at once.
# The caller controls how many 's' key-presses to simulate before the
# capture loop either finishes or cancels.
# ---------------------------------------------------------------------------

def _patch_capture(
    *,
    key_sequence: list[int],
    save_frame_side_effect: Any = None,
    open_webcam_side_effect: Any = None,
    read_frame_side_effect: Any = None,
    pad_outcome: PadOutcome = PadOutcome.LIVE,
    image_count: int = 3,
    mock_pad_session: Any = None,
):
    """
    Return a context manager (via patch.multiple / nested patch) that stubs:
      - src.register.open_webcam
      - src.register.read_frame
      - src.register.save_frame
      - src.register.cv2.imshow
      - src.register.cv2.waitKey  (cycles through key_sequence then returns 0)
      - src.register.release_webcam
      - src.register.close_windows
      - src.register.create_pad_session
    """
    key_iter = iter(key_sequence + [0] * 9999)

    mock_cap = MagicMock(name="cap")

    def _open_webcam(index):
        if open_webcam_side_effect:
            raise open_webcam_side_effect
        return mock_cap

    def _read_frame(cap):
        if read_frame_side_effect:
            raise read_frame_side_effect
        return _fake_frame()

    def _save_frame(**kwargs):
        if save_frame_side_effect:
            raise save_frame_side_effect
        return Path("/fake/data/alice/alice_001.jpg")

    def _wait_key(delay):
        return next(key_iter)

    if mock_pad_session is None:
        mock_pad_session = MagicMock(name="pad_session")
        mock_pad_session.needs_more_frames.return_value = False
        mock_pad_session.evaluate_final.return_value = _make_pad_result(pad_outcome)

    return (
        patch("src.register.open_webcam", side_effect=_open_webcam),
        patch("src.register.read_frame", side_effect=_read_frame),
        patch("src.register.save_frame", side_effect=_save_frame),
        patch("src.register.cv2.imshow"),
        patch("src.register.cv2.waitKey", side_effect=_wait_key),
        patch("src.register.release_webcam"),
        patch("src.register.close_windows"),
        patch(
            "src.register.create_pad_session",
            return_value=mock_pad_session,
        ),
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRegisterUser:
    """End-to-end tests for src.register.register_user."""

    # ------------------------------------------------------------------
    # 1. Successful registration
    # ------------------------------------------------------------------

    def test_successful_registration(self, tmp_data_dir: Path) -> None:
        """Happy path: 3 captures, all embeddings succeed, save completes."""
        # 's' pressed 3 times to capture the required images
        key_seq = [ord("s")] * 3
        patches = _patch_capture(key_sequence=key_seq, image_count=3)

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import register_user

            result = register_user(user_name="alice", image_count=3)

        assert result["success"] is True
        assert result["user_name"] == "alice"
        assert result["embedding_count"] == 3
        assert result["image_count"] == 3
        assert result["encoding_path"] == FAKE_ENCODING_PATH

    # ------------------------------------------------------------------
    # 2. Duplicate user (overwrite allowed — just re-runs registration)
    # ------------------------------------------------------------------

    def test_duplicate_user_overwrites_existing(self, tmp_data_dir: Path) -> None:
        """
        If the user folder already exists, registration should still succeed.
        The system does not block re-registration for the same name.
        """
        # Pre-create the user folder to simulate an existing registration
        existing_user_dir = tmp_data_dir / "bob"
        existing_user_dir.mkdir()

        key_seq = [ord("s")] * 3
        patches = _patch_capture(key_sequence=key_seq, image_count=3)

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import register_user

            result = register_user(user_name="bob", image_count=3)

        assert result["success"] is True
        assert result["user_name"] == "bob"

    # ------------------------------------------------------------------
    # 3. Camera unavailable
    # ------------------------------------------------------------------

    def test_camera_unavailable_returns_cancelled(self, tmp_data_dir: Path) -> None:
        """open_webcam raises RuntimeError -> registration is cancelled."""
        patches = _patch_capture(
            key_sequence=[],
            open_webcam_side_effect=RuntimeError("Could not open webcam with index 0."),
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import register_user

            result = register_user(user_name="alice", image_count=3)

        assert result["success"] is False
        assert result["reason"] == "cancelled"

    # ------------------------------------------------------------------
    # 4. Capture failure (read_frame raises mid-loop)
    # ------------------------------------------------------------------

    def test_capture_failure_returns_cancelled(self, tmp_data_dir: Path) -> None:
        """read_frame raises RuntimeError -> capture is aborted and cancelled."""
        patches = _patch_capture(
            key_sequence=[],
            read_frame_side_effect=RuntimeError("Could not read frame from webcam."),
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import register_user

            result = register_user(user_name="alice", image_count=3)

        assert result["success"] is False
        assert result["reason"] == "cancelled"

    # ------------------------------------------------------------------
    # 5. PAD failure — every frame rejected by liveness check
    # ------------------------------------------------------------------

    def test_pad_failure_on_all_frames_returns_not_enough_embeddings(
        self, tmp_data_dir: Path
    ) -> None:
        """
        When PAD reports ATTACK on every 's' keypress, no images are saved.
        The loop runs out of 's' presses and we return 'not_enough_valid_embeddings'
        because no images were captured that could produce embeddings.

        Note: capture_registration_images returns an empty list (no images saved)
        because PAD keeps rejecting frames. register_user then sees
        image_count == 0 and returns 'no_images', which is still a failure.
        """
        # Press 's' many times — all will be rejected by PAD
        key_seq = [ord("s")] * 20 + [ord("q")]
        patches = _patch_capture(
            key_sequence=key_seq,
            pad_outcome=PadOutcome.ATTACK,
            image_count=3,
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import register_user

            result = register_user(user_name="alice", image_count=3)

        assert result["success"] is False
        # Either cancelled (via 'q') or no images / not enough embeddings
        assert result["reason"] in {"cancelled", "no_images", "not_enough_valid_embeddings"}

    # ------------------------------------------------------------------
    # 6. Embedding failure (generate_embedding raises for every image)
    # ------------------------------------------------------------------

    def test_embedding_failure_returns_not_enough_valid_embeddings(
        self, tmp_data_dir: Path
    ) -> None:
        """
        All captured images fail to produce embeddings -> not_enough_valid_embeddings.
        """
        key_seq = [ord("s")] * 3
        patches = _patch_capture(key_sequence=key_seq, image_count=3)

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch(
                "src.register.generate_embedding",
                side_effect=ValueError("No face detected"),
            ),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import register_user

            result = register_user(user_name="alice", image_count=3)

        assert result["success"] is False
        assert result["reason"] == "not_enough_valid_embeddings"
        assert result["embedding_count"] == 0

    # ------------------------------------------------------------------
    # 7. User cancellation (pressing 'q')
    # ------------------------------------------------------------------

    def test_user_cancellation_via_q_key(self, tmp_data_dir: Path) -> None:
        """User presses 'q' before capturing enough images -> cancelled."""
        # Save one image then quit
        key_seq = [ord("s"), ord("q")]
        patches = _patch_capture(key_sequence=key_seq, image_count=3)

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import register_user

            result = register_user(user_name="alice", image_count=3)

        assert result["success"] is False
        assert result["reason"] == "cancelled"

    def test_user_cancellation_with_zero_images(self, tmp_data_dir: Path) -> None:
        """User presses 'q' immediately without saving any images -> cancelled."""
        key_seq = [ord("q")]
        patches = _patch_capture(key_sequence=key_seq, image_count=3)

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import register_user

            result = register_user(user_name="alice", image_count=3)

        assert result["success"] is False
        assert result["reason"] == "cancelled"
        assert result["image_count"] == 0

    # ------------------------------------------------------------------
    # 8. Save failure (save_user_embeddings raises IOError)
    # ------------------------------------------------------------------

    def test_save_failure_propagates_exception(self, tmp_data_dir: Path) -> None:
        """
        If save_user_embeddings raises, register_user should let the exception
        propagate (it is not caught inside register_user itself).
        app.py's run_safely wrapper handles top-level exceptions.
        """
        key_seq = [ord("s")] * 3
        patches = _patch_capture(key_sequence=key_seq, image_count=3)

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch(
                "src.register.save_user_embeddings",
                side_effect=IOError("Disk full"),
            ),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import register_user

            with pytest.raises(IOError, match="Disk full"):
                register_user(user_name="alice", image_count=3)

    # ------------------------------------------------------------------
    # 9. Invalid / empty user name
    # ------------------------------------------------------------------

    def test_empty_string_user_name_returns_invalid_user_name(
        self, tmp_data_dir: Path
    ) -> None:
        """Passing an empty string directly should return invalid_user_name."""
        with patch("src.register.ensure_project_folders"):
            from src.register import register_user

            result = register_user(user_name="   ")

        assert result["success"] is False
        assert result["reason"] == "invalid_user_name"

    def test_symbol_only_user_name_returns_invalid_user_name(
        self, tmp_data_dir: Path
    ) -> None:
        """
        A name with only special characters produces clean_name == 'unknown'
        and should be rejected as an invalid user name.
        """
        with patch("src.register.ensure_project_folders"):
            from src.register import register_user

            result = register_user(user_name="@@@###")

        assert result["success"] is False
        assert result["reason"] == "invalid_user_name"

    # ------------------------------------------------------------------
    # 10. Partial embedding failures — too few valid embeddings
    # ------------------------------------------------------------------

    def test_partial_embedding_failures_below_minimum(
        self, tmp_data_dir: Path
    ) -> None:
        """
        3 images captured but only 1 embedding succeeds (< MIN_VALID_EMBEDDINGS=3).
        Registration should be rejected as not_enough_valid_embeddings.
        """
        call_count = {"n": 0}

        def _flaky_embedding(image_input, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return FAKE_EMBEDDING
            raise ValueError("No face detected")

        key_seq = [ord("s")] * 3
        patches = _patch_capture(key_sequence=key_seq, image_count=3)

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", side_effect=_flaky_embedding),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import register_user

            result = register_user(user_name="alice", image_count=3)

        assert result["success"] is False
        assert result["reason"] == "not_enough_valid_embeddings"
        assert result["embedding_count"] == 1

    def test_check_single_image_pad_signature(self) -> None:
        """
        Regression test: check_single_image_pad must be invoked with image=frame,
        never frame=frame. An invalid keyword raises TypeError.
        """
        from src.pad_engine import check_single_image_pad

        frame = _fake_frame()
        result = check_single_image_pad(image=frame)
        assert result.outcome == PadOutcome.INCONCLUSIVE

        with pytest.raises(TypeError):
            check_single_image_pad(frame=frame)  # type: ignore[call-arg]

    def test_registration_accumulates_configured_frames_before_pad_eval(
        self, tmp_data_dir: Path
    ) -> None:
        """
        Registration must accumulate the configured number of genuine frames before evaluating PAD.
        When user presses 's' before sufficient frames exist, the loop must continue reading
        genuine frames until pad_session.needs_more_frames() is False before evaluating.
        """
        read_frames: list[np.ndarray] = []

        def _custom_read(cap):
            f = _fake_frame()
            read_frames.append(f)
            return f

        mock_session = MagicMock(name="pad_session")
        # needs_more_frames: True on initial 's' check, True on 1st extra frame, False on 2nd
        mock_session.needs_more_frames.side_effect = [True, True, False]
        mock_session.evaluate_final.return_value = LIVE_PAD_RESULT

        key_seq = [ord("s")]
        patches = _patch_capture(
            key_sequence=key_seq,
            image_count=1,
            mock_pad_session=mock_session,
        )

        with (
            patches[0],
            patch("src.register.read_frame", side_effect=_custom_read),
            patches[2], patches[3], patches[4], patches[5], patches[6],
            patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import capture_registration_images

            captured, cancelled = capture_registration_images("alice", image_count=1)

        assert cancelled is False
        assert len(captured) == 1
        # At least 3 genuine frame reads: initial frame + 2 extra frames to satisfy min_frames
        assert len(read_frames) >= 3
        assert mock_session.add_frame.call_count == len(read_frames)
        mock_session.evaluate_final.assert_called_once()
        mock_session.reset.assert_called_once()

    def test_registration_rejects_inconclusive_pad_and_resets_session(
        self, tmp_data_dir: Path
    ) -> None:
        """
        An INCONCLUSIVE PAD result (e.g. too_few_frames or insufficient motion)
        must reject the capture attempt, not save any frame, and reset the PAD session.
        """
        inconclusive_result = _make_pad_result(PadOutcome.INCONCLUSIVE)
        mock_session = MagicMock(name="pad_session")
        mock_session.needs_more_frames.return_value = False
        mock_session.evaluate_final.return_value = inconclusive_result

        # Press 's' (rejected as inconclusive), then 'q' (cancel)
        key_seq = [ord("s"), ord("q")]
        patches = _patch_capture(
            key_sequence=key_seq,
            image_count=1,
            mock_pad_session=mock_session,
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import capture_registration_images

            captured, cancelled = capture_registration_images("alice", image_count=1)

        assert cancelled is True
        assert len(captured) == 0
        mock_session.reset.assert_called_once()

    def test_registration_resets_pad_session_between_consecutive_captures(
        self, tmp_data_dir: Path
    ) -> None:
        """
        PAD session must be reset between capture attempts so temporal frames
        from one attempt do not leak into subsequent attempts.
        """
        mock_session = MagicMock(name="pad_session")
        mock_session.needs_more_frames.return_value = False
        # Attempt 1: ATTACK -> rejected & reset. Attempt 2: LIVE -> accepted & reset.
        mock_session.evaluate_final.side_effect = [ATTACK_PAD_RESULT, LIVE_PAD_RESULT]

        key_seq = [ord("s"), ord("s")]
        patches = _patch_capture(
            key_sequence=key_seq,
            image_count=1,
            mock_pad_session=mock_session,
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import capture_registration_images

            captured, cancelled = capture_registration_images("alice", image_count=1)

        assert cancelled is False
        assert len(captured) == 1
        assert mock_session.reset.call_count == 2

    def test_registration_cancellation_during_frame_accumulation(
        self, tmp_data_dir: Path
    ) -> None:
        """
        If user presses 'q' while extra frames are being gathered, registration
        must cancel cleanly and immediately release resources.
        """
        mock_session = MagicMock(name="pad_session")
        # Simulates needs_more_frames staying True until interrupted
        mock_session.needs_more_frames.return_value = True

        # First key is 's' to trigger save, but during accumulation 'q' is pressed
        key_seq = [ord("s"), ord("q")]
        patches = _patch_capture(
            key_sequence=key_seq,
            image_count=1,
            mock_pad_session=mock_session,
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import capture_registration_images

            captured, cancelled = capture_registration_images("alice", image_count=1)

        assert cancelled is True
        assert len(captured) == 0


# ---------------------------------------------------------------------------
# Isolated unit tests for helper functions (higher granularity coverage)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAskUserName:
    """Tests for src.register.ask_user_name()."""

    def test_valid_name_is_returned(self) -> None:
        with patch("builtins.input", return_value="alice"):
            from src.register import ask_user_name

            name = ask_user_name()
        assert name == "alice"

    def test_empty_input_loops_until_valid(self) -> None:
        # First two inputs are empty/invalid, third is valid
        with patch("builtins.input", side_effect=["", "   ", "bob"]):
            from src.register import ask_user_name

            name = ask_user_name()
        assert name == "bob"

    def test_symbol_only_input_loops_until_valid(self) -> None:
        with patch("builtins.input", side_effect=["###", "carol"]):
            from src.register import ask_user_name

            name = ask_user_name()
        assert name == "carol"


@pytest.mark.unit
class TestGetUserDataFolder:
    """Tests for src.register.get_user_data_folder()."""

    def test_creates_folder_for_user(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.register.DATA_DIR", tmp_path / "data")

        from src.register import get_user_data_folder

        folder = get_user_data_folder("Alice Smith")
        assert folder.exists()
        # clean_name converts "Alice Smith" -> "alice_smith"
        assert folder.name == "alice_smith"

    def test_safe_name_used_for_folder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.register.DATA_DIR", tmp_path / "data")

        from src.register import get_user_data_folder

        folder = get_user_data_folder("Test User!")
        # Exclamation mark is stripped
        assert "!" not in folder.name


@pytest.mark.unit
class TestGenerateRegistrationEmbeddings:
    """Tests for src.register.generate_registration_embeddings()."""

    def test_all_images_succeed(self, tmp_path: Path) -> None:
        paths = [tmp_path / "img1.jpg", tmp_path / "img2.jpg"]
        # Create the dummy files so Path.name is accessible
        for p in paths:
            p.touch()

        with patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING):
            from src.register import generate_registration_embeddings

            embeddings = generate_registration_embeddings(paths)

        assert len(embeddings) == 2
        assert embeddings[0] == FAKE_EMBEDDING

    def test_bad_images_are_skipped(self, tmp_path: Path) -> None:
        paths = [tmp_path / "img1.jpg", tmp_path / "img2.jpg", tmp_path / "img3.jpg"]
        for p in paths:
            p.touch()

        call_n = {"n": 0}

        def _sometimes_fails(image_input, **kwargs):
            call_n["n"] += 1
            if call_n["n"] == 2:
                raise ValueError("No face")
            return FAKE_EMBEDDING

        with patch("src.register.generate_embedding", side_effect=_sometimes_fails):
            from src.register import generate_registration_embeddings

            embeddings = generate_registration_embeddings(paths)

        # Only 2 of 3 succeeded
        assert len(embeddings) == 2

    def test_empty_paths_returns_empty_list(self) -> None:
        with patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING):
            from src.register import generate_registration_embeddings

            embeddings = generate_registration_embeddings([])

        assert embeddings == []


# ---------------------------------------------------------------------------
# Redesigned Automatic Registration & Quality / Diversity / Security Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAutomaticCaptureWorkflow:
    """Tests for continuous automatic frame capture without manual 's' presses."""

    def test_automatic_capture_without_s_key(self, tmp_data_dir: Path) -> None:
        """Capture completes automatically with zero 's' keypresses."""
        patches = _patch_capture(
            key_sequence=[0] * 50,  # Only idle key checks, no 's'
            image_count=2,
            pad_outcome=PadOutcome.LIVE,
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import capture_registration_images

            captured, cancelled = capture_registration_images("alice", image_count=2)

        assert cancelled is False
        assert len(captured) == 2

    def test_cancellation_during_automatic_capture(self, tmp_data_dir: Path) -> None:
        """User pressing 'q' at any point cancels registration and cleans up."""
        patches = _patch_capture(
            key_sequence=[0, ord("q")],
            image_count=3,
            pad_outcome=PadOutcome.LIVE,
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import capture_registration_images

            captured, cancelled = capture_registration_images("alice", image_count=3)

        assert cancelled is True
        assert len(captured) == 0


@pytest.mark.unit
class TestQualityFiltering:
    """Tests for frame quality filtering (blur, lighting, face position/count)."""

    def test_reject_blurry_frame(self) -> None:
        from src.register import check_frame_quality

        # Create a smooth blurred image (Laplacian variance near 0)
        img = np.full((300, 300, 3), 128, dtype=np.uint8)
        face_box = (50, 50, 100, 100)
        ok, reason = check_frame_quality(img, face_box)
        assert ok is False
        assert reason == "blurry"

    def test_reject_dark_frame(self) -> None:
        from src.register import check_frame_quality

        # Dark image with texture so Laplacian > 40 but brightness < 45
        img = np.random.randint(0, 30, (300, 300, 3), dtype=np.uint8)
        face_box = (50, 50, 100, 100)
        ok, reason = check_frame_quality(img, face_box)
        assert ok is False
        assert reason == "too_dark"

    def test_reject_overexposed_frame(self) -> None:
        from src.register import check_frame_quality

        # Overexposed image (> 245)
        img = np.random.randint(246, 256, (300, 300, 3), dtype=np.uint8)
        face_box = (50, 50, 100, 100)
        ok, reason = check_frame_quality(img, face_box)
        assert ok is False
        assert reason == "too_bright"

    def test_reject_face_too_small(self) -> None:
        from src.register import detect_single_face

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([[10, 10, 20, 20]])
        with patch("src.register.get_face_cascade", return_value=mock_cascade):
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            face_box, error = detect_single_face(img)
        assert face_box is None
        assert error == "face_too_small"

    def test_reject_face_too_close(self) -> None:
        from src.register import detect_single_face

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([[10, 10, 450, 450]])
        with patch("src.register.get_face_cascade", return_value=mock_cascade):
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            face_box, error = detect_single_face(img)
        assert face_box is None
        assert error == "face_too_large"

    def test_detect_single_face_rejects_multiple_faces(self) -> None:
        from src.register import detect_single_face

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([
            [50, 50, 80, 80],
            [200, 50, 80, 80],
        ])

        with patch("src.register.get_face_cascade", return_value=mock_cascade):
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            face_box, error = detect_single_face(img)

        assert face_box is None
        assert error == "multiple_faces"

    def test_detect_single_face_rejects_zero_faces(self) -> None:
        from src.register import detect_single_face

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = ()

        with patch("src.register.get_face_cascade", return_value=mock_cascade):
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            face_box, error = detect_single_face(img)

        assert face_box is None
        assert error == "no_face"


@pytest.mark.unit
class TestDiversityAndDeduplication:
    """Tests for dataset diversity and deduplication."""

    def test_reject_identical_frames(self) -> None:
        from src.register import is_frame_diverse

        frame = np.full((300, 300, 3), 120, dtype=np.uint8)
        face_box = (50, 50, 100, 100)
        crop = cv2.cvtColor(frame[50:150, 50:150], cv2.COLOR_BGR2GRAY)
        last_face = cv2.resize(crop, (100, 100))

        assert is_frame_diverse(frame, face_box, last_face) is False

    def test_accept_diverse_frames(self) -> None:
        from src.register import is_frame_diverse

        frame = np.full((300, 300, 3), 120, dtype=np.uint8)
        face_box = (50, 50, 100, 100)
        last_face = np.full((100, 100), 50, dtype=np.uint8)

        assert is_frame_diverse(frame, face_box, last_face) is True


@pytest.mark.unit
class TestPADAndActiveChallenge:
    """Tests for persistent PAD session and active challenge interaction."""

    def test_persistent_session_passes_face_box_coordinates(self, tmp_data_dir: Path) -> None:
        mock_session = MagicMock(name="pad_session")
        mock_session.needs_more_frames.return_value = False
        mock_session.evaluate_final.return_value = LIVE_PAD_RESULT

        patches = _patch_capture(
            key_sequence=[0] * 5,
            image_count=1,
            mock_pad_session=mock_session,
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import capture_registration_images

            capture_registration_images("alice", image_count=1)

        assert mock_session.add_frame.call_count >= 1
        pad_frame = mock_session.add_frame.call_args[0][0]
        assert hasattr(pad_frame, "face_box")

    def test_active_challenge_transitions_to_live(self, tmp_data_dir: Path) -> None:
        """Inconclusive PAD triggers active challenge; challenge completion yields LIVE."""
        mock_session = MagicMock(name="pad_session")
        mock_session.needs_more_frames.return_value = False

        inconclusive_result = _make_pad_result(PadOutcome.INCONCLUSIVE)
        mock_session.evaluate_final.side_effect = [inconclusive_result, LIVE_PAD_RESULT]

        active_challenge = PadChallenge(
            challenge_id="ch_1",
            challenge_type=ChallengeType.HEAD_TURN_LEFT,
            instruction_key="turn_head_left",
            state=ChallengeState.IN_PROGRESS,
            started_at_ms=100.0,
            deadline_ms=999999.0,
        )
        mock_session.current_challenge.return_value = active_challenge

        patches = _patch_capture(
            key_sequence=[0] * 10,
            image_count=1,
            mock_pad_session=mock_session,
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import capture_registration_images

            captured, cancelled = capture_registration_images("alice", image_count=1)

        assert cancelled is False
        assert len(captured) == 1

    def test_persistent_pad_attack_aborts_registration(self, tmp_data_dir: Path) -> None:
        """Repeated ATTACK outcomes abort registration cleanly and do not save enrollment."""
        mock_session = MagicMock(name="pad_session")
        mock_session.needs_more_frames.return_value = False
        mock_session.evaluate_final.return_value = ATTACK_PAD_RESULT

        patches = _patch_capture(
            key_sequence=[0] * 50,
            image_count=1,
            mock_pad_session=mock_session,
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=FAKE_ENCODING_PATH),
            patch("src.register.ensure_project_folders"),
        ):
            from src.register import capture_registration_images

            captured, cancelled = capture_registration_images("alice", image_count=1)

        assert cancelled is True
        assert len(captured) == 0


@pytest.mark.unit
class TestCleanupAndIdentityConsistency:
    """Tests for directory cleanup on failure/cancellation and username normalization."""

    def test_cleanup_removes_partially_captured_images(self, tmp_path: Path) -> None:
        from src.register import cleanup_registration_directory

        user_dir = tmp_path / "alice"
        user_dir.mkdir()
        img1 = user_dir / "alice_001.jpg"
        img2 = user_dir / "alice_002.jpg"
        img1.touch()
        img2.touch()

        cleanup_registration_directory(user_dir, [img1, img2])
        assert not img1.exists()
        assert not img2.exists()
        assert not user_dir.exists()

    def test_username_normalization_consistency(self) -> None:
        from src.utils import clean_name

        assert clean_name("Owner") == "owner"
        assert clean_name("owner") == "owner"
        assert clean_name("  Owner  ") == "owner"
        assert clean_name("User_Name_123") == "user_name_123"


@pytest.mark.unit
class TestRegisterStrictMode:
    def test_register_user_fails_closed_when_key_missing_in_strict_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """register_user() raises RuntimeError if strict mode is on and signing key is missing."""
        from src.register import register_user
        monkeypatch.setenv("FACE_LOCK_STRICT_MODE", "1")
        monkeypatch.delenv("FACE_LOCK_SIGNING_KEY", raising=False)
        monkeypatch.setenv("FACE_LOCK_OPERATOR_PIN_HASH", "hash123")

        with pytest.raises(RuntimeError, match="Strict mode is enabled"):
            register_user(user_name="alice")

    def test_register_user_fails_closed_when_operator_pin_missing_in_strict_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """register_user() raises RuntimeError if strict mode is on and operator PIN is in demo mode."""
        from src.register import register_user
        monkeypatch.setenv("FACE_LOCK_STRICT_MODE", "1")
        monkeypatch.setenv("FACE_LOCK_SIGNING_KEY", "0123456789abcdef" * 4)
        monkeypatch.delenv("FACE_LOCK_OPERATOR_PIN_HASH", raising=False)

        with pytest.raises(RuntimeError, match="Strict mode is enabled"):
            register_user(user_name="alice")


@pytest.mark.unit
class TestValidateEmbedding:
    """Unit tests for src.register.validate_embedding."""

    def test_valid_512_dim_embedding_accepted(self) -> None:
        from src.register import validate_embedding

        valid_embedding = [0.05] * 512
        assert validate_embedding(valid_embedding, expected_dim=512) is True

    def test_rejects_none(self) -> None:
        from src.register import validate_embedding

        assert validate_embedding(None, expected_dim=512) is False

    def test_rejects_empty(self) -> None:
        from src.register import validate_embedding

        assert validate_embedding([], expected_dim=512) is False

    def test_rejects_wrong_dimensions(self) -> None:
        from src.register import validate_embedding

        assert validate_embedding([0.1] * 128, expected_dim=512) is False
        assert validate_embedding([0.1] * 511, expected_dim=512) is False
        assert validate_embedding([0.1] * 513, expected_dim=512) is False

    def test_rejects_nan_and_inf(self) -> None:
        from src.register import validate_embedding

        embedding_with_nan = [0.1] * 512
        embedding_with_nan[10] = float("nan")
        assert validate_embedding(embedding_with_nan, expected_dim=512) is False

        embedding_with_inf = [0.1] * 512
        embedding_with_inf[50] = float("inf")
        assert validate_embedding(embedding_with_inf, expected_dim=512) is False

    def test_rejects_non_numeric(self) -> None:
        from src.register import validate_embedding

        invalid = ["string"] * 512
        assert validate_embedding(invalid, expected_dim=512) is False


@pytest.mark.unit
class TestOwnerReEnrollment:
    """Unit tests for clean owner re-enrollment and overwrite safeguards."""

    def test_existing_record_blocks_when_overwrite_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.register import register_user

        mock_encodings_dir = tmp_path / "encodings"
        mock_encodings_dir.mkdir()
        monkeypatch.setattr("src.register.get_user_embedding_path", lambda name: mock_encodings_dir / f"{name}.pkl")
        (mock_encodings_dir / "owner.pkl").touch()

        result = register_user(user_name="owner", overwrite=False)
        assert result["success"] is False
        assert result["reason"] == "user_already_exists"

    def test_existing_record_allows_when_overwrite_true_and_purges_old_images(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.register import register_user

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        owner_dir = data_dir / "owner"
        owner_dir.mkdir()
        old_img1 = owner_dir / "owner_old_01.jpg"
        old_img1.write_text("dummy")
        old_img2 = owner_dir / "owner_old_02.png"
        old_img2.write_text("dummy")

        mock_encodings_dir = tmp_path / "encodings"
        mock_encodings_dir.mkdir()
        (mock_encodings_dir / "owner.pkl").touch()

        monkeypatch.setattr("src.register.DATA_DIR", data_dir)
        monkeypatch.setattr("src.register.get_user_embedding_path", lambda name: mock_encodings_dir / f"{name}.pkl")
        monkeypatch.setattr("src.register.ensure_project_folders", lambda: None)

        captured_imgs = [
            owner_dir / "owner_new_01.jpg",
            owner_dir / "owner_new_02.jpg",
            owner_dir / "owner_new_03.jpg",
        ]
        for img in captured_imgs:
            img.write_text("new_img")

        patches = _patch_capture(key_sequence=[ord("s")] * 3, image_count=3)
        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5],
            patches[6], patches[7],
            patch("src.register.capture_registration_images", return_value=(captured_imgs, False)),
            patch("src.register.generate_embedding", return_value=FAKE_EMBEDDING),
            patch("src.register.save_user_embeddings", return_value=mock_encodings_dir / "owner.pkl"),
        ):
            result = register_user(user_name="owner", image_count=3, overwrite=True)

        assert result["success"] is True
        assert result["user_name"] == "owner"
        assert not old_img1.exists()
        assert not old_img2.exists()
