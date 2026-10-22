"""
System-level tests for Face Lock AI.

These tests validate complete end-to-end workflows using mocks for all
hardware and heavy-ML dependencies. No webcam, no DeepFace download, and
no PAD inference required.

Workflows covered
-----------------
1.  Application startup — warmup, folder creation, menu loop
2.  Registration workflow — happy path, auth gate, PAD rejection, cancellation
3.  Recognition workflow — granted, denied, no users, bad lighting, camera error
4.  Operator authentication — demo mode, correct PIN, wrong PIN, abort
5.  Integrity verification — signed save/load round-trip, tamper detection
6.  Signed embeddings — end-to-end save → load → recognize with signed files

Markers
-------
All tests carry @pytest.mark.system for selective skipping in fast CI.
Fast tests also carry @pytest.mark.unit so they run in the normal suite.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_EMBEDDING_ALICE = [1.0, 0.0, 0.0]
_EMBEDDING_STRANGER = [0.0, 1.0, 0.0]
_BRIGHT_FRAME = np.full((48, 64, 3), 180, dtype=np.uint8)
_DARK_FRAME = np.full((48, 64, 3), 5, dtype=np.uint8)
_FAKE_PKL_PATH = Path("/fake/encodings/alice.pkl")
_SIGNING_KEY = "deadbeefdeadbeefdeadbeefdeadbeef0102030405060708"
_PIN = "s3cr3t-pin"


def _hash_pin(pin: str) -> str:
    import hashlib
    return hashlib.sha256(pin.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_encodings(tmp_path: Path) -> Path:
    enc = tmp_path / "encodings"
    enc.mkdir()
    return enc


@pytest.fixture()
def signed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACE_LOCK_SIGNING_KEY", _SIGNING_KEY)


@pytest.fixture()
def unsigned_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FACE_LOCK_SIGNING_KEY", raising=False)


@pytest.fixture()
def pin_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACE_LOCK_OPERATOR_PIN_HASH", _hash_pin(_PIN))


@pytest.fixture()
def no_pin_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FACE_LOCK_OPERATOR_PIN_HASH", raising=False)


@pytest.fixture()
def log_capture(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    face_lock = logging.getLogger("face_lock")
    face_lock.addHandler(caplog.handler)
    face_lock.setLevel(logging.DEBUG)
    yield caplog
    face_lock.removeHandler(caplog.handler)


# ---------------------------------------------------------------------------
# 1. Application startup
# ---------------------------------------------------------------------------

@pytest.mark.system
@pytest.mark.unit
class TestApplicationStartup:
    """Verify run_app() sets up folders, warms up the model, then enters the loop."""

    def test_run_app_creates_project_folders(self, tmp_path: Path) -> None:
        """ensure_project_folders is called exactly once at startup."""
        with (
            patch("app.ensure_project_folders") as mock_folders,
            patch("app.warmup_model"),
            patch("app.show_banner"),
            patch("app.show_menu"),
            patch("app.get_menu_choice", return_value="3"),
            patch("app.print_status"),
        ):
            from app import run_app
            run_app()

        mock_folders.assert_called_once()

    def test_run_app_calls_warmup_once(self) -> None:
        with (
            patch("app.ensure_project_folders"),
            patch("app.warmup_model") as mock_warmup,
            patch("app.show_banner"),
            patch("app.show_menu"),
            patch("app.get_menu_choice", return_value="3"),
            patch("app.print_status"),
        ):
            from app import run_app
            run_app()

        mock_warmup.assert_called_once()

    def test_warmup_failure_does_not_abort_startup(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with (
            patch("app.ensure_project_folders"),
            patch("app.load_face_model", side_effect=RuntimeError("weights missing")),
            patch("app.show_banner"),
            patch("app.show_menu"),
            patch("app.get_menu_choice", return_value="3"),
            patch("app.print_status"),
        ):
            from app import warmup_model
            warmup_model()  # must not raise — warmup_model catches internally

    def test_main_handles_keyboard_interrupt(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with patch("app.run_app", side_effect=KeyboardInterrupt):
            from app import main
            main()

        out = capsys.readouterr().out
        assert "stopped" in out.lower() or "warning" in out.lower()

    def test_exit_choice_stops_loop(self) -> None:
        call_count = {"n": 0}

        def _choice():
            call_count["n"] += 1
            return "3"  # EXIT immediately

        with (
            patch("app.ensure_project_folders"),
            patch("app.warmup_model"),
            patch("app.show_banner"),
            patch("app.show_menu"),
            patch("app.get_menu_choice", side_effect=_choice),
            patch("app.print_status"),
        ):
            from app import run_app
            run_app()

        assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# 2. Registration workflow
# ---------------------------------------------------------------------------

@pytest.mark.system
@pytest.mark.unit
class TestRegistrationWorkflow:
    """End-to-end registration via app.handle_registration()."""

    def _patch_register(self, result: dict[str, Any]):
        return patch("app.register_user", return_value=result)

    def test_successful_registration_prints_success(
        self, capsys: pytest.CaptureFixture, no_pin_env: None
    ) -> None:
        with self._patch_register(
            {"success": True, "user_name": "alice", "embedding_count": 5}
        ):
            from app import handle_registration
            handle_registration()

        out = capsys.readouterr().out
        assert "alice" in out.lower() or "ok" in out.lower()

    def test_registration_blocked_when_auth_fails(self, pin_env: None) -> None:
        with (
            patch("app.verify_operator_pin", return_value=False),
            patch("app.register_user") as mock_reg,
        ):
            from app import handle_registration
            handle_registration()

        mock_reg.assert_not_called()

    def test_registration_proceeds_when_auth_passes(
        self, no_pin_env: None
    ) -> None:
        with self._patch_register(
            {"success": True, "user_name": "alice", "embedding_count": 3}
        ) as mock_reg:
            from app import handle_registration
            handle_registration()

        mock_reg.assert_called_once()

    def test_cancelled_registration_prints_warning(
        self, capsys: pytest.CaptureFixture, no_pin_env: None
    ) -> None:
        with self._patch_register({"success": False, "reason": "cancelled"}):
            from app import handle_registration
            handle_registration()

        out = capsys.readouterr().out
        assert "cancelled" in out.lower() or "warning" in out.lower()

    def test_registration_with_pad_rejection_returns_not_enough(
        self, no_pin_env: None, capsys: pytest.CaptureFixture
    ) -> None:
        with self._patch_register(
            {"success": False, "reason": "not_enough_valid_embeddings",
             "embedding_count": 0, "image_count": 3}
        ):
            from app import handle_registration
            handle_registration()

        out = capsys.readouterr().out
        assert "not_enough" in out.lower() or "warning" in out.lower()


# ---------------------------------------------------------------------------
# 3. Recognition workflow
# ---------------------------------------------------------------------------

@pytest.mark.system
@pytest.mark.unit
class TestRecognitionWorkflow:
    """End-to-end recognition via app.handle_recognition()."""

    def test_access_granted_prints_success(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with patch("app.recognize_live", return_value={
            "access_granted": True,
            "user_name": "alice",
            "secondary_text": "User: alice",
        }):
            from app import handle_recognition
            handle_recognition()

        out = capsys.readouterr().out
        assert "alice" in out.lower() or "granted" in out.lower()

    def test_access_denied_prints_warning(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with patch("app.recognize_live", return_value={
            "access_granted": False,
            "secondary_text": "No registered users",
        }):
            from app import handle_recognition
            handle_recognition()

        out = capsys.readouterr().out
        assert "denied" in out.lower() or "warning" in out.lower() or "no registered" in out.lower()

    def test_no_users_recognition_result(self) -> None:
        from src.recognize import recognize_frame
        result = recognize_frame(
            frame=_BRIGHT_FRAME,
            trusted_records=[],
        )
        assert result["access_granted"] is False
        assert "No registered" in result["secondary_text"]

    def test_bad_lighting_denies_before_embedding(self) -> None:
        mock_gen = MagicMock(return_value=_EMBEDDING_ALICE)
        with patch("src.recognize.generate_embedding", mock_gen):
            from src.recognize import recognize_frame
            result = recognize_frame(
                frame=_DARK_FRAME,
                trusted_records=[{"user_name": "alice", "model_name": "x",
                                   "embeddings": [_EMBEDDING_ALICE]}],
            )
        mock_gen.assert_not_called()
        assert result["access_granted"] is False
        assert result["secondary_text"] == "Bad lighting"

    def test_camera_error_returns_denied(self) -> None:
        from src.recognize import recognize_live

        with (
            patch("src.recognize.load_trusted_records", return_value=[]),
            patch("src.recognize.open_webcam", side_effect=RuntimeError("camera broken")),
            patch("src.recognize.release_webcam"),
            patch("src.recognize.close_windows"),
            patch("src.recognize.show_frame"),
            patch("src.recognize.draw_status_overlay"),
        ):
            result = recognize_live()

        assert result["access_granted"] is False
        # RuntimeError from open_webcam propagates into the except block
        assert "camera" in result.get("secondary_text", "").lower() or \
               result.get("access_granted") is False


# ---------------------------------------------------------------------------
# 4. Operator authentication
# ---------------------------------------------------------------------------

@pytest.mark.system
@pytest.mark.unit
class TestOperatorAuthentication:
    def test_demo_mode_allows_through(
        self, no_pin_env: None, log_capture: pytest.LogCaptureFixture
    ) -> None:
        from src.operator_auth import verify_operator_pin

        with log_capture.at_level(logging.WARNING, logger="face_lock"):
            result = verify_operator_pin()

        assert result is True
        assert any("operator_auth_demo_mode" in r.message for r in log_capture.records)

    def test_correct_pin_grants_access(self, pin_env: None) -> None:
        from src.operator_auth import verify_operator_pin
        with patch("src.operator_auth.getpass.getpass", return_value=_PIN):
            assert verify_operator_pin() is True

    def test_wrong_pin_denies_access(self, pin_env: None) -> None:
        from src.operator_auth import MAX_PIN_ATTEMPTS, verify_operator_pin
        with patch("src.operator_auth.getpass.getpass",
                   side_effect=["wrong"] * MAX_PIN_ATTEMPTS):
            assert verify_operator_pin() is False

    def test_eof_on_pin_input_denies(self, pin_env: None) -> None:
        from src.operator_auth import verify_operator_pin
        with patch("src.operator_auth.getpass.getpass", side_effect=EOFError):
            assert verify_operator_pin() is False

    def test_registration_blocked_after_max_attempts(
        self, pin_env: None
    ) -> None:
        from src.operator_auth import MAX_PIN_ATTEMPTS, verify_operator_pin
        with patch("src.operator_auth.getpass.getpass",
                   side_effect=["bad"] * MAX_PIN_ATTEMPTS) as mock_gp:
            result = verify_operator_pin()
        assert result is False
        assert mock_gp.call_count == MAX_PIN_ATTEMPTS


# ---------------------------------------------------------------------------
# 5. Integrity verification
# ---------------------------------------------------------------------------

@pytest.mark.system
@pytest.mark.unit
class TestIntegrityVerification:
    def test_signed_save_load_round_trip(
        self, tmp_encodings: Path, signed_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.ai_engine as ae
        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_encodings)
        path = tmp_encodings / "alice.pkl"

        ae.save_user_embeddings("alice", [_EMBEDDING_ALICE], output_path=path)
        record = ae.load_user_embeddings("alice", input_path=path)

        assert record["user_name"] == "alice"
        assert record["embeddings"] == [_EMBEDDING_ALICE]

    def test_tampered_file_is_rejected(
        self, tmp_encodings: Path, signed_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.ai_engine as ae
        from src.integrity import IntegrityError, wrap_record_for_storage
        from src.utils import save_pickle

        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_encodings)
        path = tmp_encodings / "tampered.pkl"

        rec = {
            "user_name": "alice", "safe_name": "alice",
            "model_name": "Facenet512", "detector_backend": "opencv",
            "distance_metric": "cosine", "threshold": 0.3,
            "embeddings": [_EMBEDDING_ALICE], "embedding_count": 1,
            "image_paths": [], "created_at": "2026-01-01T00:00:00",
        }
        payload = wrap_record_for_storage(rec)
        payload["record"]["user_name"] = "mallory"
        save_pickle(payload, path)

        with pytest.raises(IntegrityError):
            ae.load_user_embeddings("tampered", input_path=path)

    def test_legacy_file_loads_with_warning(
        self, tmp_encodings: Path, unsigned_env: None,
        monkeypatch: pytest.MonkeyPatch,
        log_capture: pytest.LogCaptureFixture,
    ) -> None:
        import src.ai_engine as ae
        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_encodings)
        path = tmp_encodings / "legacy.pkl"

        legacy = {
            "user_name": "bob", "safe_name": "bob",
            "model_name": "Facenet512", "detector_backend": "opencv",
            "distance_metric": "cosine", "threshold": 0.3,
            "embeddings": [_EMBEDDING_ALICE], "embedding_count": 1,
            "image_paths": [], "created_at": "2026-01-01T00:00:00",
        }
        with path.open("wb") as f:
            pickle.dump(legacy, f)

        with log_capture.at_level(logging.WARNING, logger="face_lock"):
            record = ae.load_user_embeddings("bob", input_path=path)

        assert record["user_name"] == "bob"
        assert any("legacy" in r.message.lower() for r in log_capture.records)

    def test_corrupt_file_skipped_in_load_all(
        self, tmp_encodings: Path, signed_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import src.ai_engine as ae
        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_encodings)

        (tmp_encodings / "bad.pkl").write_bytes(b"not a pickle")
        valid_path = tmp_encodings / "zzz_alice.pkl"
        ae.save_user_embeddings("alice", [_EMBEDDING_ALICE], output_path=valid_path)

        records = ae.load_all_user_embeddings(encodings_dir=tmp_encodings)
        assert len(records) == 1
        assert records[0]["user_name"] == "alice"


# ---------------------------------------------------------------------------
# 6. Signed embeddings — full save → load → find_best_match
# ---------------------------------------------------------------------------

@pytest.mark.system
@pytest.mark.unit
class TestSignedEmbeddingsEndToEnd:
    def test_signed_embeddings_used_in_recognition(
        self, tmp_encodings: Path, signed_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Register alice with signed embeddings, then match in find_best_match."""
        import src.ai_engine as ae
        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_encodings)

        # Save alice's embedding (signed)
        ae.save_user_embeddings(
            "alice", [_EMBEDDING_ALICE],
            output_path=tmp_encodings / "alice.pkl",
        )

        # Load all embeddings (triggers integrity check)
        records = ae.load_all_user_embeddings(encodings_dir=tmp_encodings)
        assert len(records) == 1

        # Match alice's own embedding — should grant
        result = ae.find_best_match(
            candidate_embedding=_EMBEDDING_ALICE,
            known_records=records,
            threshold=0.30,
        )
        assert result["is_match"] is True
        assert result["user_name"] == "alice"

    def test_stranger_does_not_match_alice(
        self, tmp_encodings: Path, signed_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import src.ai_engine as ae
        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_encodings)

        ae.save_user_embeddings(
            "alice", [_EMBEDDING_ALICE],
            output_path=tmp_encodings / "alice.pkl",
        )
        records = ae.load_all_user_embeddings(encodings_dir=tmp_encodings)

        result = ae.find_best_match(
            candidate_embedding=_EMBEDDING_STRANGER,
            known_records=records,
            threshold=0.30,
        )
        assert result["is_match"] is False

    def test_tampered_embedding_never_matches(
        self, tmp_encodings: Path, signed_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A tampered .pkl must be skipped — attacker cannot gain access."""
        import src.ai_engine as ae
        from src.integrity import IntegrityError, wrap_record_for_storage
        from src.utils import save_pickle

        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_encodings)

        # Write a tampered file (attacker's embedding injected)
        rec = {
            "user_name": "alice", "safe_name": "alice",
            "model_name": "Facenet512", "detector_backend": "opencv",
            "distance_metric": "cosine", "threshold": 0.3,
            "embeddings": [_EMBEDDING_ALICE], "embedding_count": 1,
            "image_paths": [], "created_at": "2026-01-01T00:00:00",
        }
        payload = wrap_record_for_storage(rec)
        # Attacker injects their embedding after signing
        payload["record"]["embeddings"] = [_EMBEDDING_STRANGER]
        save_pickle(payload, tmp_encodings / "alice.pkl")

        # load_all skips the tampered file
        records = ae.load_all_user_embeddings(encodings_dir=tmp_encodings)

        # Result: no records loaded → attacker is denied
        result = ae.find_best_match(
            candidate_embedding=_EMBEDDING_STRANGER,
            known_records=records,
            threshold=0.30,
        )
        assert result["is_match"] is False
        assert result.get("reason") == "no_registered_embeddings"
