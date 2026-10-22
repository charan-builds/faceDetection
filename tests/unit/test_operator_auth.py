"""
Unit tests for src.operator_auth.

Coverage targets
----------------
- sha256_hash_pin()     : correctness, encoding, empty input
- is_demo_mode()        : env var absent, empty, whitespace, configured
- verify_operator_pin() : demo path, correct PIN, wrong PIN, max attempts,
                          aborted input (EOF / KeyboardInterrupt)

Audit-log events verified
--------------------------
- operator_auth_demo_mode   (logged when env var is absent)
- operator_auth_success     (logged on correct PIN)
- operator_auth_failure     (logged on each wrong PIN and on abort)

Constraints
-----------
- No real PIN hash ever hardcoded in tests.
- getpass.getpass is always mocked — no stdin dependency.
- Environment variable is set/cleared via monkeypatch only.
- All tests are fast and hermetic.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterator
from unittest.mock import MagicMock, call, patch

import pytest

from src.operator_auth import (
    ENV_PIN_HASH,
    MAX_PIN_ATTEMPTS,
    is_demo_mode,
    sha256_hash_pin,
    verify_operator_pin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(pin: str) -> str:
    """Re-implement the hash independently so tests do not tautologically call
    the function under test to build their expected values."""
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


def _set_pin_hash(monkeypatch: pytest.MonkeyPatch, pin: str) -> str:
    """Hash *pin* and set the env var.  Returns the hash string."""
    h = _hash(pin)
    monkeypatch.setenv(ENV_PIN_HASH, h)
    return h


# ---------------------------------------------------------------------------
# Fixture: attach caplog to the non-propagating face_lock logger hierarchy
#
# The face_lock logger has propagate=False (set in setup_logging), so pytest's
# caplog handler — which attaches to the root logger — never receives records
# from face_lock child loggers.  This fixture manually adds caplog.handler to
# the face_lock logger so all operator_auth log events are captured.
# This mirrors the pattern used in test_logging_config.py.
# ---------------------------------------------------------------------------

@pytest.fixture()
def capture_face_lock_logs(
    caplog: pytest.LogCaptureFixture,
) -> pytest.LogCaptureFixture:
    """
    Attach caplog's handler to the face_lock logger so that audit events
    emitted by operator_auth (and any other face_lock child logger) are
    visible in caplog.records during tests.
    """
    face_lock_logger = logging.getLogger("face_lock")
    face_lock_logger.addHandler(caplog.handler)
    original_level = face_lock_logger.level
    face_lock_logger.setLevel(logging.DEBUG)

    yield caplog

    face_lock_logger.removeHandler(caplog.handler)
    face_lock_logger.setLevel(original_level)


# ---------------------------------------------------------------------------
# sha256_hash_pin
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSha256HashPin:
    def test_known_vector_matches_stdlib(self) -> None:
        """Result must match a direct hashlib call — no deviation allowed."""
        pin = "correct-horse-battery-staple"
        assert sha256_hash_pin(pin) == hashlib.sha256(pin.encode("utf-8")).hexdigest()

    def test_returns_64_char_hex_string(self) -> None:
        result = sha256_hash_pin("1234")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_pins_produce_different_hashes(self) -> None:
        assert sha256_hash_pin("aaaa") != sha256_hash_pin("aaab")

    def test_empty_pin_is_hashed_not_rejected(self) -> None:
        """Empty string is a valid input — the hash of b'' is deterministic."""
        result = sha256_hash_pin("")
        assert result == hashlib.sha256(b"").hexdigest()

    def test_unicode_pin_encoded_as_utf8(self) -> None:
        """Multi-byte characters must be encoded before hashing."""
        pin = "pïn"
        expected = hashlib.sha256(pin.encode("utf-8")).hexdigest()
        assert sha256_hash_pin(pin) == expected

    def test_same_pin_always_produces_same_hash(self) -> None:
        assert sha256_hash_pin("abc") == sha256_hash_pin("abc")


# ---------------------------------------------------------------------------
# is_demo_mode
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsDemoMode:
    def test_demo_mode_when_env_var_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_PIN_HASH, raising=False)
        assert is_demo_mode() is True

    def test_demo_mode_when_env_var_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_PIN_HASH, "")
        assert is_demo_mode() is True

    def test_demo_mode_when_env_var_only_whitespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_PIN_HASH, "   ")
        assert is_demo_mode() is True

    def test_not_demo_mode_when_hash_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_pin_hash(monkeypatch, "secret")
        assert is_demo_mode() is False

    def test_not_demo_mode_when_arbitrary_non_empty_string_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even a non-SHA-256 value disables demo mode — length is not validated here.
        monkeypatch.setenv(ENV_PIN_HASH, "x")
        assert is_demo_mode() is False


# ---------------------------------------------------------------------------
# verify_operator_pin — demo path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVerifyOperatorPinDemoMode:
    def test_returns_true_in_demo_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_PIN_HASH, raising=False)
        assert verify_operator_pin() is True

    def test_does_not_prompt_for_pin_in_demo_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_PIN_HASH, raising=False)
        with patch("src.operator_auth.getpass.getpass") as mock_getpass:
            verify_operator_pin()
        mock_getpass.assert_not_called()

    def test_logs_demo_mode_warning(
        self, monkeypatch: pytest.MonkeyPatch, capture_face_lock_logs: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv(ENV_PIN_HASH, raising=False)
        with capture_face_lock_logs.at_level(logging.WARNING, logger="face_lock"):
            verify_operator_pin()
        assert any("operator_auth_demo_mode" in r.message for r in capture_face_lock_logs.records)

    def test_demo_mode_warning_contains_env_var_name(
        self, monkeypatch: pytest.MonkeyPatch, capture_face_lock_logs: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv(ENV_PIN_HASH, raising=False)
        with capture_face_lock_logs.at_level(logging.WARNING, logger="face_lock"):
            verify_operator_pin()
        combined = " ".join(r.message for r in capture_face_lock_logs.records)
        assert ENV_PIN_HASH in combined


# ---------------------------------------------------------------------------
# verify_operator_pin — configured mode, correct PIN
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVerifyOperatorPinSuccess:
    def test_correct_pin_on_first_attempt_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_pin_hash(monkeypatch, "goodpin")
        with patch("src.operator_auth.getpass.getpass", return_value="goodpin"):
            assert verify_operator_pin() is True

    def test_correct_pin_on_second_attempt_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_pin_hash(monkeypatch, "goodpin")
        with patch(
            "src.operator_auth.getpass.getpass",
            side_effect=["wrongpin", "goodpin"],
        ):
            assert verify_operator_pin() is True

    def test_correct_pin_on_last_attempt_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_pin_hash(monkeypatch, "goodpin")
        wrong_then_right = ["wrongpin"] * (MAX_PIN_ATTEMPTS - 1) + ["goodpin"]
        with patch(
            "src.operator_auth.getpass.getpass", side_effect=wrong_then_right
        ):
            assert verify_operator_pin() is True

    def test_logs_auth_success_event(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capture_face_lock_logs: pytest.LogCaptureFixture,
    ) -> None:
        _set_pin_hash(monkeypatch, "goodpin")
        with patch("src.operator_auth.getpass.getpass", return_value="goodpin"):
            with capture_face_lock_logs.at_level(logging.INFO, logger="face_lock"):
                verify_operator_pin()
        assert any("operator_auth_success" in r.message for r in capture_face_lock_logs.records)

    def test_getpass_called_exactly_once_on_first_try(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_pin_hash(monkeypatch, "goodpin")
        with patch(
            "src.operator_auth.getpass.getpass", return_value="goodpin"
        ) as mock_gp:
            verify_operator_pin()
        mock_gp.assert_called_once()


# ---------------------------------------------------------------------------
# verify_operator_pin — configured mode, wrong PIN
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVerifyOperatorPinFailure:
    def test_all_wrong_pins_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_pin_hash(monkeypatch, "goodpin")
        with patch(
            "src.operator_auth.getpass.getpass",
            side_effect=["bad1", "bad2", "bad3"],
        ):
            assert verify_operator_pin() is False

    def test_getpass_called_max_attempts_times_on_full_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_pin_hash(monkeypatch, "goodpin")
        with patch(
            "src.operator_auth.getpass.getpass",
            side_effect=["bad"] * MAX_PIN_ATTEMPTS,
        ) as mock_gp:
            verify_operator_pin()
        assert mock_gp.call_count == MAX_PIN_ATTEMPTS

    def test_logs_failure_event_for_each_wrong_pin(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capture_face_lock_logs: pytest.LogCaptureFixture,
    ) -> None:
        _set_pin_hash(monkeypatch, "goodpin")
        with patch(
            "src.operator_auth.getpass.getpass",
            side_effect=["bad"] * MAX_PIN_ATTEMPTS,
        ):
            with capture_face_lock_logs.at_level(logging.WARNING, logger="face_lock"):
                verify_operator_pin()

        failure_events = [
            r for r in capture_face_lock_logs.records if "operator_auth_failure" in r.message
        ]
        # One failure log per wrong attempt
        assert len(failure_events) == MAX_PIN_ATTEMPTS

    def test_final_failure_logged_at_error_level(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capture_face_lock_logs: pytest.LogCaptureFixture,
    ) -> None:
        _set_pin_hash(monkeypatch, "goodpin")
        with patch(
            "src.operator_auth.getpass.getpass",
            side_effect=["bad"] * MAX_PIN_ATTEMPTS,
        ):
            with capture_face_lock_logs.at_level(logging.ERROR, logger="face_lock"):
                verify_operator_pin()

        error_failures = [
            r
            for r in capture_face_lock_logs.records
            if "operator_auth_failure" in r.message and r.levelno == logging.ERROR
        ]
        assert len(error_failures) >= 1

    def test_wrong_pin_does_not_log_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capture_face_lock_logs: pytest.LogCaptureFixture,
    ) -> None:
        _set_pin_hash(monkeypatch, "goodpin")
        with patch(
            "src.operator_auth.getpass.getpass",
            side_effect=["bad"] * MAX_PIN_ATTEMPTS,
        ):
            with capture_face_lock_logs.at_level(logging.INFO, logger="face_lock"):
                verify_operator_pin()

        assert not any(
            "operator_auth_success" in r.message for r in capture_face_lock_logs.records
        )


# ---------------------------------------------------------------------------
# verify_operator_pin — aborted input
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVerifyOperatorPinAborted:
    def test_eof_error_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-interactive environments raise EOFError from getpass."""
        _set_pin_hash(monkeypatch, "goodpin")
        with patch(
            "src.operator_auth.getpass.getpass", side_effect=EOFError
        ):
            assert verify_operator_pin() is False

    def test_keyboard_interrupt_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ctrl+C during PIN prompt must deny access, not crash."""
        _set_pin_hash(monkeypatch, "goodpin")
        with patch(
            "src.operator_auth.getpass.getpass", side_effect=KeyboardInterrupt
        ):
            assert verify_operator_pin() is False

    def test_aborted_input_logs_failure_event(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capture_face_lock_logs: pytest.LogCaptureFixture,
    ) -> None:
        _set_pin_hash(monkeypatch, "goodpin")
        with patch(
            "src.operator_auth.getpass.getpass", side_effect=EOFError
        ):
            with capture_face_lock_logs.at_level(logging.ERROR, logger="face_lock"):
                verify_operator_pin()

        assert any("operator_auth_failure" in r.message for r in capture_face_lock_logs.records)


# ---------------------------------------------------------------------------
# app.py integration — registration gate
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAppRegistrationGate:
    """
    Verify that app.handle_registration() calls verify_operator_pin() and
    only proceeds to register_user() when it returns True.
    """

    def test_registration_blocked_when_auth_fails(self) -> None:
        with (
            patch(
                "app.verify_operator_pin", return_value=False
            ) as mock_auth,
            patch("app.register_user") as mock_register,
        ):
            from app import handle_registration

            handle_registration()

        mock_auth.assert_called_once()
        mock_register.assert_not_called()

    def test_registration_proceeds_when_auth_succeeds(self) -> None:
        with (
            patch("app.verify_operator_pin", return_value=True),
            patch(
                "app.register_user",
                return_value={"success": True, "user_name": "alice", "embedding_count": 3},
            ) as mock_register,
        ):
            from app import handle_registration

            handle_registration()

        mock_register.assert_called_once()

    def test_registration_prints_error_on_auth_failure(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with (
            patch("app.verify_operator_pin", return_value=False),
            patch("app.register_user"),
        ):
            from app import handle_registration

            handle_registration()

        captured = capsys.readouterr()
        assert "denied" in captured.out.lower() or "error" in captured.out.lower()
