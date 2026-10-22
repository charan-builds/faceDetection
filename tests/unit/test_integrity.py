"""
Unit tests for src.integrity and the integrity-aware save/load paths in
src.ai_engine.

Security model verified
-----------------------
- sign_record / verify_record: HMAC-SHA256 over canonical JSON.
- Tampered records (any field changed) must fail verification.
- Missing signature in signed mode must fail verification.
- Unsigned mode (no env var) passes verification with a warning.
- Legacy flat records load with a warning, no IntegrityError.

Test inventory
--------------
1.  sign_record returns 64-char hex in signed mode
2.  sign_record returns empty string in unsigned mode
3.  sign_record produces a different digest for different records
4.  verify_record returns (True, "signature_valid") for correct signature
5.  verify_record returns (False, "signature_mismatch") when record is tampered
6.  verify_record returns (False, "missing_signature") when sig is empty in signed mode
7.  verify_record returns (True, "unsigned_mode") when no key configured
8.  Tampering a single float in embeddings is detected
9.  Tampering the user_name field is detected
10. wrap_record_for_storage / unwrap_stored_payload round-trip (signed)
11. wrap_record_for_storage / unwrap_stored_payload round-trip (unsigned)
12. unwrap_stored_payload identifies legacy flat format
13. IntegrityError is a subclass of ValueError
14. ai_engine.save_user_embeddings writes wrapped payload
15. ai_engine.load_user_embeddings verifies and returns inner record
16. ai_engine.load_user_embeddings raises IntegrityError on tampered file
17. ai_engine.load_user_embeddings loads legacy file with warning
18. ai_engine.load_all_user_embeddings rejects tampered file
19. ai_engine.load_all_user_embeddings loads legacy files with warning
20. Canonical JSON is deterministic regardless of dict insertion order
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.integrity import (
    ENV_SIGNING_KEY,
    ENV_STRICT_MODE,
    IntegrityError,
    _canonical_bytes,
    enforce_strict_mode_policy,
    is_signing_configured,
    is_strict_mode,
    sign_record,
    unwrap_stored_payload,
    verify_record,
    wrap_record_for_storage,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

_TEST_KEY = "deadbeefdeadbeefdeadbeefdeadbeef01020304050607080910111213141516"

# Minimal embedding record matching the schema produced by create_embedding_record.
_SAMPLE_RECORD: dict[str, Any] = {
    "user_name": "alice",
    "safe_name": "alice",
    "model_name": "Facenet512",
    "detector_backend": "opencv",
    "distance_metric": "cosine",
    "threshold": 0.30,
    "embeddings": [[0.1, 0.2, 0.3]],
    "embedding_count": 1,
    "image_paths": ["/data/alice/img.jpg"],
    "created_at": "2026-06-01T12:00:00",
}


def _clone(record: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy via JSON so we always get a fresh plain-dict with no shared refs."""
    return json.loads(json.dumps(record))


@pytest.fixture()
def signed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set FACE_LOCK_SIGNING_KEY for the duration of one test."""
    monkeypatch.setenv(ENV_SIGNING_KEY, _TEST_KEY)


@pytest.fixture()
def unsigned_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove FACE_LOCK_SIGNING_KEY so unsigned mode is active."""
    monkeypatch.delenv(ENV_SIGNING_KEY, raising=False)


@pytest.fixture()
def capture_face_lock_logs(
    caplog: pytest.LogCaptureFixture,
) -> pytest.LogCaptureFixture:
    """Attach caplog to the non-propagating face_lock logger."""
    face_lock_logger = logging.getLogger("face_lock")
    face_lock_logger.addHandler(caplog.handler)
    original_level = face_lock_logger.level
    face_lock_logger.setLevel(logging.DEBUG)
    yield caplog
    face_lock_logger.removeHandler(caplog.handler)
    face_lock_logger.setLevel(original_level)


# ---------------------------------------------------------------------------
# 1-2  sign_record output shape
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSignRecord:
    def test_returns_64_char_hex_in_signed_mode(self, signed_env: None) -> None:
        sig = sign_record(_clone(_SAMPLE_RECORD))
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_returns_empty_string_in_unsigned_mode(self, unsigned_env: None) -> None:
        sig = sign_record(_clone(_SAMPLE_RECORD))
        assert sig == ""

    def test_different_records_produce_different_signatures(
        self, signed_env: None
    ) -> None:
        rec_a = _clone(_SAMPLE_RECORD)
        rec_b = _clone(_SAMPLE_RECORD)
        rec_b["user_name"] = "bob"
        assert sign_record(rec_a) != sign_record(rec_b)

    def test_same_record_always_produces_same_signature(
        self, signed_env: None
    ) -> None:
        rec = _clone(_SAMPLE_RECORD)
        assert sign_record(rec) == sign_record(rec)

    def test_sign_record_uses_hmac_sha256_over_canonical_json(
        self, signed_env: None
    ) -> None:
        """Independent re-derivation of expected signature must match."""
        rec = _clone(_SAMPLE_RECORD)
        payload = _canonical_bytes(rec)
        expected = hmac.new(
            _TEST_KEY.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()
        assert sign_record(rec) == expected


# ---------------------------------------------------------------------------
# 3-7  verify_record return codes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVerifyRecord:
    def test_valid_signature_returns_true_and_reason(
        self, signed_env: None
    ) -> None:
        rec = _clone(_SAMPLE_RECORD)
        sig = sign_record(rec)
        valid, reason = verify_record(rec, sig)
        assert valid is True
        assert reason == "signature_valid"

    def test_mismatch_returns_false_and_reason(
        self, signed_env: None
    ) -> None:
        rec = _clone(_SAMPLE_RECORD)
        sig = sign_record(rec)
        # Tamper: change a field AFTER signing
        rec["user_name"] = "mallory"
        valid, reason = verify_record(rec, sig)
        assert valid is False
        assert reason == "signature_mismatch"

    def test_missing_signature_returns_false_in_signed_mode(
        self, signed_env: None
    ) -> None:
        rec = _clone(_SAMPLE_RECORD)
        valid, reason = verify_record(rec, "")
        assert valid is False
        assert reason == "missing_signature"

    def test_unsigned_mode_always_returns_true(
        self, unsigned_env: None
    ) -> None:
        rec = _clone(_SAMPLE_RECORD)
        valid, reason = verify_record(rec, "")
        assert valid is True
        assert reason == "unsigned_mode"

    def test_unsigned_mode_with_nonempty_sig_still_passes(
        self, unsigned_env: None
    ) -> None:
        """When no key is configured, any signature is ignored."""
        valid, reason = verify_record(_clone(_SAMPLE_RECORD), "aabbcc")
        assert valid is True
        assert reason == "unsigned_mode"

    def test_verify_never_raises(self, signed_env: None) -> None:
        """verify_record must return a tuple, never raise, even for garbage input."""
        result = verify_record({}, "not-a-real-sig")
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 8-9  Tamper detection — field-level sensitivity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTamperDetection:
    def test_single_float_change_detected(self, signed_env: None) -> None:
        rec = _clone(_SAMPLE_RECORD)
        sig = sign_record(rec)
        # Mutate one embedding value by a tiny epsilon
        rec["embeddings"][0][0] += 1e-9
        valid, _ = verify_record(rec, sig)
        assert valid is False

    def test_user_name_change_detected(self, signed_env: None) -> None:
        rec = _clone(_SAMPLE_RECORD)
        sig = sign_record(rec)
        rec["user_name"] = "mallory"
        valid, _ = verify_record(rec, sig)
        assert valid is False

    def test_added_field_detected(self, signed_env: None) -> None:
        rec = _clone(_SAMPLE_RECORD)
        sig = sign_record(rec)
        rec["injected"] = "payload"
        valid, _ = verify_record(rec, sig)
        assert valid is False

    def test_removed_field_detected(self, signed_env: None) -> None:
        rec = _clone(_SAMPLE_RECORD)
        sig = sign_record(rec)
        del rec["model_name"]
        valid, _ = verify_record(rec, sig)
        assert valid is False


# ---------------------------------------------------------------------------
# 10-12  wrap / unwrap round-trips
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWrapUnwrap:
    def test_signed_round_trip(self, signed_env: None) -> None:
        rec = _clone(_SAMPLE_RECORD)
        payload = wrap_record_for_storage(rec)

        assert "record" in payload
        assert "signature" in payload
        assert len(payload["signature"]) == 64

        out_rec, out_sig, is_legacy = unwrap_stored_payload(payload)
        assert out_rec == rec
        assert out_sig == payload["signature"]
        assert is_legacy is False

    def test_unsigned_round_trip(self, unsigned_env: None) -> None:
        rec = _clone(_SAMPLE_RECORD)
        payload = wrap_record_for_storage(rec)

        assert payload["signature"] == ""
        out_rec, out_sig, is_legacy = unwrap_stored_payload(payload)
        assert out_rec == rec
        assert out_sig == ""
        assert is_legacy is False

    def test_legacy_flat_record_identified(self) -> None:
        """A flat dict with 'embeddings' but no 'record' key is legacy format."""
        legacy = _clone(_SAMPLE_RECORD)
        out_rec, out_sig, is_legacy = unwrap_stored_payload(legacy)
        assert is_legacy is True
        assert out_sig == ""
        assert out_rec is legacy


# ---------------------------------------------------------------------------
# 13  Exception hierarchy
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIntegrityError:
    def test_is_subclass_of_value_error(self) -> None:
        assert issubclass(IntegrityError, ValueError)

    def test_can_be_caught_as_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise IntegrityError("tampered")


# ---------------------------------------------------------------------------
# 14  Canonical JSON is deterministic
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCanonicalBytes:
    def test_key_order_does_not_affect_bytes(self) -> None:
        a = {"z": 1, "a": 2}
        b = {"a": 2, "z": 1}
        assert _canonical_bytes(a) == _canonical_bytes(b)

    def test_same_data_always_same_bytes(self) -> None:
        assert _canonical_bytes(_SAMPLE_RECORD) == _canonical_bytes(_SAMPLE_RECORD)

    def test_different_values_produce_different_bytes(self) -> None:
        a = {"x": 1}
        b = {"x": 2}
        assert _canonical_bytes(a) != _canonical_bytes(b)


# ---------------------------------------------------------------------------
# 15-19  ai_engine integration tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAiEngineIntegration:
    """
    Integration tests for save_user_embeddings / load_user_embeddings /
    load_all_user_embeddings — no real webcam or DeepFace.
    """

    _EMBEDDINGS = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def _save_raw_legacy(self, path: Path) -> None:
        """Write a legacy flat-dict pickle directly, bypassing save_user_embeddings."""
        record = _clone(_SAMPLE_RECORD)
        with path.open("wb") as f:
            pickle.dump(record, f)

    def _save_raw_tampered(self, path: Path, signed_env: None) -> None:
        """Write a signed payload then mutate the inner record on disk."""
        from src.integrity import wrap_record_for_storage
        from src.utils import save_pickle

        rec = _clone(_SAMPLE_RECORD)
        payload = wrap_record_for_storage(rec)
        # Tamper: overwrite the inner record AFTER signing
        payload["record"]["user_name"] = "mallory"
        save_pickle(payload, path)

    # 14 — save writes wrapped payload
    def test_save_writes_wrapped_payload(
        self, tmp_path: Path, signed_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """save_user_embeddings must persist a {record, signature} dict."""
        from src.utils import ENCODINGS_DIR, load_pickle
        import src.ai_engine as ae

        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_path)
        out_path = tmp_path / "alice.pkl"

        ae.save_user_embeddings(
            user_name="alice",
            embeddings=self._EMBEDDINGS,
            output_path=out_path,
        )

        raw = load_pickle(out_path)
        assert "record" in raw
        assert "signature" in raw
        assert len(raw["signature"]) == 64

    # 15 — load verifies and returns the inner record
    def test_load_returns_inner_record(
        self, tmp_path: Path, signed_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.ai_engine as ae

        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_path)
        out_path = tmp_path / "alice.pkl"

        ae.save_user_embeddings(
            user_name="alice",
            embeddings=self._EMBEDDINGS,
            output_path=out_path,
        )

        record = ae.load_user_embeddings("alice", input_path=out_path)
        assert record["user_name"] == "alice"
        assert record["embeddings"] == self._EMBEDDINGS
        # Inner record — no wrapper keys
        assert "record" not in record
        assert "signature" not in record

    # 16 — load raises IntegrityError on tampered file
    def test_load_raises_integrity_error_on_tampered_file(
        self, tmp_path: Path, signed_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.ai_engine as ae

        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_path)
        tampered_path = tmp_path / "tampered.pkl"
        self._save_raw_tampered(tampered_path, signed_env)

        with pytest.raises(IntegrityError, match="integrity check"):
            ae.load_user_embeddings("tampered", input_path=tampered_path)

    # 17 — load_user_embeddings loads legacy file with warning, not error
    def test_load_legacy_file_succeeds_with_warning(
        self,
        tmp_path: Path,
        signed_env: None,
        monkeypatch: pytest.MonkeyPatch,
        capture_face_lock_logs: pytest.LogCaptureFixture,
    ) -> None:
        import src.ai_engine as ae

        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_path)
        legacy_path = tmp_path / "legacy.pkl"
        self._save_raw_legacy(legacy_path)

        with capture_face_lock_logs.at_level(logging.WARNING, logger="face_lock"):
            record = ae.load_user_embeddings("legacy", input_path=legacy_path)

        assert record["user_name"] == "alice"
        assert any(
            "integrity_legacy_record_loaded" in r.message
            for r in capture_face_lock_logs.records
        )

    # 18 — load_all skips tampered file, does NOT raise
    def test_load_all_skips_tampered_file(
        self, tmp_path: Path, signed_env: None,
        capture_face_lock_logs: pytest.LogCaptureFixture,
    ) -> None:
        import src.ai_engine as ae

        tampered_path = tmp_path / "tampered.pkl"
        self._save_raw_tampered(tampered_path, signed_env)

        with capture_face_lock_logs.at_level(logging.ERROR, logger="face_lock"):
            records = ae.load_all_user_embeddings(encodings_dir=tmp_path)

        # Tampered file is skipped, not raised
        assert records == []
        assert any(
            "embedding_load_skipped" in r.message for r in capture_face_lock_logs.records
        )

    def test_load_all_skips_tampered_loads_valid(
        self, tmp_path: Path, signed_env: None,
        monkeypatch: pytest.MonkeyPatch,
        capture_face_lock_logs: pytest.LogCaptureFixture,
    ) -> None:
        """One corrupt + one valid file → valid record returned, corrupt skipped."""
        import src.ai_engine as ae

        # Write a valid signed file
        valid_path = tmp_path / "alice.pkl"
        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_path)
        ae.save_user_embeddings("alice", self._EMBEDDINGS, output_path=valid_path)

        # Write a tampered file
        tampered_path = tmp_path / "tampered.pkl"
        self._save_raw_tampered(tampered_path, signed_env)

        with capture_face_lock_logs.at_level(logging.WARNING, logger="face_lock"):
            records = ae.load_all_user_embeddings(encodings_dir=tmp_path)

        assert len(records) == 1
        assert records[0]["user_name"] == "alice"
        assert any(
            "embedding_load_partial" in r.message for r in capture_face_lock_logs.records
        )

    # 19 — load_all loads legacy files with warning
    def test_load_all_loads_legacy_files_with_warning(
        self,
        tmp_path: Path,
        signed_env: None,
        capture_face_lock_logs: pytest.LogCaptureFixture,
    ) -> None:
        import src.ai_engine as ae

        legacy_path = tmp_path / "alice.pkl"
        self._save_raw_legacy(legacy_path)

        with capture_face_lock_logs.at_level(logging.WARNING, logger="face_lock"):
            records = ae.load_all_user_embeddings(encodings_dir=tmp_path)

        assert len(records) == 1
        assert records[0]["user_name"] == "alice"
        assert any(
            "integrity_legacy_record_loaded" in r.message
            for r in capture_face_lock_logs.records
        )

    # unsigned mode: save and load both succeed with warnings
    def test_unsigned_save_and_load_succeeds(
        self, tmp_path: Path, unsigned_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.ai_engine as ae

        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_path)
        out_path = tmp_path / "alice.pkl"

        ae.save_user_embeddings(
            user_name="alice",
            embeddings=self._EMBEDDINGS,
            output_path=out_path,
        )
        record = ae.load_user_embeddings("alice", input_path=out_path)
        assert record["user_name"] == "alice"


# ---------------------------------------------------------------------------
# 20  is_signing_configured
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsSigningConfigured:
    def test_returns_true_when_key_set(self, signed_env: None) -> None:
        assert is_signing_configured() is True

    def test_returns_false_when_key_absent(self, unsigned_env: None) -> None:
        assert is_signing_configured() is False

    def test_returns_false_when_key_is_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_SIGNING_KEY, "")
        assert is_signing_configured() is False

    def test_returns_false_when_key_is_whitespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_SIGNING_KEY, "   ")
        assert is_signing_configured() is False


# ---------------------------------------------------------------------------
# 21  Strict mode template integrity and fail-closed policies
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStrictModeTemplateIntegrity:
    _EMBEDDINGS: list[list[float]] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_valid_signed_template_loads_with_correct_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In strict mode, a valid signed template loads cleanly with matching key."""
        import src.ai_engine as ae
        monkeypatch.setenv(ENV_STRICT_MODE, "1")
        monkeypatch.setenv(ENV_SIGNING_KEY, _TEST_KEY)
        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_path)

        out_path = tmp_path / "synthetic_user.pkl"
        ae.save_user_embeddings("synthetic_user", self._EMBEDDINGS, output_path=out_path)

        loaded = ae.load_user_embeddings("synthetic_user", input_path=out_path)
        assert loaded["user_name"] == "synthetic_user"
        assert loaded["embeddings"] == self._EMBEDDINGS

    def test_modified_signed_template_is_rejected_in_strict_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In strict mode, tampered signed template raises IntegrityError."""
        import src.ai_engine as ae
        from src.utils import load_pickle, save_pickle
        monkeypatch.setenv(ENV_STRICT_MODE, "1")
        monkeypatch.setenv(ENV_SIGNING_KEY, _TEST_KEY)
        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_path)

        out_path = tmp_path / "synthetic_user.pkl"
        ae.save_user_embeddings("synthetic_user", self._EMBEDDINGS, output_path=out_path)

        # Tamper with the file
        payload = load_pickle(out_path)
        payload["record"]["embeddings"][0][0] += 0.99
        save_pickle(payload, out_path)

        with pytest.raises(IntegrityError, match="integrity check"):
            ae.load_user_embeddings("synthetic_user", input_path=out_path)

    def test_unsigned_legacy_template_is_rejected_in_strict_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In strict mode, flat legacy templates must fail closed with IntegrityError."""
        import src.ai_engine as ae
        from src.utils import save_pickle
        monkeypatch.setenv(ENV_STRICT_MODE, "1")
        monkeypatch.setenv(ENV_SIGNING_KEY, _TEST_KEY)
        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_path)

        legacy_path = tmp_path / "legacy_user.pkl"
        legacy_data = {
            "user_name": "legacy_user",
            "safe_name": "legacy_user",
            "model_name": "Facenet512",
            "detector_backend": "opencv",
            "distance_metric": "cosine",
            "threshold": 0.30,
            "embeddings": self._EMBEDDINGS,
            "embedding_count": 2,
            "image_paths": [],
            "created_at": "2026-01-01T00:00:00",
        }
        save_pickle(legacy_data, legacy_path)

        # Direct call to load_user_embeddings must raise IntegrityError
        with pytest.raises(IntegrityError, match="rejected in strict mode"):
            ae.load_user_embeddings("legacy_user", input_path=legacy_path)

        # Direct call to load_all_user_embeddings skips it (fails closed)
        records = ae.load_all_user_embeddings(encodings_dir=tmp_path)
        assert records == []

    def test_legacy_template_follows_documented_non_strict_policy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture_face_lock_logs: pytest.LogCaptureFixture
    ) -> None:
        """When strict mode is NOT enabled, legacy template loads with a warning."""
        import src.ai_engine as ae
        from src.utils import save_pickle
        monkeypatch.delenv(ENV_STRICT_MODE, raising=False)
        monkeypatch.setenv(ENV_SIGNING_KEY, _TEST_KEY)
        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_path)

        legacy_path = tmp_path / "legacy_user.pkl"
        legacy_data = {
            "user_name": "legacy_user",
            "safe_name": "legacy_user",
            "model_name": "Facenet512",
            "detector_backend": "opencv",
            "distance_metric": "cosine",
            "threshold": 0.30,
            "embeddings": self._EMBEDDINGS,
            "embedding_count": 2,
            "image_paths": [],
            "created_at": "2026-01-01T00:00:00",
        }
        save_pickle(legacy_data, legacy_path)

        with capture_face_lock_logs.at_level(logging.WARNING, logger="face_lock"):
            record = ae.load_user_embeddings("legacy_user", input_path=legacy_path)
        assert record["user_name"] == "legacy_user"
        assert any("integrity_legacy_record_loaded" in r.message for r in capture_face_lock_logs.records)

    def test_missing_signing_key_fails_closed_in_strict_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In strict mode without signing key, sign and load fail closed with IntegrityError."""
        import src.ai_engine as ae
        from src.utils import save_pickle
        monkeypatch.setenv(ENV_STRICT_MODE, "1")
        monkeypatch.delenv(ENV_SIGNING_KEY, raising=False)
        monkeypatch.setattr(ae, "ENCODINGS_DIR", tmp_path)

        # sign_record fails closed
        with pytest.raises(IntegrityError, match="FACE_LOCK_SIGNING_KEY is missing"):
            sign_record({"user_name": "test"})

        # verify_record fails closed
        valid, reason = verify_record({"user_name": "test"}, "deadbeef" * 8)
        assert valid is False
        assert reason == "missing_signing_key"

        # load_user_embeddings fails closed
        out_path = tmp_path / "test.pkl"
        save_pickle({"record": {"user_name": "test", "embeddings": self._EMBEDDINGS}, "signature": "abc"}, out_path)
        with pytest.raises(IntegrityError, match="missing_signing_key"):
            ae.load_user_embeddings("test", input_path=out_path)

    def test_enforce_strict_mode_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """enforce_strict_mode_policy raises RuntimeError when required env vars are missing."""
        # When strict mode is off, policy always passes
        monkeypatch.delenv(ENV_STRICT_MODE, raising=False)
        monkeypatch.delenv(ENV_SIGNING_KEY, raising=False)
        monkeypatch.delenv("FACE_LOCK_OPERATOR_PIN_HASH", raising=False)
        enforce_strict_mode_policy(require_operator_auth=True)  # No error

        # When strict mode is on, missing signing key raises
        monkeypatch.setenv(ENV_STRICT_MODE, "1")
        with pytest.raises(RuntimeError, match="Strict mode is enabled"):
            enforce_strict_mode_policy(require_operator_auth=False)

        # Configured signing key but missing operator PIN raises when required
        monkeypatch.setenv(ENV_SIGNING_KEY, _TEST_KEY)
        enforce_strict_mode_policy(require_operator_auth=False)  # passes for recognition
        with pytest.raises(RuntimeError, match="Strict mode is enabled"):
            enforce_strict_mode_policy(require_operator_auth=True)  # fails for registration

        # Both configured -> passes
        monkeypatch.setenv("FACE_LOCK_OPERATOR_PIN_HASH", "abc123hash")
        enforce_strict_mode_policy(require_operator_auth=True)  # passes
