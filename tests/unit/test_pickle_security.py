"""
Unit tests for the secure pickle deserialization introduced in src.utils.

Security model verified
-----------------------
_SafeUnpickler allows only a hard-coded set of plain Python data types:
    dict, list, tuple, str, int, float, bool  (builtins)
    datetime.datetime                          (legacy compatibility)

It blocks everything else — no matter how the pickle stream is crafted —
by overriding ``find_class`` before any code can execute.

Test inventory
--------------
1.  Valid embedding record loads without error
2.  Valid record round-trips through save_pickle / load_pickle
3.  Malformed / truncated bytes raise an exception (not silently return None)
4.  Malicious payload using os.system is blocked
5.  Malicious payload using subprocess.Popen is blocked
6.  Malicious payload using builtins.eval is blocked
7.  Reduce-based RCE gadget is blocked
8.  Arbitrary user-defined class is blocked
9.  Missing file with no default raises FileNotFoundError
10. Missing file with a default returns the default value
11. RestrictedUnpicklingError is a subclass of ValueError (caller compat)
"""

from __future__ import annotations

import io
import os
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from src.utils import RestrictedUnpicklingError, _SafeUnpickler, load_pickle, save_pickle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pickle_bytes(obj: Any) -> bytes:
    """Serialise an object to raw pickle bytes without touching the filesystem."""
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return buf.getvalue()


def _load_from_bytes(data: bytes) -> Any:
    """Run _SafeUnpickler on raw bytes (no file required)."""
    return _SafeUnpickler(io.BytesIO(data)).load()


def _make_embedding_record() -> dict[str, Any]:
    """Build a minimal embedding record that matches the real Face Lock schema."""
    return {
        "user_name": "alice",
        "safe_name": "alice",
        "model_name": "Facenet512",
        "detector_backend": "opencv",
        "distance_metric": "cosine",
        "threshold": 0.30,
        "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "embedding_count": 2,
        "image_paths": ["/data/alice/alice_001.jpg"],
        "created_at": "2026-06-01T12:00:00",
    }


# ---------------------------------------------------------------------------
# 1 & 2 — Valid data passes through cleanly
# ---------------------------------------------------------------------------

class TestValidPayloads:
    def test_valid_embedding_record_loads(self) -> None:
        """A well-formed Face Lock record is loaded without any error."""
        record = _make_embedding_record()
        data = _pickle_bytes(record)
        loaded = _load_from_bytes(data)
        assert loaded["user_name"] == "alice"
        assert loaded["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_primitive_types_all_pass(self) -> None:
        """Every type on the allow-list survives a round-trip."""
        payload = {
            "a_str": "hello",
            "an_int": 42,
            "a_float": 3.14,
            "a_bool": True,
            "a_list": [1, 2, 3],
            "a_tuple": (4, 5, 6),
            "nested": {"key": "value"},
        }
        assert _load_from_bytes(_pickle_bytes(payload)) == payload

    def test_save_and_load_pickle_round_trip(self, tmp_path: Path) -> None:
        """save_pickle + load_pickle produces an identical record."""
        record = _make_embedding_record()
        pkl_path = tmp_path / "alice.pkl"

        saved_path = save_pickle(record, pkl_path)
        loaded = load_pickle(saved_path)

        assert loaded == record

    def test_empty_embeddings_list_loads(self, tmp_path: Path) -> None:
        """Edge case: an empty list round-trips safely."""
        pkl_path = tmp_path / "empty.pkl"
        save_pickle([], pkl_path)
        assert load_pickle(pkl_path) == []

    def test_datetime_object_loads(self) -> None:
        """datetime.datetime is allowed for legacy compatibility."""
        payload = {"created_at": datetime(2026, 6, 1, 12, 0, 0)}
        loaded = _load_from_bytes(_pickle_bytes(payload))
        assert loaded["created_at"].year == 2026


# ---------------------------------------------------------------------------
# 3 — Malformed bytes
# ---------------------------------------------------------------------------

class TestMalformedPickle:
    def test_truncated_bytes_raise_exception(self) -> None:
        """Corrupt / truncated data must raise, not silently return None."""
        with pytest.raises(Exception):
            _load_from_bytes(b"\x80\x04\x95")  # valid header, missing body

    def test_random_bytes_raise_exception(self) -> None:
        """Completely random bytes must not succeed."""
        with pytest.raises(Exception):
            _load_from_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")

    def test_empty_bytes_raise_exception(self) -> None:
        """An empty buffer must raise, not return None."""
        with pytest.raises(Exception):
            _load_from_bytes(b"")


# ---------------------------------------------------------------------------
# 4-7 — Malicious payloads are blocked BEFORE execution
# ---------------------------------------------------------------------------

def _gadget_bytes(module: str, name: str, arg: str) -> bytes:
    """
    Build a valid pickle stream that calls ``module.name(arg)`` using the
    standard GLOBAL+MARK+arg+TUPLE1+REDUCE pattern (protocol 2).

    We use ``pickle.dumps`` on a helper object so Python calculates all frame
    lengths correctly, then swap out the GLOBAL reference via a template.
    The resulting bytes are fed to _SafeUnpickler which must raise before
    ``find_class`` returns the real callable.
    """
    # pickle protocol 2 GLOBAL opcode: b'c' + b'module\nname\n'
    module_bytes = module.encode() + b"\n"
    name_bytes = name.encode() + b"\n"
    arg_bytes = arg.encode()

    # Build the stream manually using protocol 2 (no FRAME opcode needed)
    buf = io.BytesIO()
    buf.write(b"\x80\x02")           # PROTO 2
    buf.write(b"c")                  # GLOBAL
    buf.write(module_bytes)
    buf.write(name_bytes)
    buf.write(b"q\x00")              # BINPUT 0
    buf.write(b"(")                  # MARK
    # SHORT_BINUNICODE for the argument string
    arg_len = len(arg_bytes)
    buf.write(b"\x8c" + bytes([arg_len]))
    buf.write(arg_bytes)
    buf.write(b"\x85")               # TUPLE1
    buf.write(b"R")                  # REDUCE
    buf.write(b".")                  # STOP
    return buf.getvalue()


class TestMaliciousPayloads:
    """
    Each test crafts a real pickle gadget (using protocol 2 opcodes directly)
    and confirms it is blocked by RestrictedUnpicklingError before any
    side-effect can execute.
    """

    def test_os_system_gadget_is_blocked(self, tmp_path: Path) -> None:
        """
        Classic reduce-based gadget: os.system("echo pwned").
        The marker file must NOT be created, and the load must raise.
        """
        marker = tmp_path / "pwned.txt"
        data = _gadget_bytes("os", "system", f"echo pwned > {marker}")

        with pytest.raises(RestrictedUnpicklingError, match="os.system"):
            _SafeUnpickler(io.BytesIO(data)).load()

        assert not marker.exists(), "Malicious payload was executed — gadget was not blocked"

    def test_subprocess_popen_gadget_is_blocked(self) -> None:
        """subprocess.Popen must be blocked before instantiation."""
        data = _gadget_bytes("subprocess", "Popen", "ls")

        with pytest.raises(RestrictedUnpicklingError, match="subprocess.Popen"):
            _SafeUnpickler(io.BytesIO(data)).load()

    def test_builtins_eval_gadget_is_blocked(self) -> None:
        """
        eval via builtins must be blocked even though 'builtins' is the module
        for permitted types.  find_class checks the (module, name) *pair*, so
        builtins.eval is distinct from builtins.dict and must be denied.
        """
        data = _gadget_bytes("builtins", "eval", "1+1")

        with pytest.raises(RestrictedUnpicklingError, match="builtins.eval"):
            _SafeUnpickler(io.BytesIO(data)).load()

    def test_reduce_rce_gadget_via_importlib_is_blocked(self) -> None:
        """importlib.import_module gadget must be blocked."""
        data = _gadget_bytes("importlib", "import_module", "os")

        with pytest.raises(RestrictedUnpicklingError):
            _SafeUnpickler(io.BytesIO(data)).load()


# ---------------------------------------------------------------------------
# Module-level class used to test that arbitrary types are blocked.
# Must be at module level so that pickle can serialise it.
# ---------------------------------------------------------------------------

class _ArbitraryClass:
    """Represents any user-defined or third-party class that is not on the allow-list."""
    value: int = 42


# ---------------------------------------------------------------------------
# 8 — Arbitrary user-defined / third-party class is blocked
# ---------------------------------------------------------------------------

class TestUnsupportedClasses:
    def test_user_defined_class_is_blocked(self) -> None:
        """
        A pickle of a user-defined class must raise RestrictedUnpicklingError.
        We serialise with the standard pickler and load with _SafeUnpickler.
        """
        obj = _ArbitraryClass()
        data = _pickle_bytes(obj)

        with pytest.raises(RestrictedUnpicklingError):
            _load_from_bytes(data)

    def test_numpy_array_is_blocked(self) -> None:
        """
        NumPy arrays are NOT on the allow-list.  Face Lock stores embeddings
        as plain Python lists of floats, so NumPy arrays should never appear
        in a valid .pkl file.  Reject rather than silently accept.
        """
        import numpy as np

        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        data = _pickle_bytes(arr)

        with pytest.raises(RestrictedUnpicklingError):
            _load_from_bytes(data)

    def test_pathlib_path_is_blocked(self) -> None:
        """pathlib.Path is not on the allow-list."""
        data = _pickle_bytes(Path("/tmp/test"))

        with pytest.raises(RestrictedUnpicklingError):
            _load_from_bytes(data)


# ---------------------------------------------------------------------------
# 9 & 10 — Missing-file behaviour (load_pickle API contract)
# ---------------------------------------------------------------------------

class TestMissingFile:
    def test_missing_file_without_default_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does_not_exist.pkl"
        with pytest.raises(FileNotFoundError, match="Pickle file not found"):
            load_pickle(missing)

    def test_missing_file_with_default_returns_default(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.pkl"
        result = load_pickle(missing, default=[])
        assert result == []

    def test_missing_file_with_none_default_returns_none(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.pkl"
        result = load_pickle(missing, default=None)
        assert result is None


# ---------------------------------------------------------------------------
# 11 — Exception hierarchy (caller compatibility)
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_restricted_unpickling_error_is_value_error(self) -> None:
        """
        RestrictedUnpicklingError must be a subclass of ValueError so that
        callers in ai_engine.py which catch ValueError continue to work.
        """
        assert issubclass(RestrictedUnpicklingError, ValueError)

    def test_restricted_unpickling_error_can_be_raised_and_caught_as_value_error(
        self,
    ) -> None:
        with pytest.raises(ValueError):
            raise RestrictedUnpicklingError("blocked type")

    def test_error_message_contains_module_and_name(self) -> None:
        """The error message must identify the blocked type for diagnostics."""
        data = _pickle_bytes(Path("/tmp"))  # pathlib.Path is blocked

        with pytest.raises(RestrictedUnpicklingError) as exc_info:
            _load_from_bytes(data)

        msg = str(exc_info.value)
        assert "pathlib" in msg
        assert "Path" in msg
