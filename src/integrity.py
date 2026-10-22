"""
Embedding integrity protection for the Face Lock AI project.

Addresses SEC-003: an attacker with write access to ``encodings/`` can swap
embedding files and impersonate any user without a face match.  This module
adds HMAC-SHA256 signatures so tampered or forged files are detected before
their data is trusted.

Security model
--------------
- A secret key is read from the environment variable ``FACE_LOCK_SIGNING_KEY``.
- ``sign_record(data)`` serialises the record to canonical JSON (keys sorted,
  ASCII-safe) and returns ``hmac.new(key, payload, sha256).hexdigest()``.
- ``verify_record(data, signature)`` re-derives the expected signature and
  compares with ``hmac.compare_digest`` (constant-time, no timing leak).
- The signature travels alongside the record inside the same ``.pkl`` file:
  ``{"record": {...}, "signature": "<64-char hex>"}``

Modes
-----
Unsigned mode (env var absent / empty):
    - ``sign_record`` returns the empty string ``""`` as a sentinel.
    - ``verify_record`` returns ``(True, "unsigned_mode")``.
    - Every save and load emits a WARNING so the absence of signing is auditable.
    - Suitable for local development only.

Signed mode (env var present):
    - Every save attaches an HMAC-SHA256 signature.
    - Every load verifies the signature before the record is trusted.
    - A missing or incorrect signature raises ``IntegrityError``.

Backward compatibility
----------------------
Legacy ``.pkl`` files written before this module existed are plain dicts with
an ``"embeddings"`` key and no ``"record"`` / ``"signature"`` wrapper.
``unwrap_stored_payload`` detects this shape and returns the flat dict with a
``_legacy=True`` marker so callers can warn and decide whether to re-sign.

Setting the signing key
-----------------------
Generate a strong random key once and store it outside source control:

    python -c "import secrets; print(secrets.token_hex(32))"

Then in your shell (or a secrets manager — never commit to VCS):

    set FACE_LOCK_SIGNING_KEY=<hex>    # Windows CMD
    $env:FACE_LOCK_SIGNING_KEY=<hex>   # Windows PowerShell
    export FACE_LOCK_SIGNING_KEY=<hex> # bash / zsh
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any

try:
    from src.logging_config import get_logger, log_event
except ModuleNotFoundError:
    from logging_config import get_logger, log_event


# Environment variable that carries the HMAC signing key (raw hex string).
ENV_SIGNING_KEY = "FACE_LOCK_SIGNING_KEY"

# Environment variable that enforces strict security mode (1 or true).
ENV_STRICT_MODE = "FACE_LOCK_STRICT_MODE"

# Sentinel returned by sign_record() when unsigned mode is active.
_UNSIGNED_SENTINEL = ""

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IntegrityError(ValueError):
    """
    Raised when an embedding file fails signature verification.

    Inherits from ValueError so existing callers that catch ValueError
    (schema validation in ai_engine) continue to behave correctly.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_signing_key() -> bytes | None:
    """
    Read the signing key from the environment.

    Returns:
        The key as UTF-8 bytes, or None when the env var is absent / empty.
    """
    raw = os.environ.get(ENV_SIGNING_KEY, "").strip()
    if not raw:
        return None
    return raw.encode("utf-8")


def _canonical_bytes(data: dict[str, Any]) -> bytes:
    """
    Serialise a dict to a canonical, deterministic JSON byte string.

    Keys are sorted recursively so the same logical record always produces the
    same bytes regardless of dict insertion order.  Non-ASCII characters are
    escaped so the output is platform-independent.

    Args:
        data: The embedding record dict to serialise.

    Returns:
        UTF-8 encoded JSON bytes with sorted keys and no extra whitespace.
    """
    return json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_signing_configured() -> bool:
    """
    Return True when a signing key is present in the environment.

    Returns:
        False when ``FACE_LOCK_SIGNING_KEY`` is absent or empty (unsigned mode).
    """
    return _get_signing_key() is not None


def is_strict_mode() -> bool:
    """
    Return True if strict security mode is enabled.

    Strict mode requires:
    - FACE_LOCK_SIGNING_KEY to be configured for all signing and verification.
    - All loaded templates to have valid HMAC signatures (legacy unsigned files rejected).
    - Operator PIN to be configured when operator authentication is enforced.
    """
    return os.environ.get(ENV_STRICT_MODE, "").strip().lower() in ("1", "true")


def enforce_strict_mode_policy(*, require_operator_auth: bool = False) -> None:
    """
    Enforce security requirements when strict mode is active.

    When FACE_LOCK_STRICT_MODE is enabled:
    - FACE_LOCK_SIGNING_KEY must be configured.
    - If require_operator_auth is True, operator authentication must not be in demo mode
      (i.e. FACE_LOCK_OPERATOR_PIN_HASH must be configured).

    Raises:
        RuntimeError: If strict mode is active but required configuration is missing.
    """
    if not is_strict_mode():
        return

    try:
        from src.operator_auth import is_demo_mode
    except ModuleNotFoundError:
        from operator_auth import is_demo_mode

    missing: list[str] = []
    if not is_signing_configured():
        missing.append("FACE_LOCK_SIGNING_KEY")
    if require_operator_auth and is_demo_mode():
        missing.append("FACE_LOCK_OPERATOR_PIN_HASH")

    if missing:
        raise RuntimeError(
            "Strict mode is enabled, but operator PIN or signing key is missing. "
            "Set FACE_LOCK_OPERATOR_PIN_HASH and FACE_LOCK_SIGNING_KEY."
        )


def sign_record(data: dict[str, Any]) -> str:
    """
    Compute an HMAC-SHA256 signature for an embedding record dict.

    The signature is derived from a canonical JSON serialisation of *data* so
    that any field modification — even a single float in the embeddings list —
    invalidates the signature.

    Args:
        data: The embedding record dict (output of ``create_embedding_record``).

    Returns:
        A 64-character lowercase hex HMAC digest, or ``""`` in unsigned mode.

    Raises:
        IntegrityError: If strict mode is enabled but FACE_LOCK_SIGNING_KEY is missing.
    """
    key = _get_signing_key()

    if key is None:
        if is_strict_mode():
            raise IntegrityError(
                "Strict mode is enabled, but FACE_LOCK_SIGNING_KEY is missing. "
                "Cannot sign embedding record."
            )
        # Unsigned mode: emit a warning so the absence of signing is visible.
        log_event(
            LOGGER,
            "integrity_unsigned_save",
            level=logging.WARNING,
            env_var=ENV_SIGNING_KEY,
            user_name=data.get("user_name", "unknown"),
            message="No signing key configured. Embedding saved without integrity protection.",
        )
        return _UNSIGNED_SENTINEL

    payload = _canonical_bytes(data)
    digest = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return digest


def verify_record(data: dict[str, Any], signature: str) -> tuple[bool, str]:
    """
    Verify that an embedding record has not been tampered with.

    Args:
        data:      The record dict exactly as it was stored.
        signature: The hex digest stored alongside the record.

    Returns:
        A ``(valid, reason_code)`` tuple:
        - ``(True,  "signature_valid")``   — record passed verification.
        - ``(True,  "unsigned_mode")``     — no key configured; always passes.
        - ``(False, "missing_signature")`` — signed mode but no sig stored.
        - ``(False, "missing_signing_key")``— strict mode but no key configured.
        - ``(False, "signature_mismatch")``— record has been modified.

    Note:
        This function never raises; the caller decides how to handle failures.
        ``load_user_embeddings`` converts False into an ``IntegrityError``.
    """
    key = _get_signing_key()

    if key is None:
        if is_strict_mode():
            log_event(
                LOGGER,
                "integrity_check_failed",
                level=logging.ERROR,
                reason="missing_signing_key_strict_mode",
                user_name=data.get("user_name", "unknown"),
                message="Strict mode enabled but FACE_LOCK_SIGNING_KEY is missing.",
            )
            return False, "missing_signing_key"

        # No key configured — unsigned mode is active; skip verification.
        log_event(
            LOGGER,
            "integrity_unsigned_load",
            level=logging.WARNING,
            env_var=ENV_SIGNING_KEY,
            user_name=data.get("user_name", "unknown"),
            message="No signing key configured. Integrity check skipped.",
        )
        return True, "unsigned_mode"

    # A signed-mode load of a file with no signature is a clear integrity failure.
    if not signature:
        log_event(
            LOGGER,
            "integrity_check_failed",
            level=logging.ERROR,
            reason="missing_signature",
            user_name=data.get("user_name", "unknown"),
        )
        return False, "missing_signature"

    expected = hmac.new(key, _canonical_bytes(data), hashlib.sha256).hexdigest()

    # compare_digest is constant-time — prevents timing side-channel attacks.
    if hmac.compare_digest(expected, signature):
        log_event(
            LOGGER,
            "integrity_check_passed",
            level=logging.DEBUG,
            user_name=data.get("user_name", "unknown"),
        )
        return True, "signature_valid"

    log_event(
        LOGGER,
        "integrity_check_failed",
        level=logging.ERROR,
        reason="signature_mismatch",
        user_name=data.get("user_name", "unknown"),
    )
    return False, "signature_mismatch"


def wrap_record_for_storage(record: dict[str, Any]) -> dict[str, Any]:
    """
    Wrap an embedding record with its HMAC signature ready for pickling.

    The on-disk structure is::

        {
            "record":    { ...embedding record... },
            "signature": "<64-char hex or empty string>",
        }

    Args:
        record: The embedding record dict to protect.

    Returns:
        A dict with ``"record"`` and ``"signature"`` keys.
    """
    signature = sign_record(record)
    return {"record": record, "signature": signature}


def unwrap_stored_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], str, bool]:
    """
    Extract the record and signature from a stored payload.

    Handles three shapes:
    1. New signed format:   ``{"record": {...}, "signature": "..."}``
    2. New unsigned format: ``{"record": {...}, "signature": ""}``
    3. Legacy flat format:  ``{"user_name": ..., "embeddings": [...], ...}``
       — detected by the absence of a ``"record"`` key.

    Args:
        payload: The raw dict loaded from the ``.pkl`` file.

    Returns:
        A three-tuple ``(record, signature, is_legacy)``:
        - ``record``    — the inner embedding record dict.
        - ``signature`` — the stored signature string (``""`` for legacy/unsigned).
        - ``is_legacy`` — True when the file was written before signing was added.
    """
    if "record" in payload and "signature" in payload:
        # New format — normal path.
        return payload["record"], payload["signature"], False

    # Legacy flat format: the payload IS the record.
    return payload, _UNSIGNED_SENTINEL, True
