"""
Operator authentication for the Face Lock AI project.

Protects administrative operations (user registration) with a PIN.

Security model
--------------
The operator PIN is never stored in plaintext. The expected value is kept as a
SHA-256 hex digest in the environment variable ``FACE_LOCK_OPERATOR_PIN_HASH``.
At runtime the entered PIN is hashed and the digests are compared with
``hmac.compare_digest``, which is constant-time and prevents timing attacks.

Modes
-----
Demo mode (env var absent):
    - No PIN prompt.
    - A WARNING is logged on every protected operation.
    - Suitable for local development only — not for any deployed use.

Configured mode (env var present):
    - The operator is prompted for their PIN via stdin.
    - Access is granted only if sha256(entered_pin) == stored_hash.
    - Three consecutive failures are logged at ERROR level.

Setting the PIN
---------------
Generate the hash and set the env var before starting the application:

    python -c "import hashlib; print(hashlib.sha256(b'your_pin').hexdigest())"

Then in your shell (or .env file — never commit the hash to source control):

    set FACE_LOCK_OPERATOR_PIN_HASH=<hash>   # Windows CMD
    $env:FACE_LOCK_OPERATOR_PIN_HASH=<hash>  # Windows PowerShell
    export FACE_LOCK_OPERATOR_PIN_HASH=<hash> # bash / zsh
"""

from __future__ import annotations

import getpass
import hashlib
import hmac
import logging
import os

try:
    from src.logging_config import get_logger, log_event
except ModuleNotFoundError:
    from logging_config import get_logger, log_event


# Environment variable that carries the expected SHA-256 hex digest.
ENV_PIN_HASH = "FACE_LOCK_OPERATOR_PIN_HASH"

# Maximum number of PIN attempts before a hard deny is returned.
MAX_PIN_ATTEMPTS = 3

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sha256_hash_pin(pin: str) -> str:
    """
    Return the SHA-256 hex digest of a PIN string (UTF-8 encoded).

    Args:
        pin: The raw PIN entered by the operator.

    Returns:
        A 64-character lowercase hex string.

    Example:
        >>> sha256_hash_pin("1234")
        '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4'
    """
    # Encode as UTF-8 bytes before hashing — consistent across platforms.
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


def is_demo_mode() -> bool:
    """
    Return True when no operator PIN hash is configured in the environment.

    Demo mode allows all operations without a PIN prompt, but always logs a
    WARNING so the absence of auth is visible in audit logs.

    Returns:
        True if ``FACE_LOCK_OPERATOR_PIN_HASH`` is absent or empty.
    """
    stored_hash = os.environ.get(ENV_PIN_HASH, "").strip()
    return stored_hash == ""


def verify_operator_pin() -> bool:
    """
    Verify the operator's identity before a protected operation.

    Behaviour
    ---------
    - Demo mode (env var absent / empty):
        Logs a WARNING and returns True unconditionally.

    - Configured mode (env var present):
        Prompts for the PIN up to ``MAX_PIN_ATTEMPTS`` times.
        Returns True only when sha256(entered_pin) == stored_hash.
        Returns False after all attempts are exhausted.

    Returns:
        True if the operator is authorised (or demo mode is active).
        False if every PIN attempt failed.
    """
    if is_demo_mode():
        log_event(
            LOGGER,
            "operator_auth_demo_mode",
            level=logging.WARNING,
            env_var=ENV_PIN_HASH,
            message="No operator PIN configured. Running in demo mode — "
                    "registration is unprotected.",
        )
        return True

    stored_hash = os.environ.get(ENV_PIN_HASH, "").strip()

    for attempt in range(1, MAX_PIN_ATTEMPTS + 1):
        # getpass hides the PIN while typing (no echo to terminal).
        try:
            entered_pin = getpass.getpass("Operator PIN: ")
        except (EOFError, KeyboardInterrupt):
            # Non-interactive environment or user aborted — deny access.
            print()  # newline after the hidden prompt
            log_event(
                LOGGER,
                "operator_auth_failure",
                level=logging.ERROR,
                reason="input_aborted",
                attempt=attempt,
            )
            return False

        entered_hash = sha256_hash_pin(entered_pin)

        # hmac.compare_digest is constant-time — prevents timing side-channels.
        if hmac.compare_digest(entered_hash, stored_hash):
            log_event(
                LOGGER,
                "operator_auth_success",
                level=logging.INFO,
                attempt=attempt,
            )
            return True

        remaining = MAX_PIN_ATTEMPTS - attempt
        if remaining > 0:
            log_event(
                LOGGER,
                "operator_auth_failure",
                level=logging.WARNING,
                reason="wrong_pin",
                attempt=attempt,
                remaining_attempts=remaining,
            )
            print(f"Incorrect PIN. {remaining} attempt(s) remaining.")
        else:
            log_event(
                LOGGER,
                "operator_auth_failure",
                level=logging.ERROR,
                reason="max_attempts_exceeded",
                attempt=attempt,
                remaining_attempts=0,
            )

    return False
