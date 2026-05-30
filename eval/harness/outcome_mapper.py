"""
Pure outcome mapping for PAD evaluation (fail_closed_v1).

No file I/O, no pad_engine imports, no external dependencies.
Aligned with FL-PAD-EVAL-001 and src.pad_engine.PadOutcome string values.
"""

from __future__ import annotations

from typing import Final, Literal

PolicyName = Literal["fail_closed_v1"]

POLICY_FAIL_CLOSED_V1: Final[PolicyName] = "fail_closed_v1"

OUTCOME_LIVE: Final = "LIVE"
OUTCOME_ATTACK: Final = "ATTACK"
OUTCOME_INCONCLUSIVE: Final = "INCONCLUSIVE"
OUTCOME_ERROR: Final = "ERROR"

VALID_OUTCOMES: Final[frozenset[str]] = frozenset(
    {OUTCOME_LIVE, OUTCOME_ATTACK, OUTCOME_INCONCLUSIVE, OUTCOME_ERROR}
)

NormalizedOutcome = Literal["LIVE", "ATTACK"]


def validate_outcome(outcome: str) -> str:
    """Return outcome if valid; raise ValueError otherwise."""
    normalized = outcome.strip().upper() if isinstance(outcome, str) else ""
    if normalized not in VALID_OUTCOMES:
        raise ValueError(
            f"Invalid PAD outcome: {outcome!r}. "
            f"Expected one of: {', '.join(sorted(VALID_OUTCOMES))}"
        )
    return normalized


def map_outcome_fail_closed_v1(outcome: str) -> NormalizedOutcome:
    """
    Collapse INCONCLUSIVE and ERROR to ATTACK; LIVE and ATTACK unchanged.

    Used before APCER / confusion-matrix calculations so ambiguous or failed
    runs are never treated as bona fide acceptance.
    """
    validated = validate_outcome(outcome)
    if validated in (OUTCOME_INCONCLUSIVE, OUTCOME_ERROR):
        return OUTCOME_ATTACK
    return validated  # type: ignore[return-value]


def normalize_outcome(
    outcome: str,
    policy: PolicyName = POLICY_FAIL_CLOSED_V1,
) -> NormalizedOutcome:
    """Map a raw PadOutcome string to a binary LIVE | ATTACK bucket."""
    if policy != POLICY_FAIL_CLOSED_V1:
        raise ValueError(f"Unsupported evaluation policy: {policy!r}")
    return map_outcome_fail_closed_v1(outcome)


def is_classified_as_live(
    outcome: str,
    policy: PolicyName = POLICY_FAIL_CLOSED_V1,
) -> bool:
    """True only when the fail-closed normalized outcome is LIVE."""
    return normalize_outcome(outcome, policy=policy) == OUTCOME_LIVE


def is_classified_as_not_live(
    outcome: str,
    policy: PolicyName = POLICY_FAIL_CLOSED_V1,
) -> bool:
    """True when normalized outcome is ATTACK (includes INCONCLUSIVE, ERROR)."""
    return not is_classified_as_live(outcome, policy=policy)


def classify_for_apcer(predicted_outcome: str) -> bool:
    """
    Return True if this attack presentation counts toward APCER (false accept).

    Under fail_closed_v1 only LIVE counts as bona-fide acceptance of an attack.
    """
    return is_classified_as_live(predicted_outcome)


def classify_for_bpcer(predicted_outcome: str) -> bool:
    """
    Return True if this bona fide presentation counts toward BPCER (false reject).

    Under fail_closed_v1 any outcome other than LIVE is a bona fide error
    (ATTACK, INCONCLUSIVE, ERROR per protocol).
    """
    validate_outcome(predicted_outcome)
    return predicted_outcome != OUTCOME_LIVE
