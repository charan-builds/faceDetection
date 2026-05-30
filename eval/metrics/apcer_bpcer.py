"""
APCER and BPCER calculators for Face Lock PAD evaluation.

APCER: attack presentations classified as bona fide (LIVE under fail_closed_v1).
BPCER: bona fide presentations classified as not live.
"""

from __future__ import annotations

from collections.abc import Sequence

from eval.harness.outcome_mapper import (
    POLICY_FAIL_CLOSED_V1,
    PolicyName,
    classify_for_apcer,
    classify_for_bpcer,
    validate_outcome,
)


class InsufficientSamplesError(ValueError):
    """Raised when a metric denominator is zero."""


def to_percentage(rate: float, *, digits: int = 4) -> float:
    """
    Convert a unit-interval rate (0.0–1.0) to a percentage (0.0–100.0).

    Example: 0.05 -> 5.0
    """
    return round(rate * 100.0, digits)


def compute_apcer(
    attack_predicted_outcomes: Sequence[str],
    *,
    policy: PolicyName = POLICY_FAIL_CLOSED_V1,
) -> float:
    """
    APCER = (# attack presentations classified as LIVE) / (# attack presentations).

    Returns a float in [0.0, 1.0]. Use to_percentage() for display.
    """
    if policy != POLICY_FAIL_CLOSED_V1:
        raise ValueError(f"Unsupported evaluation policy: {policy!r}")

    total = len(attack_predicted_outcomes)
    if total == 0:
        raise InsufficientSamplesError(
            "Cannot compute APCER: no attack presentations provided."
        )

    false_accepts = sum(
        1
        for outcome in attack_predicted_outcomes
        if classify_for_apcer(validate_outcome(outcome))
    )
    return false_accepts / total


def compute_bpcer(
    bona_fide_predicted_outcomes: Sequence[str],
    *,
    policy: PolicyName = POLICY_FAIL_CLOSED_V1,
) -> float:
    """
    BPCER = (# bona fide presentations not classified as LIVE) / (# bona fide).

    ATTACK, INCONCLUSIVE, and ERROR count as errors (fail_closed_v1).
    Returns a float in [0.0, 1.0].
    """
    if policy != POLICY_FAIL_CLOSED_V1:
        raise ValueError(f"Unsupported evaluation policy: {policy!r}")

    total = len(bona_fide_predicted_outcomes)
    if total == 0:
        raise InsufficientSamplesError(
            "Cannot compute BPCER: no bona fide presentations provided."
        )

    false_rejects = sum(
        1
        for outcome in bona_fide_predicted_outcomes
        if classify_for_bpcer(validate_outcome(outcome))
    )
    return false_rejects / total
