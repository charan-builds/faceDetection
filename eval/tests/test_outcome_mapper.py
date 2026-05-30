"""Tests for eval.harness.outcome_mapper."""

from __future__ import annotations

import pytest

from eval.harness.outcome_mapper import (
    OUTCOME_ATTACK,
    OUTCOME_ERROR,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_LIVE,
    classify_for_apcer,
    classify_for_bpcer,
    is_classified_as_live,
    is_classified_as_not_live,
    map_outcome_fail_closed_v1,
    normalize_outcome,
    validate_outcome,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("LIVE", OUTCOME_LIVE),
        ("live", OUTCOME_LIVE),
        ("ATTACK", OUTCOME_ATTACK),
        ("INCONCLUSIVE", OUTCOME_INCONCLUSIVE),
        ("ERROR", OUTCOME_ERROR),
    ],
)
def test_validate_outcome_accepts_valid(raw: str, expected: str) -> None:
    assert validate_outcome(raw) == expected


@pytest.mark.parametrize("invalid", ["", "UNKNOWN", "live-ish", None, 42])
def test_validate_outcome_rejects_invalid(invalid) -> None:
    with pytest.raises(ValueError, match="Invalid PAD outcome"):
        validate_outcome(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("outcome", "normalized"),
    [
        (OUTCOME_LIVE, OUTCOME_LIVE),
        (OUTCOME_ATTACK, OUTCOME_ATTACK),
        (OUTCOME_INCONCLUSIVE, OUTCOME_ATTACK),
        (OUTCOME_ERROR, OUTCOME_ATTACK),
    ],
)
def test_map_outcome_fail_closed_v1(outcome: str, normalized: str) -> None:
    assert map_outcome_fail_closed_v1(outcome) == normalized
    assert normalize_outcome(outcome) == normalized


def test_normalize_outcome_unsupported_policy() -> None:
    with pytest.raises(ValueError, match="Unsupported evaluation policy"):
        normalize_outcome(OUTCOME_LIVE, policy="other_policy")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("outcome", "is_live"),
    [
        (OUTCOME_LIVE, True),
        (OUTCOME_ATTACK, False),
        (OUTCOME_INCONCLUSIVE, False),
        (OUTCOME_ERROR, False),
    ],
)
def test_is_classified_as_live(outcome: str, is_live: bool) -> None:
    assert is_classified_as_live(outcome) is is_live
    assert is_classified_as_not_live(outcome) is (not is_live)


@pytest.mark.parametrize(
    ("outcome", "counts_for_apcer"),
    [
        (OUTCOME_LIVE, True),
        (OUTCOME_ATTACK, False),
        (OUTCOME_INCONCLUSIVE, False),
        (OUTCOME_ERROR, False),
    ],
)
def test_classify_for_apcer(outcome: str, counts_for_apcer: bool) -> None:
    assert classify_for_apcer(outcome) is counts_for_apcer


@pytest.mark.parametrize(
    ("outcome", "counts_for_bpcer"),
    [
        (OUTCOME_LIVE, False),
        (OUTCOME_ATTACK, True),
        (OUTCOME_INCONCLUSIVE, True),
        (OUTCOME_ERROR, True),
    ],
)
def test_classify_for_bpcer(outcome: str, counts_for_bpcer: bool) -> None:
    assert classify_for_bpcer(outcome) is counts_for_bpcer
