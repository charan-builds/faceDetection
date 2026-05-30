"""Tests for eval.metrics.apcer_bpcer."""

from __future__ import annotations

import pytest

from eval.harness.outcome_mapper import (
    OUTCOME_ATTACK,
    OUTCOME_ERROR,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_LIVE,
)
from eval.metrics.apcer_bpcer import (
    InsufficientSamplesError,
    compute_apcer,
    compute_bpcer,
    to_percentage,
)


class TestToPercentage:
    def test_zero(self) -> None:
        assert to_percentage(0.0) == 0.0

    def test_five_percent(self) -> None:
        assert to_percentage(0.05) == 5.0

    def test_one_hundred_percent(self) -> None:
        assert to_percentage(1.0) == 100.0

    def test_rounding(self) -> None:
        assert to_percentage(1 / 3, digits=2) == 33.33


class TestComputeApcer:
    def test_all_attack_correctly_rejected(self) -> None:
        outcomes = [OUTCOME_ATTACK, OUTCOME_INCONCLUSIVE, OUTCOME_ERROR]
        assert compute_apcer(outcomes) == 0.0

    def test_all_false_accepts(self) -> None:
        outcomes = [OUTCOME_LIVE, OUTCOME_LIVE]
        assert compute_apcer(outcomes) == 1.0

    def test_mixed_apcer(self) -> None:
        # 1 LIVE false accept out of 4 attacks
        outcomes = [OUTCOME_LIVE, OUTCOME_ATTACK, OUTCOME_INCONCLUSIVE, OUTCOME_ERROR]
        assert compute_apcer(outcomes) == 0.25
        assert to_percentage(compute_apcer(outcomes)) == 25.0

    def test_inconclusive_and_error_not_counted_as_live(self) -> None:
        outcomes = [OUTCOME_INCONCLUSIVE, OUTCOME_ERROR]
        assert compute_apcer(outcomes) == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(InsufficientSamplesError, match="no attack"):
            compute_apcer([])

    def test_invalid_outcome_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid PAD outcome"):
            compute_apcer(["BOGUS"])

    def test_unsupported_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported evaluation policy"):
            compute_apcer([OUTCOME_ATTACK], policy="legacy")  # type: ignore[arg-type]


class TestComputeBpcer:
    def test_all_bona_fide_accepted(self) -> None:
        assert compute_bpcer([OUTCOME_LIVE, OUTCOME_LIVE]) == 0.0

    def test_all_bona_fide_rejected(self) -> None:
        outcomes = [OUTCOME_ATTACK, OUTCOME_INCONCLUSIVE, OUTCOME_ERROR]
        assert compute_bpcer(outcomes) == 1.0

    def test_mixed_bpcer(self) -> None:
        outcomes = [OUTCOME_LIVE, OUTCOME_ATTACK, OUTCOME_LIVE, OUTCOME_INCONCLUSIVE]
        assert compute_bpcer(outcomes) == 0.5

    def test_empty_raises(self) -> None:
        with pytest.raises(InsufficientSamplesError, match="no bona fide"):
            compute_bpcer([])

    def test_invalid_outcome_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid PAD outcome"):
            compute_bpcer([OUTCOME_LIVE, ""])

    def test_unsupported_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported evaluation policy"):
            compute_bpcer([OUTCOME_LIVE], policy="legacy")  # type: ignore[arg-type]


class TestBoundaryRates:
    def test_single_attack_live(self) -> None:
        assert compute_apcer([OUTCOME_LIVE]) == 1.0

    def test_single_attack_reject(self) -> None:
        assert compute_apcer([OUTCOME_ATTACK]) == 0.0

    def test_single_bona_fide_live(self) -> None:
        assert compute_bpcer([OUTCOME_LIVE]) == 0.0

    def test_single_bona_fide_reject(self) -> None:
        assert compute_bpcer([OUTCOME_INCONCLUSIVE]) == 1.0
