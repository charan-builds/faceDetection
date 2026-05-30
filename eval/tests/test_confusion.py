"""Tests for eval.metrics.confusion."""

from __future__ import annotations

import pytest

from eval.harness.outcome_mapper import (
    OUTCOME_ATTACK,
    OUTCOME_ERROR,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_LIVE,
)
from eval.metrics.confusion import (
    ConfusionCounts,
    accuracy,
    build_confusion_counts,
    f1_score,
    fn,
    fp,
    precision,
    recall,
    tn,
    tp,
)


def test_perfect_classifier() -> None:
    presentations = [
        ("attack", OUTCOME_ATTACK),
        ("attack", OUTCOME_INCONCLUSIVE),
        ("bona_fide", OUTCOME_LIVE),
        ("bona_fide", OUTCOME_LIVE),
    ]
    counts = build_confusion_counts(presentations)
    assert counts == ConfusionCounts(tp=2, tn=2, fp=0, fn=0)
    assert accuracy(counts) == 1.0
    assert precision(counts) == 1.0
    assert recall(counts) == 1.0
    assert f1_score(counts) == 1.0


def test_worst_classifier() -> None:
    presentations = [
        ("attack", OUTCOME_LIVE),
        ("bona_fide", OUTCOME_ATTACK),
    ]
    counts = build_confusion_counts(presentations)
    assert tp(counts) == 0
    assert tn(counts) == 0
    assert fp(counts) == 1
    assert fn(counts) == 1
    assert accuracy(counts) == 0.0


def test_inconclusive_on_attack_is_tp_not_fn() -> None:
    counts = build_confusion_counts([("attack", OUTCOME_INCONCLUSIVE)])
    assert tp(counts) == 1
    assert fn(counts) == 0


def test_error_on_bona_fide_is_fp() -> None:
    counts = build_confusion_counts([("bona_fide", OUTCOME_ERROR)])
    assert fp(counts) == 1
    assert tn(counts) == 0


def test_empty_presentations() -> None:
    counts = build_confusion_counts([])
    assert counts.total == 0
    with pytest.raises(ValueError, match="empty confusion"):
        accuracy(counts)


def test_invalid_ground_truth() -> None:
    with pytest.raises(ValueError, match="Invalid ground_truth"):
        build_confusion_counts([("spoof", OUTCOME_ATTACK)])


def test_invalid_predicted_outcome() -> None:
    with pytest.raises(ValueError, match="Invalid PAD outcome"):
        build_confusion_counts([("attack", "MAYBE")])


def test_precision_division_by_zero() -> None:
    counts = ConfusionCounts(tp=0, tn=5, fp=0, fn=0)
    with pytest.raises(ValueError, match="precision"):
        precision(counts)


def test_recall_division_by_zero() -> None:
    counts = ConfusionCounts(tp=0, tn=5, fp=0, fn=0)
    with pytest.raises(ValueError, match="recall"):
        recall(counts)


def test_f1_division_by_zero() -> None:
    counts = ConfusionCounts(tp=0, tn=0, fp=0, fn=0)
    with pytest.raises(ValueError, match="accuracy"):
        accuracy(counts)


def test_derived_metrics_mixed() -> None:
    presentations = [
        ("attack", OUTCOME_ATTACK),
        ("attack", OUTCOME_LIVE),
        ("attack", OUTCOME_ERROR),
        ("bona_fide", OUTCOME_LIVE),
        ("bona_fide", OUTCOME_INCONCLUSIVE),
        ("bona_fide", OUTCOME_ATTACK),
    ]
    counts = build_confusion_counts(presentations)
    # tp=2 (attack+attack, attack+error), fn=1 (attack+live)
    # tn=1 (bf+live), fp=2 (bf+inconclusive, bf+attack)
    assert tp(counts) == 2
    assert fn(counts) == 1
    assert tn(counts) == 1
    assert fp(counts) == 2
    assert accuracy(counts) == pytest.approx(3 / 6)
    assert recall(counts) == pytest.approx(2 / 3)
    assert precision(counts) == pytest.approx(2 / 4)
    assert f1_score(counts) == pytest.approx(2 * (2 / 3) * (2 / 4) / ((2 / 3) + (2 / 4)))


def test_unsupported_policy() -> None:
    with pytest.raises(ValueError, match="Unsupported evaluation policy"):
        build_confusion_counts(
            [("attack", OUTCOME_ATTACK)],
            policy="other",  # type: ignore[arg-type]
        )
