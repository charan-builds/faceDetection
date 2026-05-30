"""
Confusion-matrix helpers for binary PAD evaluation.

Positive class: attack (spoof). Negative class: bona fide (live).

Predicted outcomes are normalized with fail_closed_v1 before counting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from eval.harness.outcome_mapper import (
    POLICY_FAIL_CLOSED_V1,
    PolicyName,
    is_classified_as_live,
    validate_outcome,
)

GroundTruth = Literal["attack", "bona_fide"]
LabeledPresentation = tuple[GroundTruth, str]


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Raw confusion matrix cell counts."""

    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn


def build_confusion_counts(
    presentations: Sequence[LabeledPresentation],
    *,
    policy: PolicyName = POLICY_FAIL_CLOSED_V1,
) -> ConfusionCounts:
    """Build tp/tn/fp/fn from (ground_truth, predicted_outcome) pairs."""
    if policy != POLICY_FAIL_CLOSED_V1:
        raise ValueError(f"Unsupported evaluation policy: {policy!r}")

    tp_count = tn_count = fp_count = fn_count = 0
    for ground_truth, predicted in presentations:
        if ground_truth not in ("attack", "bona_fide"):
            raise ValueError(
                f"Invalid ground_truth: {ground_truth!r}. "
                "Expected 'attack' or 'bona_fide'."
            )
        validate_outcome(predicted)
        predicted_live = is_classified_as_live(predicted, policy=policy)

        if ground_truth == "attack":
            if predicted_live:
                fn_count += 1
            else:
                tp_count += 1
        else:
            if predicted_live:
                tn_count += 1
            else:
                fp_count += 1

    return ConfusionCounts(tp=tp_count, tn=tn_count, fp=fp_count, fn=fn_count)


def tp(counts: ConfusionCounts) -> int:
    """True positives: attack correctly classified as not live."""
    return counts.tp


def tn(counts: ConfusionCounts) -> int:
    """True negatives: bona fide correctly classified as live."""
    return counts.tn


def fp(counts: ConfusionCounts) -> int:
    """False positives: bona fide classified as not live (BPCER errors)."""
    return counts.fp


def fn(counts: ConfusionCounts) -> int:
    """False negatives: attack classified as live (APCER errors)."""
    return counts.fn


def accuracy(counts: ConfusionCounts) -> float:
    """(tp + tn) / total."""
    if counts.total == 0:
        raise ValueError("Cannot compute accuracy: empty confusion matrix.")
    return (counts.tp + counts.tn) / counts.total


def precision(counts: ConfusionCounts) -> float:
    """tp / (tp + fp)."""
    denom = counts.tp + counts.fp
    if denom == 0:
        raise ValueError("Cannot compute precision: tp + fp is zero.")
    return counts.tp / denom


def recall(counts: ConfusionCounts) -> float:
    """tp / (tp + fn) — attack detection rate (1 - APCER on attacks only)."""
    denom = counts.tp + counts.fn
    if denom == 0:
        raise ValueError("Cannot compute recall: tp + fn is zero.")
    return counts.tp / denom


def f1_score(counts: ConfusionCounts) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(counts)
    r = recall(counts)
    if p + r == 0:
        raise ValueError("Cannot compute F1: precision + recall is zero.")
    return 2 * p * r / (p + r)
