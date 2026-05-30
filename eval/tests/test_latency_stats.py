"""Tests for eval.metrics.latency_stats."""

from __future__ import annotations

import math

import pytest

from eval.metrics.latency_stats import LatencyStats, compute_latency_stats


class TestComputeLatencyStats:
    def test_single_sample(self) -> None:
        stats = compute_latency_stats([42.0])
        assert stats == LatencyStats(
            mean=42.0,
            median=42.0,
            min=42.0,
            max=42.0,
            p95=42.0,
            p99=42.0,
            count=1,
        )

    def test_known_distribution(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_latency_stats(values)
        assert stats.count == 5
        assert stats.mean == 30.0
        assert stats.median == 30.0
        assert stats.min == 10.0
        assert stats.max == 50.0
        assert stats.p95 == pytest.approx(48.0)
        assert stats.p99 == pytest.approx(49.6)

    def test_unsorted_input(self) -> None:
        stats = compute_latency_stats([100.0, 10.0, 50.0])
        assert stats.min == 10.0
        assert stats.max == 100.0
        assert stats.median == 50.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty input"):
            compute_latency_stats([])

    def test_non_finite_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            compute_latency_stats([1.0, float("nan")])

        with pytest.raises(ValueError, match="non-finite"):
            compute_latency_stats([float("inf")])

    def test_p95_p99_on_larger_sample(self) -> None:
        values = [float(i) for i in range(1, 101)]
        stats = compute_latency_stats(values)
        assert stats.p95 == pytest.approx(95.05)
        assert stats.p99 == pytest.approx(99.01)

    def test_dataclass_is_frozen(self) -> None:
        stats = compute_latency_stats([1.0, 2.0])
        with pytest.raises(AttributeError):
            stats.mean = 0.0  # type: ignore[misc]


class TestPercentileBoundary:
    def test_two_values_median(self) -> None:
        stats = compute_latency_stats([10.0, 20.0])
        assert stats.median == 15.0

    def test_identical_values(self) -> None:
        stats = compute_latency_stats([5.0, 5.0, 5.0, 5.0])
        assert stats.mean == 5.0
        assert stats.p95 == 5.0
        assert stats.p99 == 5.0
