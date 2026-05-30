"""
Latency distribution statistics for PAD evaluation runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Summary statistics for presentation latency samples (milliseconds)."""

    mean: float
    median: float
    min: float
    max: float
    p95: float
    p99: float
    count: int


def _percentile_linear(sorted_values: list[float], percent: float) -> float:
    """
    Linear-interpolation percentile on a sorted list.

    ``percent`` is in [0, 100] (e.g. 95 for P95).
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (percent / 100.0) * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def compute_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    """
    Compute mean, median, min, max, P95, and P99 from latency samples.

    Raises ValueError if the input list is empty or contains non-finite values.
    """
    if not latencies_ms:
        raise ValueError("Cannot compute latency stats: empty input list.")

    for value in latencies_ms:
        if not math.isfinite(value):
            raise ValueError(
                f"Cannot compute latency stats: non-finite value {value!r}."
            )

    sorted_vals = sorted(latencies_ms)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    median = _percentile_linear(sorted_vals, 50.0)

    return LatencyStats(
        mean=mean,
        median=median,
        min=sorted_vals[0],
        max=sorted_vals[-1],
        p95=_percentile_linear(sorted_vals, 95.0),
        p99=_percentile_linear(sorted_vals, 99.0),
        count=n,
    )
