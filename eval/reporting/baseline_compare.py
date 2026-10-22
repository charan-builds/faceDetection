"""
Compare current PAD evaluation runs against approved baseline summaries.

Uses E3 summary.json files only — no PAD execution or dataset loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eval.reporting.report_builder import RunSummary, load_summary


@dataclass(frozen=True, slots=True)
class BaselineCompareConfig:
    """Maximum allowed absolute increase vs baseline before flagging regression."""

    apcer_max_delta: float = 0.01
    bpcer_max_delta: float = 0.01
    require_matching_model: bool = True
    require_matching_tier: bool = True


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """Comparison of one metric between baseline and current run."""

    metric: str
    baseline: float | None
    current: float | None
    delta: float
    is_regression: bool


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """Outcome of baseline comparison."""

    baseline_path: Path
    baseline_run_id: str
    current_run_id: str
    deltas: tuple[MetricDelta, ...]
    has_regression: bool
    regression_summary: str


def resolve_baseline_path(
    baselines_dir: Path,
    model_id: str,
    assurance_tier: str,
) -> Path | None:
    """
    Resolve baseline file using naming convention: {model_id}_{tier}.json

    Returns None if no matching baseline exists.
    """
    if not baselines_dir.is_dir():
        return None

    candidates = [
        baselines_dir / f"{model_id}_{assurance_tier}.json",
        baselines_dir / f"{model_id}_{assurance_tier}_iso_lab.json",
    ]
    for path in candidates:
        if path.is_file():
            return path

    # Fallback: any file starting with model_id
    for path in sorted(baselines_dir.glob(f"{model_id}_*.json")):
        if path.is_file():
            return path
    return None


def load_baseline_summary(path: Path) -> RunSummary:
    """Load an approved baseline summary JSON file."""
    return load_summary(path)


def _metric_delta(
    metric: str,
    baseline: float | None,
    current: float | None,
    max_allowed_increase: float,
) -> MetricDelta:
    if baseline is None or current is None:
        return MetricDelta(
            metric=metric,
            baseline=baseline,
            current=current,
            delta=0.0,
            is_regression=False,
        )

    delta = current - baseline
    # Higher APCER/BPCER is worse → regression when delta exceeds tolerance
    is_regression = delta > max_allowed_increase
    return MetricDelta(
        metric=metric,
        baseline=baseline,
        current=current,
        delta=delta,
        is_regression=is_regression,
    )


def compare_to_baseline(
    current: RunSummary,
    baseline: RunSummary,
    baseline_path: Path,
    *,
    config: BaselineCompareConfig | None = None,
) -> BaselineComparison:
    """
    Compare current run metrics to an approved baseline summary.

    Raises ValueError when model/tier mismatch and strict matching is enabled.
    """
    cfg = config or BaselineCompareConfig()

    if cfg.require_matching_model and current.model_id != baseline.model_id:
        raise ValueError(
            f"Model mismatch: current={current.model_id!r} "
            f"baseline={baseline.model_id!r}"
        )
    if cfg.require_matching_tier and current.assurance_tier != baseline.assurance_tier:
        raise ValueError(
            f"Tier mismatch: current={current.assurance_tier!r} "
            f"baseline={baseline.assurance_tier!r}"
        )

    deltas = (
        _metric_delta(
            "apcer_overall",
            baseline.apcer_overall,
            current.apcer_overall,
            cfg.apcer_max_delta,
        ),
        _metric_delta(
            "bpcer_overall",
            baseline.bpcer_overall,
            current.bpcer_overall,
            cfg.bpcer_max_delta,
        ),
    )

    regressions = [delta for delta in deltas if delta.is_regression]
    has_regression = bool(regressions)

    if has_regression:
        parts = [
            f"{delta.metric} worsened by {delta.delta:+.4f} "
            f"(baseline={delta.baseline:.4f}, current={delta.current:.4f})"
            for delta in regressions
        ]
        summary = "Regressions detected: " + "; ".join(parts)
    else:
        summary = "No metric regression beyond configured tolerance."

    return BaselineComparison(
        baseline_path=baseline_path,
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        deltas=deltas,
        has_regression=has_regression,
        regression_summary=summary,
    )
