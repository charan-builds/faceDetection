"""
Validate PAD evaluation runs against protocol tier gates.

Uses E3 summary.json metrics and optional protocol YAML thresholds.
Does not execute PAD or load datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

from eval.metrics.apcer_bpcer import compute_apcer, to_percentage

GateStatus = Literal["pass", "fail", "skipped"]

DEFAULT_POOLED_PAI_IDS: tuple[str, ...] = ("PA-01", "PA-02", "PA-03")

PROTOCOL_FILE_BY_ID: dict[str, str] = {
    "PAD-ISO-LAB-v1": "pad_iso_lab_v1.yaml",
    "PAD-LIVE-SIM-v1": "pad_live_sim_v1.yaml",
    "PAD-REG-v1": "pad_registration_v1.yaml",
}


class SummaryLike(Protocol):
    """Minimal summary fields required for tier gate validation."""

    protocol_id: str
    assurance_tier: str
    apcer_overall: float | None
    bpcer_overall: float | None

    @property
    def latency_p95_ms(self) -> float | None: ...


class PresentationLike(Protocol):
    """Minimal presentation row for pooled APCER recomputation."""

    ground_truth: str
    pai_id: str
    predicted_outcome: str


@dataclass(frozen=True, slots=True)
class TierGateThresholds:
    """Maximum allowed metric values for an assurance tier."""

    assurance_tier: str
    apcer_max_pooled: float
    bpcer_max: float
    latency_ms_p95_max: float | None
    apcer_pooled_pai_ids: tuple[str, ...] = DEFAULT_POOLED_PAI_IDS


@dataclass(frozen=True, slots=True)
class MetricGateCheck:
    """Single metric compared against a tier threshold."""

    metric_name: str
    value: float | None
    threshold: float | None
    status: GateStatus
    detail: str


@dataclass(frozen=True, slots=True)
class TierGateValidation:
    """Full tier-gate validation outcome for a run."""

    assurance_tier: str
    pooled_apcer: float | None
    bpcer: float | None
    latency_p95_ms: float | None
    checks: tuple[MetricGateCheck, ...]
    apcer_status: GateStatus
    bpcer_status: GateStatus
    latency_status: GateStatus
    overall_status: GateStatus

    @property
    def passed(self) -> bool:
        return self.overall_status == "pass"


def protocol_path_for_id(protocol_id: str, protocols_dir: Path) -> Path:
    """Resolve a protocol YAML path from a protocol_id."""
    filename = PROTOCOL_FILE_BY_ID.get(protocol_id)
    if not filename:
        raise ValueError(f"Unknown protocol_id for tier gates: {protocol_id!r}")
    path = protocols_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {path}")
    return path


def load_thresholds_from_protocol(
    protocol_path: Path,
    assurance_tier: str,
) -> TierGateThresholds:
    """Load tier gate thresholds from a protocol YAML file."""
    import yaml

    raw = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid protocol YAML: {protocol_path}")

    tier_gates = raw.get("tier_gates")
    if not isinstance(tier_gates, dict):
        raise ValueError(f"tier_gates missing in {protocol_path}")

    gates = tier_gates.get(assurance_tier)
    if not isinstance(gates, dict):
        raise ValueError(
            f"Assurance tier {assurance_tier!r} not defined in {protocol_path}"
        )

    pooled_ids = gates.get("apcer_pooled_pai_ids", list(DEFAULT_POOLED_PAI_IDS))
    if not isinstance(pooled_ids, list):
        pooled_ids = list(DEFAULT_POOLED_PAI_IDS)

    latency_limit = gates.get("latency_ms_p95_max")
    return TierGateThresholds(
        assurance_tier=assurance_tier,
        apcer_max_pooled=float(gates.get("apcer_max_pooled", 1.0)),
        bpcer_max=float(gates.get("bpcer_max", 1.0)),
        latency_ms_p95_max=float(latency_limit) if latency_limit is not None else None,
        apcer_pooled_pai_ids=tuple(str(pai) for pai in pooled_ids),
    )


def compute_pooled_apcer(
    presentations: Sequence[PresentationLike],
    pai_ids: tuple[str, ...],
) -> float | None:
    """Compute APCER over attack presentations in the given PAI set."""
    outcomes = [
        row.predicted_outcome
        for row in presentations
        if row.ground_truth == "attack" and row.pai_id in pai_ids
    ]
    if not outcomes:
        return None
    return compute_apcer(outcomes)


def _check_max_metric(
    *,
    metric_name: str,
    value: float | None,
    threshold: float | None,
) -> MetricGateCheck:
    if value is None or threshold is None:
        return MetricGateCheck(
            metric_name=metric_name,
            value=value,
            threshold=threshold,
            status="skipped",
            detail=f"{metric_name}: insufficient data",
        )
    status: GateStatus = "pass" if value <= threshold else "fail"
    return MetricGateCheck(
        metric_name=metric_name,
        value=value,
        threshold=threshold,
        status=status,
        detail=(
            f"{metric_name}={value:.4f} (threshold ≤ {threshold:.4f}) → {status}"
        ),
    )


def _derive_overall_status(
    apcer_status: GateStatus,
    bpcer_status: GateStatus,
    latency_status: GateStatus,
) -> GateStatus:
    if apcer_status == "fail" or bpcer_status == "fail" or latency_status == "fail":
        return "fail"
    if apcer_status == "pass" and bpcer_status == "pass":
        return "pass"
    return "fail"


def validate_tier_gates(
    summary: SummaryLike,
    thresholds: TierGateThresholds,
    presentations: Sequence[PresentationLike] = (),
) -> TierGateValidation:
    """
    Validate a run summary against tier gate thresholds.

    Pooled APCER is recomputed from presentations when provided; otherwise
    falls back to summary.apcer_overall.
    """
    pooled_apcer = compute_pooled_apcer(presentations, thresholds.apcer_pooled_pai_ids)
    if pooled_apcer is None:
        pooled_apcer = summary.apcer_overall

    bpcer = summary.bpcer_overall
    latency_p95 = summary.latency_p95_ms

    apcer_check = _check_max_metric(
        metric_name="apcer_pooled",
        value=pooled_apcer,
        threshold=thresholds.apcer_max_pooled,
    )
    bpcer_check = _check_max_metric(
        metric_name="bpcer",
        value=bpcer,
        threshold=thresholds.bpcer_max,
    )
    latency_check = _check_max_metric(
        metric_name="latency_p95_ms",
        value=latency_p95,
        threshold=thresholds.latency_ms_p95_max,
    )

    overall = _derive_overall_status(
        apcer_check.status,
        bpcer_check.status,
        latency_check.status,
    )

    return TierGateValidation(
        assurance_tier=thresholds.assurance_tier,
        pooled_apcer=pooled_apcer,
        bpcer=bpcer,
        latency_p95_ms=latency_p95,
        checks=(apcer_check, bpcer_check, latency_check),
        apcer_status=apcer_check.status,
        bpcer_status=bpcer_check.status,
        latency_status=latency_check.status,
        overall_status=overall,
    )


def validate_from_run_summary(
    summary: SummaryLike,
    *,
    protocol_path: Path | None = None,
    protocols_dir: Path | None = None,
    presentations: Sequence[PresentationLike] = (),
) -> TierGateValidation:
    """Load thresholds from protocol (if path given) and validate."""
    if protocol_path is None:
        if protocols_dir is None:
            raise ValueError("protocol_path or protocols_dir is required")
        protocol_path = protocol_path_for_id(summary.protocol_id, protocols_dir)

    thresholds = load_thresholds_from_protocol(protocol_path, summary.assurance_tier)
    return validate_tier_gates(summary, thresholds, presentations)


def format_gate_targets(thresholds: TierGateThresholds) -> dict[str, str]:
    """Format threshold values for report placeholders (percentages where applicable)."""
    return {
        "apcer_target_pct": f"{to_percentage(thresholds.apcer_max_pooled):.2f}%",
        "bpcer_target_pct": f"{to_percentage(thresholds.bpcer_max):.2f}%",
        "latency_target_ms": (
            f"{thresholds.latency_ms_p95_max:.0f}"
            if thresholds.latency_ms_p95_max is not None
            else "n/a"
        ),
    }
