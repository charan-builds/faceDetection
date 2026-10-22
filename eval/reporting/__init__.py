"""PAD evaluation reporting and baseline analysis (Phase E4)."""

from eval.reporting.baseline_compare import (
    BaselineCompareConfig,
    BaselineComparison,
    MetricDelta,
    compare_to_baseline,
    load_baseline_summary,
    resolve_baseline_path,
)
from eval.reporting.report_builder import (
    PresentationRow,
    ReportBuildResult,
    RunArtifacts,
    RunSummary,
    build_report,
    load_results_jsonl,
    load_run_artifacts,
    load_settings_snapshot,
    load_summary,
    write_report,
)
from eval.reporting.tier_gate_validator import (
    GateStatus,
    TierGateThresholds,
    TierGateValidation,
    load_thresholds_from_protocol,
    validate_tier_gates,
)

__all__ = [
    "BaselineCompareConfig",
    "BaselineComparison",
    "GateStatus",
    "MetricDelta",
    "PresentationRow",
    "ReportBuildResult",
    "RunArtifacts",
    "RunSummary",
    "TierGateThresholds",
    "TierGateValidation",
    "build_report",
    "compare_to_baseline",
    "load_baseline_summary",
    "load_results_jsonl",
    "load_run_artifacts",
    "load_settings_snapshot",
    "load_summary",
    "load_thresholds_from_protocol",
    "resolve_baseline_path",
    "validate_tier_gates",
    "write_report",
]
