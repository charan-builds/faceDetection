"""
Build PAD evaluation REPORT.md from E3 run artifacts.

Reads summary.json, results.jsonl, and settings_snapshot.json only.
Does not execute PAD or load evaluation datasets.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eval.reporting.baseline_compare import BaselineComparison

from eval.metrics.apcer_bpcer import to_percentage
from eval.reporting.tier_gate_validator import (
    TierGateThresholds,
    TierGateValidation,
    format_gate_targets,
    load_thresholds_from_protocol,
    protocol_path_for_id,
    validate_tier_gates,
)
from tools.project_paths import resolve_project_root

_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

PAI_LABELS: dict[str, str] = {
    "PA-01": "Print",
    "PA-02": "Static screen",
    "PA-03": "Video replay",
    "PA-04": "Virtual camera",
    "PA-05": "Other",
}


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Parsed E3 summary.json."""

    run_id: str
    date_utc: str
    protocol_id: str
    protocol_version: str
    collection_id: str
    manifest_version: str
    model_id: str
    evaluation_policy: str
    assurance_tier: str
    n_attack: int
    n_bona_fide: int
    apcer_overall: float | None
    bpcer_overall: float | None
    apcer_overall_pct: float | None
    bpcer_overall_pct: float | None
    apcer_by_pai: dict[str, float]
    confusion: dict[str, int]
    tier_gate: dict[str, str] | None
    latency_ms: dict[str, Any] | None
    raw: dict[str, Any]

    @property
    def latency_p95_ms(self) -> float | None:
        if not self.latency_ms:
            return None
        value = self.latency_ms.get("p95")
        return float(value) if value is not None else None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSummary:
        return cls(
            run_id=str(data["run_id"]),
            date_utc=str(data.get("date_utc", "")),
            protocol_id=str(data["protocol_id"]),
            protocol_version=str(data.get("protocol_version", "")),
            collection_id=str(data["collection_id"]),
            manifest_version=str(data.get("manifest_version", "")),
            model_id=str(data["model_id"]),
            evaluation_policy=str(data.get("evaluation_policy", "fail_closed_v1")),
            assurance_tier=str(data.get("assurance_tier", "standard")),
            n_attack=int(data.get("n_attack", 0)),
            n_bona_fide=int(data.get("n_bona_fide", 0)),
            apcer_overall=_optional_float(data.get("apcer_overall")),
            bpcer_overall=_optional_float(data.get("bpcer_overall")),
            apcer_overall_pct=_optional_float(data.get("apcer_overall_pct")),
            bpcer_overall_pct=_optional_float(data.get("bpcer_overall_pct")),
            apcer_by_pai={
                str(k): float(v) for k, v in (data.get("apcer_by_pai") or {}).items()
            },
            confusion={str(k): int(v) for k, v in (data.get("confusion") or {}).items()},
            tier_gate=dict(data["tier_gate"]) if data.get("tier_gate") else None,
            latency_ms=dict(data["latency_ms"]) if data.get("latency_ms") else None,
            raw=data,
        )


@dataclass(frozen=True, slots=True)
class PresentationRow:
    """One line from E3 results.jsonl."""

    sample_id: str
    pai_id: str
    ground_truth: str
    subject_id: str
    predicted_outcome: str
    normalized_outcome: str
    reason_code: str
    model_id: str
    model_version: str
    latency_ms: float
    frames_analyzed: int
    entrypoint: str
    source_paths: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PresentationRow:
        paths = data.get("source_paths") or []
        return cls(
            sample_id=str(data["sample_id"]),
            pai_id=str(data["pai_id"]),
            ground_truth=str(data["ground_truth"]),
            subject_id=str(data["subject_id"]),
            predicted_outcome=str(data["predicted_outcome"]),
            normalized_outcome=str(data.get("normalized_outcome", data["predicted_outcome"])),
            reason_code=str(data.get("reason_code", "")),
            model_id=str(data.get("model_id", "")),
            model_version=str(data.get("model_version", "")),
            latency_ms=float(data.get("latency_ms", 0.0)),
            frames_analyzed=int(data.get("frames_analyzed", 0)),
            entrypoint=str(data.get("entrypoint", "")),
            source_paths=tuple(str(path) for path in paths),
        )


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """All inputs required to build a report for one run directory."""

    run_dir: Path
    summary: RunSummary
    presentations: tuple[PresentationRow, ...]
    settings_snapshot: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class ReportBuildResult:
    """Output of a report build operation."""

    run_dir: Path
    report_path: Path
    tier_validation: TierGateValidation
    baseline_comparison: BaselineComparison | None  # noqa: F821
    overall_pass: bool
    has_regression: bool
    context: dict[str, str]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_summary(path: Path) -> RunSummary:
    """Load summary.json produced by the E3 harness."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid summary JSON: {path}")
    return RunSummary.from_dict(data)


def load_results_jsonl(path: Path) -> tuple[PresentationRow, ...]:
    """Load per-presentation results from results.jsonl."""
    if not path.is_file():
        raise FileNotFoundError(f"results.jsonl not found: {path}")

    rows: list[PresentationRow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        data = json.loads(stripped)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid JSONL record at {path}:{line_no}")
        rows.append(PresentationRow.from_dict(data))
    return tuple(rows)


def load_settings_snapshot(path: Path) -> dict[str, Any]:
    """Load settings_snapshot.json from an E3 run."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid settings snapshot: {path}")
    return data


def load_run_artifacts(run_dir: Path) -> RunArtifacts:
    """Load standard E3 artifacts from a run directory."""
    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.jsonl"
    snapshot_path = run_dir / "settings_snapshot.json"

    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")

    summary = load_summary(summary_path)
    presentations = load_results_jsonl(results_path) if results_path.is_file() else ()
    snapshot = (
        load_settings_snapshot(snapshot_path) if snapshot_path.is_file() else None
    )
    return RunArtifacts(
        run_dir=run_dir,
        summary=summary,
        presentations=presentations,
        settings_snapshot=snapshot,
    )


def _format_pct(rate: float | None) -> str:
    if rate is None:
        return "n/a"
    pct = to_percentage(rate)
    return f"{pct:.2f}%"


def _format_rate(rate: float | None) -> str:
    if rate is None:
        return "n/a"
    return f"{rate:.4f}"


def _count_bf_outcomes(presentations: tuple[PresentationRow, ...]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in presentations:
        if row.ground_truth == "bona_fide":
            counter[row.predicted_outcome] += 1
    return counter


def _count_pai_presentations(
    presentations: tuple[PresentationRow, ...],
    pai_id: str,
) -> int:
    return sum(
        1
        for row in presentations
        if row.ground_truth == "attack" and row.pai_id == pai_id
    )


def _recommendations(
    tier_validation: TierGateValidation,
    baseline_comparison: BaselineComparison | None  # noqa: F821,
) -> tuple[str, str]:
    rec1 = "No action required; metrics meet tier gates."
    rec2 = "Continue monitoring with fixture regression in CI."

    if not tier_validation.passed:
        rec1 = "Review PAD thresholds and passive model before promotion."
        rec2 = "Re-run evaluation after configuration or model changes."

    if baseline_comparison and baseline_comparison.has_regression:
        rec1 = "Investigate metric regression against approved baseline."
        rec2 = baseline_comparison.regression_summary or rec2

    return rec1, rec2


def build_report_context(
    artifacts: RunArtifacts,
    *,
    tier_validation: TierGateValidation,
    thresholds: TierGateThresholds,
    baseline_comparison: BaselineComparison | None = None,  # noqa: F821
) -> dict[str, str]:
    """Assemble template placeholder values for REPORT.md."""
    summary = artifacts.summary
    presentations = artifacts.presentations
    snapshot = artifacts.settings_snapshot or {}

    environment = snapshot.get("environment") or {}
    pad_settings = snapshot.get("pad_settings") or {}

    bf_counts = _count_bf_outcomes(presentations)
    bf_total = summary.n_bona_fide or sum(bf_counts.values()) or 1

    def bf_pct(outcome: str) -> str:
        count = bf_counts.get(outcome, 0)
        return f"{(count / bf_total) * 100:.2f}%"

    confusion = summary.confusion
    pooled_apcer = tier_validation.pooled_apcer

    apcer_by_pai = summary.apcer_by_pai
    targets = format_gate_targets(thresholds)

    context: dict[str, str] = {
        "run_id": summary.run_id,
        "date_utc": summary.date_utc,
        "protocol_id": summary.protocol_id,
        "git_sha": str(environment.get("git_sha") or "n/a"),
        "python_version": str(environment.get("python_version") or "n/a"),
        "opencv_version": str(environment.get("opencv_version") or "n/a"),
        "model_id": summary.model_id,
        "model_version": str(snapshot.get("model_version") or presentations[0].model_version if presentations else "n/a"),
        "pad_mode": str(pad_settings.get("mode", "n/a")),
        "assurance_tier": summary.assurance_tier,
        "evaluation_policy": summary.evaluation_policy,
        "collection_id": summary.collection_id,
        "manifest_version": summary.manifest_version,
        "apcer_pooled_pct": _format_pct(pooled_apcer),
        "bpcer_pct": _format_pct(summary.bpcer_overall),
        "apcer_target_pct": targets["apcer_target_pct"],
        "bpcer_target_pct": targets["bpcer_target_pct"],
        "latency_target_ms": targets["latency_target_ms"],
        "apcer_gate": tier_validation.apcer_status.upper(),
        "bpcer_gate": tier_validation.bpcer_status.upper(),
        "latency_gate": tier_validation.latency_status.upper(),
        "overall_gate": tier_validation.overall_status.upper(),
        "latency_p95_ms": (
            f"{tier_validation.latency_p95_ms:.1f}"
            if tier_validation.latency_p95_ms is not None
            else "n/a"
        ),
        "n_pa01": str(_count_pai_presentations(presentations, "PA-01") or "0"),
        "n_pa02": str(_count_pai_presentations(presentations, "PA-02") or "0"),
        "n_pa03": str(_count_pai_presentations(presentations, "PA-03") or "0"),
        "n_pa04": str(_count_pai_presentations(presentations, "PA-04") or "0"),
        "apcer_pa01_pct": _format_pct(apcer_by_pai.get("PA-01")),
        "apcer_pa02_pct": _format_pct(apcer_by_pai.get("PA-02")),
        "apcer_pa03_pct": _format_pct(apcer_by_pai.get("PA-03")),
        "apcer_pa04_pct": _format_pct(apcer_by_pai.get("PA-04")),
        "n_attack_pooled": str(
            sum(
                _count_pai_presentations(presentations, pai)
                for pai in thresholds.apcer_pooled_pai_ids
            )
        ),
        "bf_live_count": str(bf_counts.get("LIVE", 0)),
        "bf_inconclusive_count": str(bf_counts.get("INCONCLUSIVE", 0)),
        "bf_attack_count": str(bf_counts.get("ATTACK", 0)),
        "bf_error_count": str(bf_counts.get("ERROR", 0)),
        "bf_live_pct": bf_pct("LIVE"),
        "bf_inconclusive_pct": bf_pct("INCONCLUSIVE"),
        "bf_attack_pct": bf_pct("ATTACK"),
        "bf_error_pct": bf_pct("ERROR"),
        "cm_attack_as_live": str(confusion.get("fn", 0)),
        "cm_attack_correct": str(confusion.get("tp", 0)),
        "cm_bf_live": str(confusion.get("tn", 0)),
        "cm_bf_rejected": str(confusion.get("fp", 0)),
        "passive_pass_threshold": str(pad_settings.get("passive_pass_threshold", "n/a")),
        "passive_reject_threshold": str(pad_settings.get("passive_reject_threshold", "n/a")),
        "borderline_low": str(pad_settings.get("borderline_low", "n/a")),
        "borderline_high": str(pad_settings.get("borderline_high", "n/a")),
        "min_frames_passive": str(pad_settings.get("min_frames_passive", "n/a")),
    }

    rec1, rec2 = _recommendations(tier_validation, baseline_comparison)
    context["recommendation_1"] = rec1
    context["recommendation_2"] = rec2

    if baseline_comparison:
        context["recommendation_2"] = (
            f"{rec2} | Baseline: {baseline_comparison.baseline_path.name}"
        )

    return context


def render_report_template(template_text: str, context: dict[str, str]) -> str:
    """Replace {{ placeholder }} tokens in the report template."""

    def replace_match(match: re.Match[str]) -> str:
        key = match.group(1)
        return context.get(key, "n/a")

    rendered = _PLACEHOLDER_PATTERN.sub(replace_match, template_text)
    return rendered


def append_baseline_section(
    report_text: str,
    baseline_comparison: BaselineComparison | None  # noqa: F821,
) -> str:
    """Append baseline comparison section to the rendered report."""
    if baseline_comparison is None:
        return report_text

    lines = [
        "",
        "---",
        "",
        "## Baseline comparison",
        "",
        f"**Baseline file:** `{baseline_comparison.baseline_path.name}`",
        f"**Baseline run ID:** `{baseline_comparison.baseline_run_id}`",
        f"**Regression detected:** `{'YES' if baseline_comparison.has_regression else 'NO'}`",
        "",
        "| Metric | Baseline | Current | Delta | Status |",
        "|--------|----------|---------|-------|--------|",
    ]
    for delta in baseline_comparison.deltas:
        status = "REGRESSION" if delta.is_regression else "ok"
        lines.append(
            f"| {delta.metric} | {_format_rate(delta.baseline)} | "
            f"{_format_rate(delta.current)} | {delta.delta:+.4f} | {status} |"
        )
    if baseline_comparison.regression_summary:
        lines.extend(["", baseline_comparison.regression_summary])

    return report_text + "\n".join(lines) + "\n"


def write_report(path: Path, content: str) -> Path:
    """Write REPORT.md to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def build_arg_parser():
    """CLI argument parser for report generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build PAD evaluation REPORT.md from E3 run artifacts.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="E3 run directory containing summary.json and results.jsonl.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Report template path (default: eval/reports/template/report_template.md).",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="Protocol YAML for tier gate thresholds.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Explicit baseline summary.json path.",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=None,
        help="Directory of approved baseline summaries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Phase E4 report generation."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = build_report(
        args.run_dir,
        template_path=args.template,
        protocol_path=args.protocol,
        baselines_dir=args.baselines_dir,
        baseline_path=args.baseline,
    )
    status = "PASS" if result.overall_pass else "FAIL"
    print(
        f"Report written: {result.report_path} | tier_gate={result.tier_validation.overall_status} "
        f"| regression={result.has_regression} | overall={status}"
    )
    return 0 if result.overall_pass else 1


def build_report(
    run_dir: Path,
    *,
    template_path: Path | None = None,
    protocol_path: Path | None = None,
    protocols_dir: Path | None = None,
    baselines_dir: Path | None = None,
    baseline_path: Path | None = None,
    baseline_config: BaselineCompareConfig | None = None,
) -> ReportBuildResult:
    """
    Build REPORT.md for an E3 run directory.

    Validates tier gates, optionally compares to baseline, and writes REPORT.md.
    """
    project_root = resolve_project_root()
    template_path = template_path or (
        project_root / "eval" / "reports" / "template" / "report_template.md"
    )
    protocols_dir = protocols_dir or (project_root / "eval" / "protocols")
    baselines_dir = baselines_dir or (project_root / "eval" / "baselines")

    artifacts = load_run_artifacts(run_dir)
    summary = artifacts.summary

    if protocol_path is None:
        protocol_path = protocol_path_for_id(summary.protocol_id, protocols_dir)

    thresholds = load_thresholds_from_protocol(protocol_path, summary.assurance_tier)
    tier_validation = validate_tier_gates(
        summary, thresholds, artifacts.presentations
    )

    from eval.reporting.baseline_compare import (
        compare_to_baseline,
        load_baseline_summary,
        resolve_baseline_path,
    )

    comparison = None
    resolved_baseline = baseline_path or resolve_baseline_path(
        baselines_dir, summary.model_id, summary.assurance_tier
    )
    if resolved_baseline is not None:
        baseline_summary = load_baseline_summary(resolved_baseline)
        comparison = compare_to_baseline(
            summary,
            baseline_summary,
            resolved_baseline,
            config=baseline_config,
        )

    context = build_report_context(
        artifacts,
        tier_validation=tier_validation,
        thresholds=thresholds,
        baseline_comparison=comparison,
    )
    template_text = template_path.read_text(encoding="utf-8")
    report_body = render_report_template(template_text, context)
    report_body = append_baseline_section(report_body, comparison)

    report_path = write_report(run_dir / "REPORT.md", report_body)

    has_regression = bool(comparison and comparison.has_regression)
    overall_pass = tier_validation.passed and not has_regression

    return ReportBuildResult(
        run_dir=run_dir,
        report_path=report_path,
        tier_validation=tier_validation,
        baseline_comparison=comparison,
        overall_pass=overall_pass,
        has_regression=has_regression,
        context=context,
    )


if __name__ == "__main__":
    raise SystemExit(main())
