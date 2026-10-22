"""Tests for eval.reporting.report_builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.reporting.report_builder import (
    build_report,
    build_report_context,
    load_results_jsonl,
    load_run_artifacts,
    load_summary,
    render_report_template,
)
from eval.reporting.tier_gate_validator import (
    load_thresholds_from_protocol,
    validate_tier_gates,
)
from tools.project_paths import resolve_project_root

PROJECT_ROOT = resolve_project_root()


@pytest.fixture
def sample_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_sample"
    run_dir.mkdir()

    summary = {
        "run_id": "run_sample",
        "date_utc": "2026-05-30T12:00:00+00:00",
        "protocol_id": "PAD-ISO-LAB-v1",
        "protocol_version": "1.0.0",
        "collection_id": "test_collection",
        "manifest_version": "1.0.0",
        "model_id": "mock_always_attack",
        "evaluation_policy": "fail_closed_v1",
        "assurance_tier": "standard",
        "n_attack": 1,
        "n_bona_fide": 1,
        "apcer_overall": 0.0,
        "bpcer_overall": 1.0,
        "apcer_overall_pct": 0.0,
        "bpcer_overall_pct": 100.0,
        "apcer_by_pai": {"PA-01": 0.0},
        "confusion": {"tp": 1, "tn": 0, "fp": 1, "fn": 0},
        "latency_ms": {
            "mean": 10.0,
            "median": 10.0,
            "min": 8.0,
            "max": 12.0,
            "p95": 11.5,
            "p99": 11.9,
            "count": 2,
        },
        "tier_gate": {
            "apcer": "pass",
            "bpcer": "fail",
            "latency": "pass",
            "overall": "fail",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = [
        {
            "sample_id": "ATK_001",
            "pai_id": "PA-01",
            "ground_truth": "attack",
            "subject_id": "subj_001",
            "predicted_outcome": "ATTACK",
            "normalized_outcome": "ATTACK",
            "reason_code": "passive_attack",
            "model_id": "mock_always_attack",
            "model_version": "1.0.0",
            "latency_ms": 10.0,
            "frames_analyzed": 4,
            "entrypoint": "PadSession.evaluate_final",
        },
        {
            "sample_id": "BF_001",
            "pai_id": "BF",
            "ground_truth": "bona_fide",
            "subject_id": "subj_002",
            "predicted_outcome": "ATTACK",
            "normalized_outcome": "ATTACK",
            "reason_code": "passive_attack",
            "model_id": "mock_always_attack",
            "model_version": "1.0.0",
            "latency_ms": 12.0,
            "frames_analyzed": 4,
            "entrypoint": "PadSession.evaluate_final",
        },
    ]
    jsonl = "\n".join(json.dumps(row) for row in rows) + "\n"
    (run_dir / "results.jsonl").write_text(jsonl, encoding="utf-8")

    snapshot = {
        "run_id": "run_sample",
        "protocol_id": "PAD-ISO-LAB-v1",
        "pad_settings": {
            "mode": "passive_only",
            "passive_pass_threshold": 0.62,
            "passive_reject_threshold": 0.38,
            "borderline_low": 0.45,
            "borderline_high": 0.58,
            "min_frames_passive": 2,
        },
        "environment": {
            "python_version": "3.11.0",
            "opencv_version": "4.10.0",
            "git_sha": "abc123",
        },
    }
    (run_dir / "settings_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    return run_dir


def test_load_summary_and_jsonl(sample_run_dir: Path) -> None:
    summary = load_summary(sample_run_dir / "summary.json")
    assert summary.run_id == "run_sample"
    rows = load_results_jsonl(sample_run_dir / "results.jsonl")
    assert len(rows) == 2


def test_render_report_template() -> None:
    template = "Run {{ run_id }} APCER {{ apcer_pooled_pct }}"
    rendered = render_report_template(template, {"run_id": "x", "apcer_pooled_pct": "0.00%"})
    assert rendered == "Run x APCER 0.00%"


def test_build_report_context(sample_run_dir: Path) -> None:
    artifacts = load_run_artifacts(sample_run_dir)
    protocol_path = PROJECT_ROOT / "eval" / "protocols" / "pad_iso_lab_v1.yaml"
    thresholds = load_thresholds_from_protocol(protocol_path, "standard")
    validation = validate_tier_gates(
        artifacts.summary, thresholds, artifacts.presentations
    )
    context = build_report_context(
        artifacts,
        tier_validation=validation,
        thresholds=thresholds,
    )
    assert context["run_id"] == "run_sample"
    assert context["git_sha"] == "abc123"
    assert "Review PAD" in context["recommendation_1"] or "No action" in context["recommendation_1"]


def test_build_report_writes_report_md(sample_run_dir: Path, tmp_path: Path) -> None:
    baselines = tmp_path / "baselines"
    baselines.mkdir()
    baseline_summary = {
        "run_id": "baseline_run",
        "protocol_id": "PAD-ISO-LAB-v1",
        "collection_id": "test",
        "model_id": "mock_always_attack",
        "evaluation_policy": "fail_closed_v1",
        "assurance_tier": "standard",
        "n_attack": 1,
        "n_bona_fide": 1,
        "apcer_overall": 0.0,
        "bpcer_overall": 0.5,
        "apcer_by_pai": {},
        "confusion": {},
        "tier_gate": {"apcer": "pass", "bpcer": "pass", "overall": "pass"},
    }
    (baselines / "mock_always_attack_standard.json").write_text(
        json.dumps(baseline_summary), encoding="utf-8"
    )

    result = build_report(
        sample_run_dir,
        baselines_dir=baselines,
    )
    assert result.report_path.is_file()
    text = result.report_path.read_text(encoding="utf-8")
    assert "run_sample" in text
    assert "Baseline comparison" in text
    assert result.has_regression is True
    assert result.overall_pass is False


def test_build_report_missing_summary_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="summary.json"):
        load_run_artifacts(tmp_path / "empty")
