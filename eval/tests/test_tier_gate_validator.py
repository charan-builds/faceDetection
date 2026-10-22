"""Tests for eval.reporting.tier_gate_validator."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eval.reporting.report_builder import PresentationRow, RunSummary
from eval.reporting.tier_gate_validator import (
    load_thresholds_from_protocol,
    validate_tier_gates,
)
from tools.project_paths import resolve_project_root

PROJECT_ROOT = resolve_project_root()


def _summary(
    *,
    apcer: float = 0.02,
    bpcer: float = 0.03,
    p95: float = 500.0,
) -> RunSummary:
    return RunSummary.from_dict(
        {
            "run_id": "run_test",
            "date_utc": "2026-01-01T00:00:00+00:00",
            "protocol_id": "PAD-ISO-LAB-v1",
            "protocol_version": "1.0.0",
            "collection_id": "test",
            "manifest_version": "1.0.0",
            "model_id": "mock_always_live",
            "evaluation_policy": "fail_closed_v1",
            "assurance_tier": "standard",
            "n_attack": 2,
            "n_bona_fide": 2,
            "apcer_overall": apcer,
            "bpcer_overall": bpcer,
            "apcer_by_pai": {"PA-01": apcer},
            "confusion": {"tp": 1, "fn": 1, "tn": 2, "fp": 0},
            "latency_ms": {"p95": p95, "mean": 100.0},
        }
    )


def test_load_thresholds_from_protocol() -> None:
    path = PROJECT_ROOT / "eval" / "protocols" / "pad_iso_lab_v1.yaml"
    thresholds = load_thresholds_from_protocol(path, "standard")
    assert thresholds.apcer_max_pooled == 0.05
    assert thresholds.bpcer_max == 0.05
    assert thresholds.latency_ms_p95_max == 1200


def test_validate_tier_gates_pass() -> None:
    path = PROJECT_ROOT / "eval" / "protocols" / "pad_iso_lab_v1.yaml"
    thresholds = load_thresholds_from_protocol(path, "standard")
    validation = validate_tier_gates(_summary(), thresholds)
    assert validation.passed
    assert validation.apcer_status == "pass"
    assert validation.bpcer_status == "pass"
    assert validation.latency_status == "pass"


def test_validate_tier_gates_fail_apcer() -> None:
    path = PROJECT_ROOT / "eval" / "protocols" / "pad_iso_lab_v1.yaml"
    thresholds = load_thresholds_from_protocol(path, "standard")
    validation = validate_tier_gates(_summary(apcer=0.10, bpcer=0.01), thresholds)
    assert not validation.passed
    assert validation.apcer_status == "fail"


def test_validate_pooled_apcer_from_presentations() -> None:
    path = PROJECT_ROOT / "eval" / "protocols" / "pad_iso_lab_v1.yaml"
    thresholds = load_thresholds_from_protocol(path, "standard")
    presentations = (
        PresentationRow.from_dict(
            {
                "sample_id": "A1",
                "pai_id": "PA-01",
                "ground_truth": "attack",
                "subject_id": "subj_001",
                "predicted_outcome": "LIVE",
                "normalized_outcome": "LIVE",
                "reason_code": "x",
                "model_id": "m",
                "model_version": "1",
                "latency_ms": 1.0,
                "frames_analyzed": 1,
                "entrypoint": "e",
            }
        ),
        PresentationRow.from_dict(
            {
                "sample_id": "A2",
                "pai_id": "PA-01",
                "ground_truth": "attack",
                "subject_id": "subj_001",
                "predicted_outcome": "ATTACK",
                "normalized_outcome": "ATTACK",
                "reason_code": "x",
                "model_id": "m",
                "model_version": "1",
                "latency_ms": 1.0,
                "frames_analyzed": 1,
                "entrypoint": "e",
            }
        ),
    )
    validation = validate_tier_gates(
        _summary(apcer=0.0),
        thresholds,
        presentations,
    )
    assert validation.pooled_apcer == 0.5
    assert validation.apcer_status == "fail"


def test_missing_tier_raises(tmp_path: Path) -> None:
    proto = {"tier_gates": {"standard": {"apcer_max_pooled": 0.05}}}
    path = tmp_path / "p.yaml"
    path.write_text(yaml.dump(proto), encoding="utf-8")
    with pytest.raises(ValueError, match="not defined"):
        load_thresholds_from_protocol(path, "high")
