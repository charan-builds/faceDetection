"""Tests for eval.reporting.baseline_compare."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.reporting.baseline_compare import (
    BaselineCompareConfig,
    compare_to_baseline,
    resolve_baseline_path,
)
from eval.reporting.report_builder import RunSummary


def _summary(
    run_id: str,
    *,
    apcer: float,
    bpcer: float,
    model_id: str = "heuristic_rgb_v1",
) -> RunSummary:
    return RunSummary.from_dict(
        {
            "run_id": run_id,
            "date_utc": "2026-01-01T00:00:00+00:00",
            "protocol_id": "PAD-ISO-LAB-v1",
            "collection_id": "test",
            "model_id": model_id,
            "evaluation_policy": "fail_closed_v1",
            "assurance_tier": "standard",
            "n_attack": 10,
            "n_bona_fide": 10,
            "apcer_overall": apcer,
            "bpcer_overall": bpcer,
            "apcer_by_pai": {},
            "confusion": {},
        }
    )


def test_compare_no_regression(tmp_path: Path) -> None:
    baseline_path = tmp_path / "heuristic_rgb_v1_standard.json"
    baseline = _summary("baseline_run", apcer=0.02, bpcer=0.03)
    current = _summary("current_run", apcer=0.025, bpcer=0.035)

    baseline_path.write_text("{}", encoding="utf-8")  # not used for load in compare

    result = compare_to_baseline(current, baseline, baseline_path)
    assert not result.has_regression
    assert len(result.deltas) == 2


def test_compare_detects_regression(tmp_path: Path) -> None:
    baseline_path = tmp_path / "heuristic_rgb_v1_standard.json"
    baseline = _summary("baseline_run", apcer=0.02, bpcer=0.03)
    current = _summary("current_run", apcer=0.05, bpcer=0.03)

    result = compare_to_baseline(
        current,
        baseline,
        baseline_path,
        config=BaselineCompareConfig(apcer_max_delta=0.01),
    )
    assert result.has_regression
    assert "apcer_overall" in result.regression_summary


def test_compare_model_mismatch() -> None:
    baseline = _summary("b", apcer=0.01, bpcer=0.01, model_id="model_a")
    current = _summary("c", apcer=0.01, bpcer=0.01, model_id="model_b")
    with pytest.raises(ValueError, match="Model mismatch"):
        compare_to_baseline(current, baseline, Path("baseline.json"))


def test_resolve_baseline_path(tmp_path: Path) -> None:
    path = tmp_path / "mock_always_attack_standard.json"
    path.write_text("{}", encoding="utf-8")
    resolved = resolve_baseline_path(
        tmp_path, "mock_always_attack", "standard"
    )
    assert resolved == path


def test_resolve_baseline_missing(tmp_path: Path) -> None:
    assert resolve_baseline_path(tmp_path, "unknown_model", "standard") is None
