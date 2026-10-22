"""
Unit tests for tools.freeze_baseline.

No real evaluation harness or network access required.
All summary.json files are written inline to tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.freeze_baseline import (
    BaselineExistsError,
    FreezeResult,
    FreezeValidationError,
    baseline_filename,
    freeze_baseline,
    load_and_validate_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _passing_summary(
    run_id: str = "run_001",
    model_id: str = "heuristic_rgb_v1",
    assurance_tier: str = "standard",
    apcer: str = "pass",
    bpcer: str = "pass",
    latency: str = "pass",
    overall: str = "pass",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_id": model_id,
        "assurance_tier": assurance_tier,
        "protocol_id": "PAD-ISO-LAB-v1",
        "collection_id": "face_lock_v1",
        "evaluation_policy": "fail_closed_v1",
        "n_attack": 150,
        "n_bona_fide": 150,
        "apcer_overall": 0.03,
        "bpcer_overall": 0.02,
        "tier_gate": {
            "apcer": apcer,
            "bpcer": bpcer,
            "latency": latency,
            "overall": overall,
        },
    }


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests: load_and_validate_summary
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadAndValidateSummary:
    def test_passing_summary_loads_cleanly(self, tmp_path: Path) -> None:
        path = _write_summary(tmp_path, _passing_summary())
        data = load_and_validate_summary(path)
        assert data["run_id"] == "run_001"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FreezeValidationError, match="not found"):
            load_and_validate_summary(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "summary.json"
        bad.write_text("not json", encoding="utf-8")
        with pytest.raises(FreezeValidationError, match="not valid JSON"):
            load_and_validate_summary(bad)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        s = _passing_summary()
        del s["tier_gate"]
        path = _write_summary(tmp_path, s)
        with pytest.raises(FreezeValidationError, match="missing required fields"):
            load_and_validate_summary(path)

    def test_overall_fail_raises(self, tmp_path: Path) -> None:
        s = _passing_summary(overall="fail", apcer="fail")
        path = _write_summary(tmp_path, s)
        with pytest.raises(FreezeValidationError, match="APCER"):
            load_and_validate_summary(path)

    def test_apcer_gate_fail_raises(self, tmp_path: Path) -> None:
        s = _passing_summary(apcer="fail", overall="fail")
        path = _write_summary(tmp_path, s)
        with pytest.raises(FreezeValidationError, match="APCER"):
            load_and_validate_summary(path)

    def test_bpcer_gate_fail_raises(self, tmp_path: Path) -> None:
        s = _passing_summary(bpcer="fail", overall="fail")
        path = _write_summary(tmp_path, s)
        with pytest.raises(FreezeValidationError, match="BPCER"):
            load_and_validate_summary(path)

    def test_latency_skipped_is_accepted(self, tmp_path: Path) -> None:
        """Latency=skipped should not block a freeze."""
        s = _passing_summary(latency="skipped")
        path = _write_summary(tmp_path, s)
        data = load_and_validate_summary(path)
        assert data["tier_gate"]["latency"] == "skipped"

    def test_tier_gate_not_dict_raises(self, tmp_path: Path) -> None:
        s = _passing_summary()
        s["tier_gate"] = "pass"  # wrong type
        path = _write_summary(tmp_path, s)
        with pytest.raises(FreezeValidationError, match="object"):
            load_and_validate_summary(path)


# ---------------------------------------------------------------------------
# Tests: baseline_filename
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBaselineFilename:
    def test_standard_name(self) -> None:
        assert baseline_filename("heuristic_rgb_v1", "standard") == "heuristic_rgb_v1_standard.json"

    def test_spaces_replaced_with_underscores(self) -> None:
        name = baseline_filename("my model", "high tier")
        assert " " not in name
        assert "my_model_high_tier.json" == name

    def test_path_separators_replaced(self) -> None:
        name = baseline_filename("model/v1", "standard")
        assert "/" not in name


# ---------------------------------------------------------------------------
# Tests: freeze_baseline
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFreezeBaseline:
    def test_happy_path_copies_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        _write_summary(run_dir, _passing_summary())

        result = freeze_baseline(run_dir, baselines_dir=baselines_dir)

        assert result.baseline_path.is_file()
        assert result.dry_run is False
        assert result.model_id == "heuristic_rgb_v1"
        assert result.assurance_tier == "standard"
        assert result.backup_path is None  # no previous file

    def test_baseline_content_matches_summary(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        summary = _passing_summary(run_id="test_run_42")
        _write_summary(run_dir, summary)

        result = freeze_baseline(run_dir, baselines_dir=baselines_dir)

        frozen = json.loads(result.baseline_path.read_text())
        assert frozen["run_id"] == "test_run_42"

    def test_existing_baseline_raises_without_force(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        _write_summary(run_dir, _passing_summary())

        # Pre-create the baseline file
        existing = baselines_dir / "heuristic_rgb_v1_standard.json"
        existing.write_text("{}", encoding="utf-8")

        with pytest.raises(BaselineExistsError):
            freeze_baseline(run_dir, baselines_dir=baselines_dir, force=False)

    def test_force_overwrites_and_backs_up(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        _write_summary(run_dir, _passing_summary(run_id="new_run"))

        # Pre-create an existing baseline with different content
        existing = baselines_dir / "heuristic_rgb_v1_standard.json"
        existing.write_text(json.dumps({"run_id": "old_run"}), encoding="utf-8")

        result = freeze_baseline(run_dir, baselines_dir=baselines_dir, force=True)

        assert result.previous_existed is True
        assert result.backup_path is not None
        assert result.backup_path.is_file()

        # New baseline has new content
        frozen = json.loads(result.baseline_path.read_text())
        assert frozen["run_id"] == "new_run"

        # Backup has old content
        backup = json.loads(result.backup_path.read_text())
        assert backup["run_id"] == "old_run"

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        _write_summary(run_dir, _passing_summary())

        result = freeze_baseline(run_dir, baselines_dir=baselines_dir, dry_run=True)

        assert result.dry_run is True
        assert not result.baseline_path.exists()

    def test_dry_run_with_existing_baseline_does_not_raise(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        _write_summary(run_dir, _passing_summary())

        existing = baselines_dir / "heuristic_rgb_v1_standard.json"
        existing.write_text("{}", encoding="utf-8")

        # dry_run + force=False should NOT raise even if file exists
        result = freeze_baseline(run_dir, baselines_dir=baselines_dir, dry_run=True)
        assert result.dry_run is True
        assert result.previous_existed is True

    def test_missing_run_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FreezeValidationError, match="not found"):
            freeze_baseline(tmp_path / "missing_run")

    def test_failing_summary_raises_validation_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_fail"
        _write_summary(run_dir, _passing_summary(overall="fail", apcer="fail"))
        with pytest.raises(FreezeValidationError):
            freeze_baseline(run_dir, baselines_dir=tmp_path / "baselines")

    def test_result_to_dict(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        _write_summary(run_dir, _passing_summary())

        result = freeze_baseline(run_dir, baselines_dir=baselines_dir)
        d = result.to_dict()

        assert d["run_id"] == "run_001"
        assert d["dry_run"] is False
        assert d["backup_path"] is None
        assert "frozen_at_utc" in d


# ---------------------------------------------------------------------------
# Tests: main() CLI
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFreezeBaselineCLI:
    def test_cli_happy_path_exits_0(self, tmp_path: Path) -> None:
        from tools.freeze_baseline import main

        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        _write_summary(run_dir, _passing_summary())

        result = main([
            "--run-dir", str(run_dir),
            "--baselines-dir", str(baselines_dir),
        ])
        assert result == 0
        assert (baselines_dir / "heuristic_rgb_v1_standard.json").is_file()

    def test_cli_missing_run_dir_exits_1(self, tmp_path: Path) -> None:
        from tools.freeze_baseline import main

        result = main([
            "--run-dir", str(tmp_path / "missing"),
            "--baselines-dir", str(tmp_path / "baselines"),
        ])
        assert result == 1

    def test_cli_failing_summary_exits_1(self, tmp_path: Path) -> None:
        from tools.freeze_baseline import main

        run_dir = tmp_path / "run_fail"
        _write_summary(run_dir, _passing_summary(overall="fail", apcer="fail"))

        result = main([
            "--run-dir", str(run_dir),
            "--baselines-dir", str(tmp_path / "baselines"),
        ])
        assert result == 1

    def test_cli_existing_baseline_exits_2(self, tmp_path: Path) -> None:
        from tools.freeze_baseline import main

        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        _write_summary(run_dir, _passing_summary())
        (baselines_dir / "heuristic_rgb_v1_standard.json").write_text("{}", encoding="utf-8")

        result = main([
            "--run-dir", str(run_dir),
            "--baselines-dir", str(baselines_dir),
        ])
        assert result == 2

    def test_cli_force_flag_exits_0(self, tmp_path: Path) -> None:
        from tools.freeze_baseline import main

        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        _write_summary(run_dir, _passing_summary())
        (baselines_dir / "heuristic_rgb_v1_standard.json").write_text("{}", encoding="utf-8")

        result = main([
            "--run-dir", str(run_dir),
            "--baselines-dir", str(baselines_dir),
            "--force",
        ])
        assert result == 0

    def test_cli_dry_run_exits_0_no_file_written(self, tmp_path: Path) -> None:
        from tools.freeze_baseline import main

        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        _write_summary(run_dir, _passing_summary())

        result = main([
            "--run-dir", str(run_dir),
            "--baselines-dir", str(baselines_dir),
            "--dry-run",
        ])
        assert result == 0
        assert not (baselines_dir / "heuristic_rgb_v1_standard.json").exists()

    def test_cli_output_flag_writes_metadata(self, tmp_path: Path) -> None:
        from tools.freeze_baseline import main

        run_dir = tmp_path / "run_001"
        baselines_dir = tmp_path / "baselines"
        output = tmp_path / "freeze_meta.json"
        _write_summary(run_dir, _passing_summary(run_id="meta_test"))

        main([
            "--run-dir", str(run_dir),
            "--baselines-dir", str(baselines_dir),
            "--output", str(output),
        ])

        assert output.is_file()
        meta = json.loads(output.read_text())
        assert meta["run_id"] == "meta_test"
        assert "frozen_at_utc" in meta
