"""
Unit tests for eval.tools.dataset_status.

All tests are fast and offline — no real dataset, webcam, or DeepFace.
Protocol YAML and collection files are created in tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from eval.tools.dataset_status import (
    CategoryProgress,
    DatasetStatus,
    _count_from_manifest,
    _count_from_qc,
    _load_protocol_requirements,
    compute_status,
    print_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_protocol_yaml(tmp_path: Path) -> Path:
    """Write a minimal protocol YAML matching the real pad_iso_lab_v1 structure."""
    protocol = {
        "protocol_id": "PAD-ISO-LAB-v1",
        "version": "1.0.0",
        "sample_sizes": {
            "bona_fide": {"min_presentations": 150, "min_subjects": 30},
            "attacks": {
                "PA-01": {"min_presentations": 50, "min_subjects": 10},
                "PA-02": {"min_presentations": 50, "min_subjects": 10},
                "PA-03": {"min_presentations": 50, "min_subjects": 10},
                "PA-04": {"min_presentations": 30, "min_subjects": 6},
            },
        },
        "tier_gates": {
            "standard": {"apcer_max_pooled": 0.05, "bpcer_max": 0.05, "latency_ms_p95_max": 1200}
        },
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.dump(protocol), encoding="utf-8")
    return path


def _make_manifest(tmp_path: Path, samples: list[dict[str, Any]]) -> Path:
    doc = {
        "manifest_version": "1.0.0",
        "collection_id": "test_collection",
        "samples": samples,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def _make_qc_report(
    tmp_path: Path,
    decision: str = "WARNING",
    category_counts: dict[str, int] | None = None,
    subject_counts: dict[str, int] | None = None,
    error_count: int = 0,
    warning_count: int = 5,
) -> Path:
    report = {
        "collection_id": "test_collection",
        "decision": decision,
        "errors": [{"code": "e"} for _ in range(error_count)],
        "warnings": [{"code": "w"} for _ in range(warning_count)],
        "summary": {
            "category_counts": category_counts or {},
            "subject_counts": subject_counts or {},
            "error_count": error_count,
            "warning_count": warning_count,
        },
    }
    path = tmp_path / "dataset_qc_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _bf_sample(subject: str, session: str) -> dict[str, Any]:
    return {
        "sample_id": f"BF_{subject}_{session}",
        "pai_id": "BF",
        "ground_truth": "bona_fide",
        "subject_id": subject,
        "path": f"bona_fide/{subject}/{session}",
        "presentation_type": "clip",
    }


def _attack_sample(pai_id: str, subject: str, attempt: str) -> dict[str, Any]:
    folder = pai_id.replace("-", "_").lower()
    return {
        "sample_id": f"{pai_id}_{subject}_{attempt}",
        "pai_id": pai_id,
        "ground_truth": "attack",
        "subject_id": subject,
        "path": f"attacks/{folder}/{subject}/{attempt}",
        "presentation_type": "clip",
    }


# ---------------------------------------------------------------------------
# Tests: _load_protocol_requirements
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadProtocolRequirements:
    def test_reads_bona_fide_minimums(self, tmp_path: Path) -> None:
        protocol = _make_protocol_yaml(tmp_path)
        reqs = _load_protocol_requirements(protocol)
        assert reqs["BF"]["presentations"] == 150
        assert reqs["BF"]["subjects"] == 30

    def test_reads_attack_minimums(self, tmp_path: Path) -> None:
        protocol = _make_protocol_yaml(tmp_path)
        reqs = _load_protocol_requirements(protocol)
        assert reqs["PA-01"]["presentations"] == 50
        assert reqs["PA-02"]["subjects"] == 10

    def test_missing_section_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "minimal.yaml"
        path.write_text(yaml.dump({"protocol_id": "X"}), encoding="utf-8")
        reqs = _load_protocol_requirements(path)
        assert reqs.get("BF", {}).get("presentations", 0) == 0


# ---------------------------------------------------------------------------
# Tests: _count_from_manifest
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCountFromManifest:
    def test_counts_presentations_per_pai(self) -> None:
        samples = [
            _bf_sample("subj_001", "s01"),
            _bf_sample("subj_001", "s02"),
            _bf_sample("subj_002", "s01"),
            _attack_sample("PA-01", "subj_003", "a01"),
        ]
        manifest = {"samples": samples}
        pres, subjs = _count_from_manifest(manifest)
        assert pres["BF"] == 3
        assert pres["PA-01"] == 1

    def test_counts_unique_subjects(self) -> None:
        samples = [
            _bf_sample("subj_001", "s01"),
            _bf_sample("subj_001", "s02"),  # same subject, different session
            _bf_sample("subj_002", "s01"),
        ]
        manifest = {"samples": samples}
        _, subjs = _count_from_manifest(manifest)
        assert len(subjs["BF"]) == 2  # two unique subjects

    def test_empty_samples_returns_zero(self) -> None:
        pres, subjs = _count_from_manifest({"samples": []})
        assert pres == {}
        assert subjs == {}

    def test_missing_pai_id_skipped(self) -> None:
        manifest = {"samples": [{"subject_id": "subj_001"}]}
        pres, _ = _count_from_manifest(manifest)
        assert pres == {}


# ---------------------------------------------------------------------------
# Tests: _count_from_qc
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCountFromQC:
    def test_reads_category_and_subject_counts(self) -> None:
        qc: dict[str, Any] = {
            "summary": {
                "category_counts": {"BF": 4, "PA-02": 1},
                "subject_counts": {"BF": 2, "PA-02": 1},
            }
        }
        pres, subjs = _count_from_qc(qc)
        assert pres["BF"] == 4
        assert subjs["PA-02"] == 1

    def test_missing_summary_returns_empty(self) -> None:
        pres, subjs = _count_from_qc({})
        assert pres == {}
        assert subjs == {}


# ---------------------------------------------------------------------------
# Tests: CategoryProgress
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCategoryProgress:
    def test_presentations_remaining_capped_at_zero(self) -> None:
        cat = CategoryProgress("BF", "Bona fide", 200, 150, 35, 30)
        assert cat.presentations_remaining == 0

    def test_presentations_remaining_positive(self) -> None:
        cat = CategoryProgress("PA-01", "Print", 10, 50, 2, 10)
        assert cat.presentations_remaining == 40

    def test_presentation_pct_caps_at_100(self) -> None:
        cat = CategoryProgress("BF", "Bona fide", 200, 150, 35, 30)
        assert cat.presentation_pct == 100.0

    def test_presentation_pct_partial(self) -> None:
        cat = CategoryProgress("PA-01", "Print", 25, 50, 5, 10)
        assert cat.presentation_pct == pytest.approx(50.0)

    def test_is_complete_true_when_both_met(self) -> None:
        cat = CategoryProgress("BF", "Bona fide", 150, 150, 30, 30)
        assert cat.is_complete is True

    def test_is_complete_false_when_subjects_short(self) -> None:
        cat = CategoryProgress("BF", "Bona fide", 150, 150, 25, 30)
        assert cat.is_complete is False

    def test_is_complete_false_when_presentations_short(self) -> None:
        cat = CategoryProgress("BF", "Bona fide", 100, 150, 30, 30)
        assert cat.is_complete is False


# ---------------------------------------------------------------------------
# Tests: compute_status
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeStatus:
    def test_pilot_scale_returns_low_readiness(self, tmp_path: Path) -> None:
        """Current face_lock_v1 data: 4 BF, 1 PA-02 → very low readiness."""
        protocol = _make_protocol_yaml(tmp_path)
        samples = [
            _bf_sample("subj_001", "s01"),
            _bf_sample("subj_001", "s02"),
            _bf_sample("subj_002", "s01"),
            _bf_sample("subj_002", "s02"),
            _attack_sample("PA-02", "subj_001", "a01"),
        ]
        _make_manifest(tmp_path, samples)
        _make_qc_report(tmp_path, decision="WARNING", error_count=0, warning_count=13)

        status = compute_status(tmp_path, protocol)

        assert status.overall_readiness_pct < 10.0
        assert status.ready_for_baseline is False

        bf = next(c for c in status.categories if c.pai_id == "BF")
        assert bf.collected_presentations == 4
        assert bf.required_presentations == 150

    def test_full_dataset_returns_100_pct(self, tmp_path: Path) -> None:
        """When all required categories are met and QC passes."""
        protocol = _make_protocol_yaml(tmp_path)

        samples = []
        # 150 BF across 30 subjects (5 sessions each)
        for s in range(1, 31):
            for sess in range(1, 6):
                samples.append(_bf_sample(f"subj_{s:03d}", f"s{sess:02d}"))
        # 50 each for PA-01/02/03 across 10 subjects
        for pai in ("PA-01", "PA-02", "PA-03"):
            for s in range(1, 11):
                for a in range(1, 6):
                    samples.append(_attack_sample(pai, f"subj_{s:03d}", f"a{a:02d}"))

        _make_manifest(tmp_path, samples)
        _make_qc_report(tmp_path, decision="PASS", error_count=0, warning_count=0)

        status = compute_status(tmp_path, protocol)

        assert status.overall_readiness_pct == pytest.approx(100.0)
        assert status.ready_for_baseline is True

    def test_qc_errors_block_baseline_readiness(self, tmp_path: Path) -> None:
        """Even with enough samples, QC errors should block ready_for_baseline."""
        protocol = _make_protocol_yaml(tmp_path)
        samples = []
        for s in range(1, 31):
            for sess in range(1, 6):
                samples.append(_bf_sample(f"subj_{s:03d}", f"s{sess:02d}"))
        for pai in ("PA-01", "PA-02", "PA-03"):
            for s in range(1, 11):
                for a in range(1, 6):
                    samples.append(_attack_sample(pai, f"subj_{s:03d}", f"a{a:02d}"))

        _make_manifest(tmp_path, samples)
        _make_qc_report(tmp_path, decision="FAIL", error_count=3, warning_count=0)

        status = compute_status(tmp_path, protocol)
        assert status.ready_for_baseline is False

    def test_no_manifest_falls_back_to_qc_counts(self, tmp_path: Path) -> None:
        """When no manifest.json, counts come from dataset_qc_report.json."""
        protocol = _make_protocol_yaml(tmp_path)
        _make_qc_report(
            tmp_path,
            category_counts={"BF": 50, "PA-01": 10},
            subject_counts={"BF": 15, "PA-01": 3},
        )

        status = compute_status(tmp_path, protocol)

        bf = next(c for c in status.categories if c.pai_id == "BF")
        pa01 = next(c for c in status.categories if c.pai_id == "PA-01")
        assert bf.collected_presentations == 50
        assert pa01.collected_presentations == 10
        assert status.manifest_found is False

    def test_no_manifest_no_qc_returns_zero_counts(self, tmp_path: Path) -> None:
        """When neither file exists, all counts are zero."""
        protocol = _make_protocol_yaml(tmp_path)
        status = compute_status(tmp_path, protocol)
        assert status.overall_readiness_pct == 0.0
        for cat in status.categories:
            assert cat.collected_presentations == 0

    def test_collection_id_uses_directory_name_by_default(self, tmp_path: Path) -> None:
        protocol = _make_protocol_yaml(tmp_path)
        # tmp_path name varies; just confirm it matches the dir name
        status = compute_status(tmp_path, protocol)
        assert status.collection_id == tmp_path.name

    def test_collection_id_override(self, tmp_path: Path) -> None:
        protocol = _make_protocol_yaml(tmp_path)
        status = compute_status(tmp_path, protocol, collection_id="my_collection")
        assert status.collection_id == "my_collection"

    def test_to_dict_includes_computed_fields(self, tmp_path: Path) -> None:
        protocol = _make_protocol_yaml(tmp_path)
        status = compute_status(tmp_path, protocol)
        d = status.to_dict()
        assert "overall_readiness_pct" in d
        assert "categories" in d
        # Each category dict should have computed fields
        for cat_d in d["categories"]:
            assert "presentations_remaining" in cat_d
            assert "presentation_pct" in cat_d
            assert "is_complete" in cat_d

    def test_manifest_version_captured(self, tmp_path: Path) -> None:
        protocol = _make_protocol_yaml(tmp_path)
        _make_manifest(tmp_path, [_bf_sample("subj_001", "s01")])
        status = compute_status(tmp_path, protocol)
        assert status.manifest_version == "1.0.0"
        assert status.manifest_found is True


# ---------------------------------------------------------------------------
# Tests: print_status (smoke — just ensure no exception)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPrintStatus:
    def test_prints_without_exception(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        protocol = _make_protocol_yaml(tmp_path)
        _make_manifest(tmp_path, [_bf_sample("subj_001", "s01")])
        status = compute_status(tmp_path, protocol)
        print_status(status)
        out = capsys.readouterr().out
        assert "Dataset readiness" in out
        assert "BF" in out
        assert "PA-01" in out

    def test_output_contains_readiness_percentage(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        protocol = _make_protocol_yaml(tmp_path)
        status = compute_status(tmp_path, protocol)
        print_status(status)
        out = capsys.readouterr().out
        assert "0.0%" in out or "readiness" in out.lower()


# ---------------------------------------------------------------------------
# Tests: main() CLI
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDatasetStatusCLI:
    def test_missing_collection_root_exits_2(self, tmp_path: Path) -> None:
        from eval.tools.dataset_status import main

        result = main(["--collection-root", str(tmp_path / "missing")])
        assert result == 2

    def test_output_flag_writes_json(self, tmp_path: Path) -> None:
        from eval.tools.dataset_status import main

        protocol = _make_protocol_yaml(tmp_path)
        collection = tmp_path / "collection"
        collection.mkdir()
        output = tmp_path / "status.json"

        main([
            "--collection-root", str(collection),
            "--protocol", str(protocol),
            "--output", str(output),
            "--quiet",
        ])

        assert output.is_file()
        data = json.loads(output.read_text())
        assert "overall_readiness_pct" in data
        assert "categories" in data

    def test_not_ready_exits_1(self, tmp_path: Path) -> None:
        from eval.tools.dataset_status import main

        protocol = _make_protocol_yaml(tmp_path)
        collection = tmp_path / "empty_collection"
        collection.mkdir()

        result = main(["--collection-root", str(collection), "--protocol", str(protocol), "--quiet"])
        assert result == 1

    def test_ready_exits_0(self, tmp_path: Path) -> None:
        from eval.tools.dataset_status import main

        protocol = _make_protocol_yaml(tmp_path)
        collection = tmp_path / "full_collection"
        collection.mkdir()

        samples = []
        for s in range(1, 31):
            for sess in range(1, 6):
                samples.append(_bf_sample(f"subj_{s:03d}", f"s{sess:02d}"))
        for pai in ("PA-01", "PA-02", "PA-03"):
            for s in range(1, 11):
                for a in range(1, 6):
                    samples.append(_attack_sample(pai, f"subj_{s:03d}", f"a{a:02d}"))

        _make_manifest(collection, samples)
        _make_qc_report(collection, decision="PASS", error_count=0, warning_count=0)

        result = main(["--collection-root", str(collection), "--protocol", str(protocol), "--quiet"])
        assert result == 0
