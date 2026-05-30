"""Tests for eval.harness.run_pad_eval."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("cv2")
import cv2  # noqa: E402

from eval.harness.clip_loader import frames_from_arrays
from eval.harness.run_pad_eval import (
    DatasetManifest,
    ManifestSample,
    PadEvalConfig,
    PresentationResult,
    aggregate_metrics,
    execute_presentation,
    load_manifest_json,
    load_protocol_yaml,
    run_iso_lab,
    run_pad_evaluation,
)
from eval.harness.settings_snapshot import merge_pad_settings
from src.config import PAD_SETTINGS, resolve_project_root
from src.pad_engine import PadOutcome

PROJECT_ROOT = resolve_project_root()


def _write_frame(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((48, 48, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _bright_frames(count: int = 6) -> tuple:
    return tuple(np.full((48, 48, 3), 220, dtype=np.uint8) for _ in range(count))


def _mock_settings(model_id: str, *, min_frames: int = 2) -> object:
    return replace(
        PAD_SETTINGS,
        enabled=True,
        mode="passive_only",
        passive_model_id=model_id,
        min_frames_passive=min_frames,
        max_frames_buffer=24,
    )


@pytest.fixture
def iso_protocol_path(tmp_path: Path) -> Path:
    source = PROJECT_ROOT / "eval" / "protocols" / "pad_iso_lab_v1.yaml"
    protocol = yaml.safe_load(source.read_text(encoding="utf-8"))
    protocol["pad_settings"] = {
        "enabled": True,
        "mode": "passive_only",
        "passive_model_id": "mock_always_attack",
        "min_frames_passive": 2,
        "max_frames_buffer": 24,
        "assurance_tier": "standard",
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.dump(protocol), encoding="utf-8")
    return path


@pytest.fixture
def tiny_manifest(tmp_path: Path) -> Path:
    manifest = {
        "manifest_version": "1.0.0",
        "collection_id": "test_tiny_v1",
        "samples": [
            {
                "sample_id": "ATK_001",
                "pai_id": "PA-01",
                "ground_truth": "attack",
                "subject_id": "subj_001",
                "path": "attacks/session_01",
                "presentation_type": "clip",
                "frame_count": 2,
            },
            {
                "sample_id": "BF_001",
                "pai_id": "BF",
                "ground_truth": "bona_fide",
                "subject_id": "subj_002",
                "path": "bona_fide/session_01",
                "presentation_type": "clip",
                "frame_count": 2,
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def collection_root(tmp_path: Path) -> Path:
    attack_session = tmp_path / "attacks" / "session_01"
    bona_session = tmp_path / "bona_fide" / "session_01"
    for index in range(1, 3):
        _write_frame(attack_session / f"frame_{index:04d}.jpg", 30)
        _write_frame(bona_session / f"frame_{index:04d}.jpg", 220)
    return tmp_path


def test_load_protocol_yaml() -> None:
    protocol = load_protocol_yaml(PROJECT_ROOT / "eval" / "protocols" / "pad_iso_lab_v1.yaml")
    assert protocol.protocol_id == "PAD-ISO-LAB-v1"
    assert protocol.entrypoint == "PadSession.evaluate_final"
    assert protocol.evaluation_policy == "fail_closed_v1"


def test_load_protocol_registration_entrypoint() -> None:
    protocol = load_protocol_yaml(
        PROJECT_ROOT / "eval" / "protocols" / "pad_registration_v1.yaml"
    )
    assert protocol.entrypoint == "check_single_image_pad"


def test_load_protocol_live_sim_entrypoint() -> None:
    protocol = load_protocol_yaml(
        PROJECT_ROOT / "eval" / "protocols" / "pad_live_sim_v1.yaml"
    )
    assert protocol.entrypoint == "live_sim_session"


def test_load_manifest_json(tiny_manifest: Path) -> None:
    manifest = load_manifest_json(tiny_manifest)
    assert manifest.collection_id == "test_tiny_v1"
    assert len(manifest.samples) == 2


def test_load_manifest_rejects_invalid_ground_truth(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "manifest_version": "1.0.0",
                "collection_id": "x",
                "samples": [
                    {
                        "sample_id": "1",
                        "pai_id": "BF",
                        "ground_truth": "spoof",
                        "subject_id": "subj_001",
                        "path": "a",
                        "presentation_type": "clip",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ground_truth"):
        load_manifest_json(path)


def test_run_iso_lab_mock_outcomes() -> None:
    settings = _mock_settings("mock_always_live")
    presentation = frames_from_arrays(_bright_frames(4))
    result = run_iso_lab(presentation, settings)  # type: ignore[arg-type]
    assert result.outcome == PadOutcome.LIVE


def test_execute_presentation_attack_mock(iso_protocol_path: Path) -> None:
    protocol = load_protocol_yaml(iso_protocol_path)
    pad_settings = merge_pad_settings(protocol.pad_settings)
    sample = ManifestSample(
        sample_id="ATK_001",
        pai_id="PA-01",
        ground_truth="attack",
        subject_id="subj_001",
        path="attacks/session_01",
        presentation_type="clip",
    )
    presentation = frames_from_arrays(_bright_frames(4))
    row = execute_presentation(sample, presentation, protocol, pad_settings)
    assert row.predicted_outcome == "ATTACK"
    assert row.normalized_outcome == "ATTACK"


def test_aggregate_metrics() -> None:
    protocol = load_protocol_yaml(PROJECT_ROOT / "eval" / "protocols" / "pad_iso_lab_v1.yaml")
    rows = (
        PresentationResult(
            sample_id="A1",
            pai_id="PA-01",
            ground_truth="attack",
            subject_id="subj_001",
            predicted_outcome="LIVE",
            normalized_outcome="LIVE",
            reason_code="passive_live",
            model_id="mock",
            model_version="1.0.0",
            latency_ms=10.0,
            frames_analyzed=4,
            entrypoint="PadSession.evaluate_final",
        ),
        PresentationResult(
            sample_id="B1",
            pai_id="BF",
            ground_truth="bona_fide",
            subject_id="subj_002",
            predicted_outcome="ATTACK",
            normalized_outcome="ATTACK",
            reason_code="passive_attack",
            model_id="mock",
            model_version="1.0.0",
            latency_ms=12.0,
            frames_analyzed=4,
            entrypoint="PadSession.evaluate_final",
        ),
    )
    metrics = aggregate_metrics(rows, protocol)
    assert metrics.apcer == 1.0
    assert metrics.bpcer == 1.0
    assert metrics.confusion["fn"] == 1
    assert metrics.confusion["fp"] == 1
    assert metrics.latency is not None
    assert metrics.tier_gate is not None
    assert metrics.tier_gate.overall == "fail"


def test_run_pad_evaluation_end_to_end(
    iso_protocol_path: Path,
    tiny_manifest: Path,
    collection_root: Path,
    tmp_path: Path,
) -> None:
    config = PadEvalConfig(
        protocol_path=iso_protocol_path,
        manifest_path=tiny_manifest,
        collection_root=collection_root,
        output_dir=tmp_path / "runs",
        run_id="test_run_e2e",
    )
    result = run_pad_evaluation(config)

    assert result.run_id == "test_run_e2e"
    assert len(result.presentations) == 2
    assert result.metrics.apcer == 0.0
    assert result.metrics.bpcer == 1.0
    assert result.settings_snapshot_path.is_file()
    assert result.results_jsonl_path.is_file()
    assert result.summary_json_path.is_file()

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert summary["protocol_id"] == "PAD-ISO-LAB-v1"
    assert summary["apcer_overall"] == 0.0
    assert summary["bpcer_overall"] == 1.0

    lines = result.results_jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_run_pad_evaluation_skip_missing_media(
    iso_protocol_path: Path,
    tiny_manifest: Path,
    tmp_path: Path,
) -> None:
    config = PadEvalConfig(
        protocol_path=iso_protocol_path,
        manifest_path=tiny_manifest,
        collection_root=tmp_path / "empty",
        output_dir=tmp_path / "runs",
        run_id="skip_run",
        skip_missing_media=True,
    )
    with pytest.raises(ValueError, match="No presentations evaluated"):
        run_pad_evaluation(config)


def test_inconclusive_normalized_for_metrics() -> None:
    protocol = load_protocol_yaml(PROJECT_ROOT / "eval" / "protocols" / "pad_iso_lab_v1.yaml")
    rows = (
        PresentationResult(
            sample_id="A1",
            pai_id="PA-01",
            ground_truth="attack",
            subject_id="subj_001",
            predicted_outcome="INCONCLUSIVE",
            normalized_outcome="ATTACK",
            reason_code="passive_inconclusive",
            model_id="mock",
            model_version="1.0.0",
            latency_ms=5.0,
            frames_analyzed=2,
            entrypoint="PadSession.evaluate_final",
        ),
    )
    metrics = aggregate_metrics(rows, protocol)
    assert metrics.apcer == 0.0
