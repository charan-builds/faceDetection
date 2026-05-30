"""Tests for eval.harness.settings_snapshot."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from eval.harness.settings_snapshot import (
    build_settings_snapshot,
    capture_environment,
    merge_pad_settings,
    pad_settings_to_dict,
    read_settings_snapshot,
    write_settings_snapshot,
)
from src.config import PAD_SETTINGS


def test_pad_settings_to_dict_roundtrip_keys() -> None:
    data = pad_settings_to_dict(PAD_SETTINGS)
    assert data["passive_model_id"] == PAD_SETTINGS.passive_model_id
    assert data["min_frames_passive"] == PAD_SETTINGS.min_frames_passive


def test_merge_pad_settings_overrides() -> None:
    merged = merge_pad_settings(
        {
            "passive_model_id": "mock_always_live",
            "min_frames_passive": 2,
            "unknown_key": "ignored",
        }
    )
    assert merged.passive_model_id == "mock_always_live"
    assert merged.min_frames_passive == 2
    assert merged.passive_pass_threshold == PAD_SETTINGS.passive_pass_threshold


def test_merge_pad_settings_empty_returns_default() -> None:
    assert merge_pad_settings(None) is PAD_SETTINGS
    assert merge_pad_settings({}) == PAD_SETTINGS


def test_capture_environment() -> None:
    env = capture_environment()
    assert env.python_version
    assert env.platform


def test_build_and_write_snapshot(tmp_path: Path) -> None:
    settings = replace(PAD_SETTINGS, passive_model_id="mock_always_attack")
    snapshot = build_settings_snapshot(
        run_id="test_run",
        protocol_id="PAD-ISO-LAB-v1",
        protocol_version="1.0.0",
        evaluation_policy="fail_closed_v1",
        pad_settings=settings,
        collection_id="test_collection",
        manifest_version="1.0.0",
    )
    path = write_settings_snapshot(tmp_path / "settings_snapshot.json", snapshot)
    loaded = read_settings_snapshot(path)
    assert loaded["run_id"] == "test_run"
    assert loaded["pad_settings"]["passive_model_id"] == "mock_always_attack"
    assert loaded["environment"]["python_version"]

    # Stable JSON formatting
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed == loaded
