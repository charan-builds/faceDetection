"""
Reproducible PAD settings and environment snapshots for evaluation runs.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PADSettings, PAD_SETTINGS


@dataclass(frozen=True, slots=True)
class EnvironmentMetadata:
    """Captured host and toolchain metadata for audit reproducibility."""

    python_version: str
    platform: str
    opencv_version: str | None
    git_sha: str | None
    git_dirty: bool | None


def capture_environment() -> EnvironmentMetadata:
    """Collect Python, platform, OpenCV, and git metadata."""
    opencv_version: str | None
    try:
        import cv2

        opencv_version = cv2.__version__
    except ImportError:
        opencv_version = None

    git_sha, git_dirty = _read_git_metadata()
    return EnvironmentMetadata(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        opencv_version=opencv_version,
        git_sha=git_sha,
        git_dirty=git_dirty,
    )


def _read_git_metadata() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return sha.stdout.strip(), bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def pad_settings_to_dict(settings: PADSettings) -> dict[str, Any]:
    """Serialize PADSettings to a JSON-friendly dict."""
    return asdict(settings)


def merge_pad_settings(protocol_overrides: dict[str, Any] | None) -> PADSettings:
    """
    Build effective PADSettings by overlaying protocol YAML values on defaults.

    Unknown keys are ignored so protocol files can carry non-PAD metadata.
    """
    if not protocol_overrides:
        return PAD_SETTINGS

    valid_fields = {field.name for field in fields(PADSettings)}
    filtered = {
        key: value
        for key, value in protocol_overrides.items()
        if key in valid_fields
    }
    return replace(PAD_SETTINGS, **filtered)


def build_settings_snapshot(
    *,
    run_id: str,
    protocol_id: str,
    protocol_version: str,
    evaluation_policy: str,
    pad_settings: PADSettings,
    collection_id: str,
    manifest_version: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a reproducible settings snapshot document."""
    environment = capture_environment()
    snapshot: dict[str, Any] = {
        "snapshot_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "protocol_id": protocol_id,
        "protocol_version": protocol_version,
        "evaluation_policy": evaluation_policy,
        "collection_id": collection_id,
        "manifest_version": manifest_version,
        "pad_settings": pad_settings_to_dict(pad_settings),
        "environment": asdict(environment),
    }
    if extra:
        snapshot["extra"] = extra
    return snapshot


def write_settings_snapshot(path: Path, snapshot: dict[str, Any]) -> Path:
    """Write snapshot JSON with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_settings_snapshot(path: Path) -> dict[str, Any]:
    """Load a previously written settings snapshot."""
    return json.loads(path.read_text(encoding="utf-8"))
