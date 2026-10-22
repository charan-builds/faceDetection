"""
Baseline freeze utility for Face Lock AI PAD evaluation.

Validates that a run directory contains a passing summary.json, then copies
it to eval/baselines/{model_id}_{assurance_tier}.json.

Usage
-----
    python -m tools.freeze_baseline --run-dir eval/reports/runs/<run_id>
    python -m tools.freeze_baseline --run-dir eval/reports/runs/<run_id> --force
    python -m tools.freeze_baseline --run-dir eval/reports/runs/<run_id> --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools.project_paths import resolve_project_root
except ModuleNotFoundError:
    import importlib.util as _ilu, sys as _sys
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from tools.project_paths import resolve_project_root


LOGGER = logging.getLogger("face_lock.freeze_baseline")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FreezeValidationError(ValueError):
    """Raised when the run directory or summary.json fails validation."""


class BaselineExistsError(FileExistsError):
    """Raised when a baseline file already exists and --force was not given."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FreezeResult:
    """Output of a successful or dry-run freeze operation."""

    run_id: str
    model_id: str
    assurance_tier: str
    source_path: Path
    baseline_path: Path
    backup_path: Path | None
    frozen_at_utc: str
    dry_run: bool
    previous_existed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "assurance_tier": self.assurance_tier,
            "source_path": str(self.source_path),
            "baseline_path": str(self.baseline_path),
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "frozen_at_utc": self.frozen_at_utc,
            "dry_run": self.dry_run,
            "previous_existed": self.previous_existed,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = (
    "run_id",
    "model_id",
    "assurance_tier",
    "apcer_overall",
    "bpcer_overall",
    "tier_gate",
)


def load_and_validate_summary(summary_path: Path) -> dict[str, Any]:
    """
    Load summary.json and verify it meets baseline freeze criteria.

    Raises:
        FreezeValidationError: If the file is missing, malformed, or failing.
    """
    if not summary_path.is_file():
        raise FreezeValidationError(f"summary.json not found: {summary_path}")

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FreezeValidationError(f"summary.json is not valid JSON: {error}") from error

    if not isinstance(data, dict):
        raise FreezeValidationError("summary.json must be a JSON object.")

    # Check required fields
    missing = [f for f in _REQUIRED_FIELDS if f not in data]
    if missing:
        raise FreezeValidationError(
            f"summary.json is missing required fields: {missing}"
        )

    # Validate tier_gate shape
    tier_gate = data.get("tier_gate")
    if not isinstance(tier_gate, dict):
        raise FreezeValidationError("summary.json tier_gate must be an object.")

    # Validate individual gates first so error messages are specific
    apcer_status = str(tier_gate.get("apcer", "")).lower()
    bpcer_status = str(tier_gate.get("bpcer", "")).lower()
    if apcer_status == "fail":
        raise FreezeValidationError(
            "APCER gate failed (tier_gate.apcer='fail'). Cannot freeze."
        )
    if bpcer_status == "fail":
        raise FreezeValidationError(
            "BPCER gate failed (tier_gate.bpcer='fail'). Cannot freeze."
        )

    overall = str(tier_gate.get("overall", "")).lower()
    if overall != "pass":
        latency = tier_gate.get("latency", "skipped")
        raise FreezeValidationError(
            f"Cannot freeze a failing run. "
            f"tier_gate.overall={overall!r} "
            f"(apcer={apcer_status}, bpcer={bpcer_status}, latency={latency}). "
            "Only runs with overall='pass' can be frozen as baselines."
        )

    return data


def baseline_filename(model_id: str, assurance_tier: str) -> str:
    """Return the canonical baseline filename for a model/tier pair."""
    # Sanitise: replace spaces with underscores, remove path separators
    safe_model = model_id.replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_tier = assurance_tier.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"{safe_model}_{safe_tier}.json"


# ---------------------------------------------------------------------------
# Core freeze logic
# ---------------------------------------------------------------------------

def freeze_baseline(
    run_dir: Path,
    *,
    baselines_dir: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> FreezeResult:
    """
    Validate a run directory and freeze its summary.json as an approved baseline.

    Args:
        run_dir:       E3 run directory containing summary.json.
        baselines_dir: Directory to write the baseline file.
                       Defaults to eval/baselines/ relative to project root.
        force:         Overwrite an existing baseline without error.
        dry_run:       Validate and report without writing anything.

    Returns:
        FreezeResult describing what was (or would be) done.

    Raises:
        FreezeValidationError:  Run is missing required files or failing gates.
        BaselineExistsError:    Baseline already exists and force=False.
    """
    if baselines_dir is None:
        baselines_dir = resolve_project_root() / "eval" / "baselines"

    summary_path = run_dir / "summary.json"
    summary = load_and_validate_summary(summary_path)

    model_id = str(summary["model_id"])
    assurance_tier = str(summary["assurance_tier"])
    run_id = str(summary["run_id"])

    filename = baseline_filename(model_id, assurance_tier)
    baseline_path = baselines_dir / filename

    now = datetime.now(timezone.utc).isoformat()
    previous_existed = baseline_path.is_file()
    backup_path: Path | None = None

    if previous_existed and not force and not dry_run:
        raise BaselineExistsError(
            f"Baseline already exists: {baseline_path}\n"
            "Use --force to overwrite it."
        )

    if dry_run:
        LOGGER.info(
            "DRY RUN: would freeze %s → %s (force=%s, previous_existed=%s)",
            summary_path, baseline_path, force, previous_existed,
        )
        return FreezeResult(
            run_id=run_id,
            model_id=model_id,
            assurance_tier=assurance_tier,
            source_path=summary_path,
            baseline_path=baseline_path,
            backup_path=None,
            frozen_at_utc=now,
            dry_run=True,
            previous_existed=previous_existed,
        )

    # Create baselines directory if needed
    baselines_dir.mkdir(parents=True, exist_ok=True)

    # Back up the existing baseline before overwriting
    if previous_existed:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = baseline_path.with_name(
            baseline_path.stem + f"_backup_{ts}" + baseline_path.suffix
        )
        shutil.copy2(baseline_path, backup_path)
        LOGGER.info("Backed up existing baseline to %s", backup_path)

    shutil.copy2(summary_path, baseline_path)
    LOGGER.info("Frozen baseline: %s → %s", summary_path, baseline_path)

    return FreezeResult(
        run_id=run_id,
        model_id=model_id,
        assurance_tier=assurance_tier,
        source_path=summary_path,
        baseline_path=baseline_path,
        backup_path=backup_path,
        frozen_at_utc=now,
        dry_run=False,
        previous_existed=previous_existed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = resolve_project_root()
    default_baselines = project_root / "eval" / "baselines"

    parser = argparse.ArgumentParser(
        description="Freeze an approved PAD evaluation run as a baseline.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="E3 run directory containing summary.json.",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=default_baselines,
        help=f"Directory to store baseline JSON files (default: {default_baselines}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing baseline. The old file is backed up first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show what would happen without writing any files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write freeze metadata JSON to this path.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit 0=success, 1=validation failure, 2=baseline exists."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    if not args.run_dir.is_dir():
        print(f"[ERROR] Run directory not found: {args.run_dir}", file=sys.stderr)
        return 1

    try:
        result = freeze_baseline(
            run_dir=args.run_dir,
            baselines_dir=args.baselines_dir,
            force=args.force,
            dry_run=args.dry_run,
        )
    except FreezeValidationError as error:
        print(f"[ERROR] Validation failed: {error}", file=sys.stderr)
        return 1
    except BaselineExistsError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2

    # Console output
    if result.dry_run:
        print(f"[DRY RUN] Would freeze: {result.source_path}")
        print(f"[DRY RUN]   → {result.baseline_path}")
        if result.previous_existed:
            print(f"[DRY RUN]   Previous baseline would be backed up first.")
    else:
        print(f"[OK] Baseline frozen: {result.baseline_path}")
        if result.backup_path:
            print(f"[OK] Previous baseline backed up to: {result.backup_path}")

    print(f"      model_id       : {result.model_id}")
    print(f"      assurance_tier : {result.assurance_tier}")
    print(f"      run_id         : {result.run_id}")
    print(f"      frozen_at      : {result.frozen_at_utc}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[OK] Metadata written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
