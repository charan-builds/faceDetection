"""
Dataset collection dashboard for Face Lock AI PAD evaluation.

Reads manifest.json, dataset_qc_report.json, and the PAD protocol YAML to
display collection progress and readiness percentage toward baseline approval.

Usage
-----
    python -m eval.tools.dataset_status
    python -m eval.tools.dataset_status --collection-root eval/datasets/face_lock_v1
    python -m eval.tools.dataset_status --output dataset_status.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import yaml

try:
    from tools.project_paths import resolve_project_root
except ModuleNotFoundError:
    import importlib.util as _ilu, sys as _sys
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from tools.project_paths import resolve_project_root


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION_ID: Final[str] = "face_lock_v1"
DEFAULT_PROTOCOL: Final[str] = "eval/protocols/pad_iso_lab_v1.yaml"

# PAI IDs that count toward the pooled APCER gate (required for baseline).
REQUIRED_PAI_IDS: Final[tuple[str, ...]] = ("BF", "PA-01", "PA-02", "PA-03")
OPTIONAL_PAI_IDS: Final[tuple[str, ...]] = ("PA-04",)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CategoryProgress:
    """Progress for a single PAI category."""

    pai_id: str
    label: str
    collected_presentations: int
    required_presentations: int
    collected_subjects: int
    required_subjects: int

    @property
    def presentations_remaining(self) -> int:
        return max(0, self.required_presentations - self.collected_presentations)

    @property
    def subjects_remaining(self) -> int:
        return max(0, self.required_subjects - self.collected_subjects)

    @property
    def presentation_pct(self) -> float:
        if self.required_presentations == 0:
            return 100.0
        return min(100.0, self.collected_presentations / self.required_presentations * 100.0)

    @property
    def is_complete(self) -> bool:
        return (
            self.collected_presentations >= self.required_presentations
            and self.collected_subjects >= self.required_subjects
        )


@dataclass
class DatasetStatus:
    """Full dataset collection status."""

    collection_id: str
    collection_root: str
    generated_at_utc: str
    protocol_id: str

    # Per-category progress for required PAIs
    categories: list[CategoryProgress] = field(default_factory=list)

    # QC state
    qc_decision: str = "UNKNOWN"
    qc_error_count: int = 0
    qc_warning_count: int = 0

    # Computed readiness
    overall_readiness_pct: float = 0.0
    ready_for_baseline: bool = False

    # Manifest info
    manifest_found: bool = False
    manifest_version: str = ""
    total_presentations_in_manifest: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # flatten CategoryProgress list to plain dicts with extra computed fields
        d["categories"] = [
            {
                **asdict(c),
                "presentations_remaining": c.presentations_remaining,
                "subjects_remaining": c.subjects_remaining,
                "presentation_pct": round(c.presentation_pct, 1),
                "is_complete": c.is_complete,
            }
            for c in self.categories
        ]
        return d


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

_PAI_LABELS: dict[str, str] = {
    "BF": "Bona fide",
    "PA-01": "Print",
    "PA-02": "Static screen",
    "PA-03": "Video replay",
    "PA-04": "Virtual camera",
}


def _load_protocol_requirements(protocol_path: Path) -> dict[str, dict[str, int]]:
    """
    Parse sample_sizes from the protocol YAML.

    Returns a dict keyed by PAI ID:
        {"BF": {"presentations": 150, "subjects": 30}, "PA-01": {...}, ...}
    """
    raw = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    sizes = raw.get("sample_sizes", {})

    result: dict[str, dict[str, int]] = {}

    bf = sizes.get("bona_fide", {})
    result["BF"] = {
        "presentations": int(bf.get("min_presentations", 0)),
        "subjects": int(bf.get("min_subjects", 0)),
    }

    attacks = sizes.get("attacks", {})
    for pai_id, spec in attacks.items():
        if isinstance(spec, dict):
            result[pai_id] = {
                "presentations": int(spec.get("min_presentations", 0)),
                "subjects": int(spec.get("min_subjects", 0)),
            }

    return result


def _load_manifest(collection_root: Path) -> dict[str, Any] | None:
    """Load manifest.json if it exists, otherwise return None."""
    manifest_path = collection_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_qc_report(collection_root: Path) -> dict[str, Any] | None:
    """Load dataset_qc_report.json if it exists, otherwise return None."""
    qc_path = collection_root / "dataset_qc_report.json"
    if not qc_path.is_file():
        return None
    return json.loads(qc_path.read_text(encoding="utf-8"))


def _count_from_manifest(
    manifest: dict[str, Any],
) -> tuple[dict[str, int], dict[str, set[str]]]:
    """
    Count presentations and unique subjects per PAI from manifest samples.

    Returns:
        presentation_counts: {pai_id: count}
        subject_sets:        {pai_id: set of subject_ids}
    """
    pres: dict[str, int] = {}
    subjs: dict[str, set[str]] = {}

    for sample in manifest.get("samples", []):
        if not isinstance(sample, dict):
            continue
        pai_id = str(sample.get("pai_id", ""))
        subject_id = str(sample.get("subject_id", ""))
        if not pai_id:
            continue
        pres[pai_id] = pres.get(pai_id, 0) + 1
        subjs.setdefault(pai_id, set()).add(subject_id)

    return pres, subjs


def _count_from_qc(
    qc: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Extract presentation and subject counts from QC report summary.

    Returns:
        pres_counts:  {pai_id: count}
        subj_counts:  {pai_id: count}
    """
    summary = qc.get("summary", {})
    pres = {k: int(v) for k, v in summary.get("category_counts", {}).items()}
    subjs = {k: int(v) for k, v in summary.get("subject_counts", {}).items()}
    return pres, subjs


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_status(
    collection_root: Path,
    protocol_path: Path,
    collection_id: str | None = None,
) -> DatasetStatus:
    """
    Compute the full dataset collection status.

    Args:
        collection_root: Root directory of the dataset collection.
        protocol_path:   Path to the PAD protocol YAML.
        collection_id:   Override collection ID (defaults to directory name).

    Returns:
        DatasetStatus dataclass.
    """
    cid = collection_id or collection_root.name or DEFAULT_COLLECTION_ID
    now = datetime.now(timezone.utc).isoformat()

    # Load protocol requirements
    requirements = _load_protocol_requirements(protocol_path)
    protocol_raw = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol_id = str(protocol_raw.get("protocol_id", "unknown"))

    # Load manifest and QC report
    manifest = _load_manifest(collection_root)
    qc = _load_qc_report(collection_root)

    # Resolve presentation and subject counts — prefer manifest, fall back to QC
    if manifest is not None:
        pres_counts, subj_sets = _count_from_manifest(manifest)
        subj_counts = {k: len(v) for k, v in subj_sets.items()}
        total_in_manifest = len(manifest.get("samples", []))
        manifest_found = True
        manifest_version = str(manifest.get("manifest_version", ""))
    elif qc is not None:
        pres_counts, subj_counts = _count_from_qc(qc)
        total_in_manifest = 0
        manifest_found = False
        manifest_version = ""
    else:
        pres_counts = {}
        subj_counts = {}
        total_in_manifest = 0
        manifest_found = False
        manifest_version = ""

    # QC status
    if qc is not None:
        qc_decision = str(qc.get("decision", "UNKNOWN"))
        qc_summary = qc.get("summary", {})
        qc_error_count = int(qc_summary.get("error_count", len(qc.get("errors", []))))
        qc_warning_count = int(qc_summary.get("warning_count", len(qc.get("warnings", []))))
    else:
        qc_decision = "NOT_RUN"
        qc_error_count = 0
        qc_warning_count = 0

    # Build per-category progress
    categories: list[CategoryProgress] = []
    all_pai_ids = list(REQUIRED_PAI_IDS) + [p for p in OPTIONAL_PAI_IDS if p in requirements]

    for pai_id in all_pai_ids:
        req = requirements.get(pai_id, {"presentations": 0, "subjects": 0})
        categories.append(CategoryProgress(
            pai_id=pai_id,
            label=_PAI_LABELS.get(pai_id, pai_id),
            collected_presentations=pres_counts.get(pai_id, 0),
            required_presentations=req["presentations"],
            collected_subjects=subj_counts.get(pai_id, 0),
            required_subjects=req["subjects"],
        ))

    # Compute overall readiness: weighted by required presentation counts,
    # only over the required PAI IDs (not optional PA-04)
    required_cats = [c for c in categories if c.pai_id in REQUIRED_PAI_IDS]
    total_required = sum(c.required_presentations for c in required_cats)
    total_collected = sum(
        min(c.collected_presentations, c.required_presentations)
        for c in required_cats
    )
    overall_pct = (total_collected / total_required * 100.0) if total_required > 0 else 0.0

    ready = (
        all(c.is_complete for c in required_cats)
        and qc_decision in ("PASS", "WARNING")
        and qc_error_count == 0
    )

    return DatasetStatus(
        collection_id=cid,
        collection_root=str(collection_root.resolve()),
        generated_at_utc=now,
        protocol_id=protocol_id,
        categories=categories,
        qc_decision=qc_decision,
        qc_error_count=qc_error_count,
        qc_warning_count=qc_warning_count,
        overall_readiness_pct=round(overall_pct, 1),
        ready_for_baseline=ready,
        manifest_found=manifest_found,
        manifest_version=manifest_version,
        total_presentations_in_manifest=total_in_manifest,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_status(status: DatasetStatus) -> None:
    """Print a human-readable dashboard to stdout."""
    bar_width = 20

    def _bar(pct: float) -> str:
        filled = int(pct / 100 * bar_width)
        return "[" + "#" * filled + "-" * (bar_width - filled) + "]"

    print()
    print("=" * 56)
    print(f"  Face Lock AI — Dataset Collection Status")
    print("=" * 56)
    print(f"  Collection : {status.collection_id}")
    print(f"  Protocol   : {status.protocol_id}")
    print(f"  Generated  : {status.generated_at_utc}")
    print()

    overall_bar = _bar(status.overall_readiness_pct)
    print(f"  Dataset readiness: {status.overall_readiness_pct:.1f}%  {overall_bar}")
    print()

    # Category table
    header = f"  {'PAI':<10} {'Label':<16} {'Pres':>6} {'/ Req':>6}  {'Subj':>5} {'/ Req':>6}  {'Remain':>7}  {'%':>6}"
    print(header)
    print("  " + "-" * 74)
    for cat in status.categories:
        remain = cat.presentations_remaining
        pct = cat.presentation_pct
        tick = "✓" if cat.is_complete else " "
        print(
            f"  {tick} {cat.pai_id:<9} {cat.label:<16} "
            f"{cat.collected_presentations:>6} {cat.required_presentations:>6}  "
            f"{cat.collected_subjects:>5} {cat.required_subjects:>6}  "
            f"{remain:>7}  {pct:>5.1f}%"
        )
    print()

    # QC status
    qc_icon = {"PASS": "✓", "WARNING": "⚠", "FAIL": "✗", "NOT_RUN": "?", "UNKNOWN": "?"}.get(
        status.qc_decision, "?"
    )
    print(f"  QC status   : {qc_icon} {status.qc_decision}  "
          f"(errors: {status.qc_error_count}, warnings: {status.qc_warning_count})")

    manifest_str = (
        f"Found (v{status.manifest_version}, {status.total_presentations_in_manifest} samples)"
        if status.manifest_found
        else "Not found — run generate_manifest.py"
    )
    print(f"  Manifest    : {manifest_str}")
    print()

    ready_str = "YES — ready for baseline evaluation" if status.ready_for_baseline else "NO  — collection incomplete or QC not passing"
    print(f"  Baseline ready: {ready_str}")
    print("=" * 56)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    project_root = resolve_project_root()
    default_root = project_root / "eval" / "datasets" / DEFAULT_COLLECTION_ID
    default_protocol = project_root / DEFAULT_PROTOCOL

    parser = argparse.ArgumentParser(
        description="Show dataset collection progress toward PAD baseline approval.",
    )
    parser.add_argument(
        "--collection-root",
        type=Path,
        default=default_root,
        help=f"Dataset root directory (default: {default_root})",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=default_protocol,
        help=f"PAD protocol YAML (default: {DEFAULT_PROTOCOL})",
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default=None,
        help="Override collection ID (default: directory name).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write dataset_status.json to this path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (useful with --output).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit 0=ready, 1=not ready."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.collection_root.is_dir():
        print(f"[ERROR] Collection root not found: {args.collection_root}", file=sys.stderr)
        return 2

    if not args.protocol.is_file():
        print(f"[ERROR] Protocol file not found: {args.protocol}", file=sys.stderr)
        return 2

    status = compute_status(
        collection_root=args.collection_root,
        protocol_path=args.protocol,
        collection_id=args.collection_id,
    )

    if not args.quiet:
        print_status(status)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(status.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not args.quiet:
            print(f"[OK] Wrote {args.output}")

    return 0 if status.ready_for_baseline else 1


if __name__ == "__main__":
    raise SystemExit(main())
