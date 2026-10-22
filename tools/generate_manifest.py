#!/usr/bin/env python3
"""
Generate manifest.json for a Face Lock PAD evaluation collection.

Scans eval/datasets/<collection_id>/ per FACE_LOCK_V1_COLLECTION_GUIDE.md layout.
Does not run PAD, evaluation harnesses, or modify application code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_bootstrap_spec = importlib.util.spec_from_file_location(
    "face_lock_script_bootstrap",
    Path(__file__).resolve().parent / "_script_bootstrap.py",
)
if _bootstrap_spec and _bootstrap_spec.loader:
    _bootstrap_mod = importlib.util.module_from_spec(_bootstrap_spec)
    _bootstrap_spec.loader.exec_module(_bootstrap_mod)
    _bootstrap_mod.ensure_repository_root_on_sys_path(__file__)

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final

from tools.project_paths import resolve_project_root

from tools.collect_pad_dataset import (
    FRAME_FILENAME_PATTERN,
    MIN_FRAMES_PROTOCOL,
    PRESENTATION_SPECS,
    SUBJECT_ID_PATTERN,
)

LOGGER = logging.getLogger("face_lock.generate_manifest")

MANIFEST_VERSION: Final[str] = "1.0.0"
DEFAULT_COLLECTION_ID: Final[str] = "face_lock_v1"
DEFAULT_PROTOCOL_IDS: Final[list[str]] = ["PAD-ISO-LAB-v1"]

SUBJECT_FOLDER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^subject_(\d{3})$")
SESSION_FOLDER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^session_(\d{2})$")
ATTEMPT_FOLDER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^attempt_(\d{2})$")


class ManifestGenerationError(ValueError):
    """Raised when the dataset cannot produce a valid manifest."""


@dataclass(frozen=True, slots=True)
class ManifestSample:
    """One manifest sample row."""

    sample_id: str
    pai_id: str
    ground_truth: str
    subject_id: str
    path: str
    presentation_type: str
    frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "pai_id": self.pai_id,
            "ground_truth": self.ground_truth,
            "subject_id": self.subject_id,
            "path": self.path,
            "presentation_type": self.presentation_type,
            "frame_count": self.frame_count,
        }


@dataclass
class ScanWarning:
    """Non-fatal issue discovered while scanning."""

    level: str
    path: str
    message: str

    def format(self) -> str:
        return f"[{self.level}] {self.path}: {self.message}"


@dataclass
class CollectionStatistics:
    """Presentation counts per PAI category."""

    bf: int = 0
    pa01: int = 0
    pa02: int = 0
    pa03: int = 0
    pa04: int = 0
    total: int = 0
    subjects_bf: set[str] = field(default_factory=set)
    subjects_pa01: set[str] = field(default_factory=set)
    subjects_pa02: set[str] = field(default_factory=set)
    subjects_pa03: set[str] = field(default_factory=set)
    subjects_pa04: set[str] = field(default_factory=set)

    def record(self, pai_id: str, subject_id: str) -> None:
        self.total += 1
        if pai_id == "BF":
            self.bf += 1
            self.subjects_bf.add(subject_id)
        elif pai_id == "PA-01":
            self.pa01 += 1
            self.subjects_pa01.add(subject_id)
        elif pai_id == "PA-02":
            self.pa02 += 1
            self.subjects_pa02.add(subject_id)
        elif pai_id == "PA-03":
            self.pa03 += 1
            self.subjects_pa03.add(subject_id)
        elif pai_id == "PA-04":
            self.pa04 += 1
            self.subjects_pa04.add(subject_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_presentations": self.total,
            "BF": self.bf,
            "PA-01": self.pa01,
            "PA-02": self.pa02,
            "PA-03": self.pa03,
            "PA-04": self.pa04,
            "unique_subjects": {
                "BF": len(self.subjects_bf),
                "PA-01": len(self.subjects_pa01),
                "PA-02": len(self.subjects_pa02),
                "PA-03": len(self.subjects_pa03),
                "PA-04": len(self.subjects_pa04),
            },
        }


@dataclass
class ScanResult:
    """Output of scanning a collection directory."""

    samples: list[ManifestSample]
    warnings: list[ScanWarning]
    statistics: CollectionStatistics


def configure_logging(verbose: bool = False) -> None:
    """Configure module logging for CLI use."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_collection_root(collection_root: Path) -> Path:
    """Resolve collection root and ensure it exists."""
    root = collection_root.expanduser().resolve()
    if not root.is_dir():
        raise ManifestGenerationError(f"Collection root not found: {root}")
    return root


def folder_name_to_subject_id(folder_name: str) -> str | None:
    """Convert subject_007 → subj_007 if valid."""
    match = SUBJECT_FOLDER_PATTERN.match(folder_name)
    if not match:
        return None
    subject_id = f"subj_{match.group(1)}"
    if SUBJECT_ID_PATTERN.match(subject_id):
        return subject_id
    return None


def count_frames(presentation_dir: Path) -> tuple[int, list[int]]:
    """
    Count sequential frame_*.jpg files in a presentation directory.

    Returns (count, list of missing indices in 1..max if gaps exist).
    """
    indices: list[int] = []
    for path in presentation_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_FILENAME_PATTERN.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        return 0, []
    indices.sort()
    expected = list(range(1, indices[-1] + 1))
    missing = [index for index in expected if index not in indices]
    return len(indices), missing


def ground_truth_for_pai(pai_id: str) -> str:
    """Return manifest ground_truth for a PAI id."""
    return "bona_fide" if pai_id == "BF" else "attack"


def make_sample_id(pai_id: str, subject_id: str, sequence_name: str) -> str:
    """
    Build a unique sample_id per collection guide conventions.

    Examples: BF_subj001_s03, PA01_subj007_a02
    """
    subject_compact = subject_id.replace("_", "")
    if pai_id == "BF":
        prefix = "BF"
    else:
        prefix = pai_id.replace("-", "")
    if "_" not in sequence_name:
        raise ManifestGenerationError(f"Invalid sequence folder name: {sequence_name}")
    seq_num = sequence_name.split("_", 1)[1]
    if pai_id == "BF":
        suffix = f"s{seq_num}"
    else:
        suffix = f"a{seq_num}"
    return f"{prefix}_{subject_compact}_{suffix}"


def ensure_unique_sample_id(
    base_id: str,
    used_ids: set[str],
) -> str:
    """Ensure sample_id is unique within the manifest."""
    if base_id not in used_ids:
        used_ids.add(base_id)
        return base_id
    counter = 2
    while True:
        candidate = f"{base_id}_{counter}"
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
        counter += 1


def validate_sequence_folder(
    folder_name: str,
    expected_prefix: str,
) -> bool:
    """Return True if folder matches session_NN or attempt_NN."""
    if expected_prefix == "session":
        return bool(SESSION_FOLDER_PATTERN.match(folder_name))
    if expected_prefix == "attempt":
        return bool(ATTEMPT_FOLDER_PATTERN.match(folder_name))
    return False


def scan_presentation_dir(
    *,
    collection_root: Path,
    pai_id: str,
    presentation_type_code: str,
    relative_root: str,
    sequence_prefix: str,
    presentation_dir: Path,
    sequence_name: str,
    subject_id: str,
    min_frames_warn: int,
    used_sample_ids: set[str],
    warnings: list[ScanWarning],
) -> ManifestSample | None:
    """Validate one presentation folder and return a sample row."""
    rel_path = presentation_dir.relative_to(collection_root).as_posix()

    if not presentation_dir.is_dir():
        warnings.append(
            ScanWarning("error", rel_path, "Presentation path is not a directory")
        )
        return None

    frame_count, missing_indices = count_frames(presentation_dir)
    if frame_count == 0:
        warnings.append(
            ScanWarning("error", rel_path, "No frame_*.jpg files found")
        )
        return None

    if missing_indices:
        warnings.append(
            ScanWarning(
                "warning",
                rel_path,
                f"Gaps in frame sequence (missing indices: {missing_indices[:5]})",
            )
        )

    if frame_count < MIN_FRAMES_PROTOCOL:
        warnings.append(
            ScanWarning(
                "warning",
                rel_path,
                f"Frame count {frame_count} < protocol minimum {MIN_FRAMES_PROTOCOL}",
            )
        )
    elif frame_count < min_frames_warn:
        warnings.append(
            ScanWarning(
                "warning",
                rel_path,
                f"Frame count {frame_count} < recommended minimum {min_frames_warn}",
            )
        )

    base_sample_id = make_sample_id(pai_id, subject_id, sequence_name)
    sample_id = ensure_unique_sample_id(base_sample_id, used_sample_ids)

    return ManifestSample(
        sample_id=sample_id,
        pai_id=pai_id,
        ground_truth=ground_truth_for_pai(pai_id),
        subject_id=subject_id,
        path=rel_path,
        presentation_type="clip",
        frame_count=frame_count,
    )


def scan_collection(
    collection_root: Path,
    *,
    min_frames_warn: int = 12,
) -> ScanResult:
    """
    Scan collection_root and discover all valid presentations.

    Walks bona_fide/ and attacks/PA-* trees produced by collect_pad_dataset.py.
    """
    root = resolve_collection_root(collection_root)
    samples: list[ManifestSample] = []
    warnings: list[ScanWarning] = []
    statistics = CollectionStatistics()
    used_sample_ids: set[str] = set()

    for type_code, spec in sorted(
        PRESENTATION_SPECS.items(),
        key=lambda item: item[0],
    ):
        category_root = root / spec.relative_root
        if not category_root.is_dir():
            warnings.append(
                ScanWarning(
                    "warning",
                    spec.relative_root,
                    f"Category directory missing for {type_code}",
                )
            )
            continue

        for subject_dir in sorted(category_root.iterdir()):
            if not subject_dir.is_dir():
                warnings.append(
                    ScanWarning(
                        "warning",
                        str(subject_dir.relative_to(root)),
                        "Unexpected non-directory entry under category",
                    )
                )
                continue

            subject_id = folder_name_to_subject_id(subject_dir.name)
            if subject_id is None:
                warnings.append(
                    ScanWarning(
                        "error",
                        str(subject_dir.relative_to(root)),
                        f"Invalid subject folder name (expected subject_NNN): {subject_dir.name}",
                    )
                )
                continue

            for seq_dir in sorted(subject_dir.iterdir()):
                rel_seq = seq_dir.relative_to(root)
                if not seq_dir.is_dir():
                    warnings.append(
                        ScanWarning(
                            "warning",
                            str(rel_seq),
                            "Unexpected file under subject directory",
                        )
                    )
                    continue

                if not validate_sequence_folder(seq_dir.name, spec.sequence_prefix):
                    warnings.append(
                        ScanWarning(
                            "error",
                            str(rel_seq),
                            f"Invalid folder name (expected {spec.sequence_prefix}_NN)",
                        )
                    )
                    continue

                sample = scan_presentation_dir(
                    collection_root=root,
                    pai_id=spec.pai_id,
                    presentation_type_code=type_code,
                    relative_root=spec.relative_root,
                    sequence_prefix=spec.sequence_prefix,
                    presentation_dir=seq_dir,
                    sequence_name=seq_dir.name,
                    subject_id=subject_id,
                    min_frames_warn=min_frames_warn,
                    used_sample_ids=used_sample_ids,
                    warnings=warnings,
                )
                if sample is not None:
                    samples.append(sample)
                    statistics.record(spec.pai_id, subject_id)

    samples.sort(key=lambda row: (row.pai_id, row.subject_id, row.path))
    return ScanResult(samples=samples, warnings=warnings, statistics=statistics)


def build_manifest_document(
    *,
    collection_id: str,
    samples: list[ManifestSample],
    description: str | None = None,
    protocol_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble top-level manifest.json content."""
    if not samples:
        raise ManifestGenerationError("No valid samples found; manifest not written.")

    return {
        "manifest_version": MANIFEST_VERSION,
        "collection_id": collection_id,
        "created_at": date.today().isoformat(),
        "description": description
        or f"Auto-generated manifest for {collection_id}",
        "protocol_ids": protocol_ids or list(DEFAULT_PROTOCOL_IDS),
        "samples": [sample.to_dict() for sample in samples],
    }


def write_manifest(path: Path, document: dict[str, Any]) -> Path:
    """Write manifest JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def print_statistics(statistics: CollectionStatistics) -> None:
    """Log collection statistics summary."""
    stats = statistics.as_dict()
    LOGGER.info("Collection statistics:")
    LOGGER.info("  Total presentations: %s", stats["total_presentations"])
    LOGGER.info("  BF:  %s (%s subjects)", statistics.bf, len(statistics.subjects_bf))
    LOGGER.info("  PA-01: %s (%s subjects)", statistics.pa01, len(statistics.subjects_pa01))
    LOGGER.info("  PA-02: %s (%s subjects)", statistics.pa02, len(statistics.subjects_pa02))
    LOGGER.info("  PA-03: %s (%s subjects)", statistics.pa03, len(statistics.subjects_pa03))
    LOGGER.info("  PA-04: %s (%s subjects)", statistics.pa04, len(statistics.subjects_pa04))


def print_warnings(warnings: list[ScanWarning]) -> None:
    """Log all scan warnings."""
    for warning in warnings:
        log_fn = LOGGER.error if warning.level == "error" else LOGGER.warning
        log_fn(warning.format())


def count_errors(warnings: list[ScanWarning]) -> int:
    """Return number of error-level warnings."""
    return sum(1 for warning in warnings if warning.level == "error")


def generate_manifest(
    collection_root: Path,
    output_path: Path | None = None,
    *,
    collection_id: str | None = None,
    description: str | None = None,
    min_frames_warn: int = 12,
    strict: bool = False,
    dry_run: bool = False,
) -> tuple[dict[str, Any], ScanResult]:
    """
    Scan dataset and write manifest.json.

    Returns (manifest document, scan result).
    """
    root = resolve_collection_root(collection_root)
    collection_id = collection_id or root.name or DEFAULT_COLLECTION_ID
    output_path = output_path or (root / "manifest.json")

    result = scan_collection(root, min_frames_warn=min_frames_warn)
    print_warnings(result.warnings)
    print_statistics(result.statistics)

    errors = count_errors(result.warnings)
    if strict and errors:
        raise ManifestGenerationError(
            f"Strict mode: {errors} error(s) found; manifest not written."
        )

    document = build_manifest_document(
        collection_id=collection_id,
        samples=result.samples,
        description=description,
    )

    if not dry_run:
        write_manifest(output_path, document)
        LOGGER.info("Wrote manifest: %s (%s samples)", output_path, len(result.samples))
    else:
        LOGGER.info("Dry run: would write %s (%s samples)", output_path, len(result.samples))

    return document, result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    project_root = resolve_project_root()
    default_root = project_root / "eval" / "datasets" / DEFAULT_COLLECTION_ID

    parser = argparse.ArgumentParser(
        description="Generate PAD evaluation manifest.json from a collected dataset.",
    )
    parser.add_argument(
        "--collection-root",
        type=Path,
        default=default_root,
        help=f"Dataset root to scan (default: {default_root})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output manifest path (default: <collection-root>/manifest.json).",
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default=None,
        help="collection_id field (default: directory name).",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional manifest description string.",
    )
    parser.add_argument(
        "--min-frames-warn",
        type=int,
        default=12,
        help="Warn when frame count is below this value (default: 12).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any error-level warnings occur.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without writing manifest.json.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        generate_manifest(
            args.collection_root,
            args.output,
            collection_id=args.collection_id,
            description=args.description,
            min_frames_warn=args.min_frames_warn,
            strict=args.strict,
            dry_run=args.dry_run,
        )
    except ManifestGenerationError as error:
        LOGGER.error("%s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
