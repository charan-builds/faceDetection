#!/usr/bin/env python3
"""
Dataset quality-control validation for Face Lock PAD evaluation collections.

Inspects eval/datasets/<collection_id>/ per FACE_LOCK_V1_COLLECTION_GUIDE.md.
Read-only: does not modify data, run PAD, evaluation, or reporting.

Writes dataset_qc_report.json and prints a console summary with PASS / WARNING / FAIL.
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
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from tools.project_paths import resolve_project_root

from tools.collect_pad_dataset import (
    FRAME_FILENAME_PATTERN,
    MIN_FRAMES_PROTOCOL,
    PRESENTATION_SPECS,
    SUBJECT_ID_PATTERN,
)
from tools.generate_manifest import (
    ATTEMPT_FOLDER_PATTERN,
    SESSION_FOLDER_PATTERN,
    SUBJECT_FOLDER_PATTERN,
    count_frames,
    folder_name_to_subject_id,
    ground_truth_for_pai,
)

LOGGER = logging.getLogger("face_lock.qc_dataset")

REPORT_VERSION: Final[str] = "1.0.0"
DEFAULT_COLLECTION_ID: Final[str] = "face_lock_v1"
RECOMMENDED_MIN_FRAMES: Final[int] = 12

VALID_PAI_IDS: Final[frozenset[str]] = frozenset(
    {"BF", "PA-01", "PA-02", "PA-03", "PA-04", "PA-05"}
)
VALID_GROUND_TRUTH: Final[frozenset[str]] = frozenset({"bona_fide", "attack"})

Decision = Literal["PASS", "WARNING", "FAIL"]
Severity = Literal["error", "warning"]

# PAD-ISO-LAB-v1 standard tier minimums (FACE_LOCK_V1_COLLECTION_GUIDE.md)
STANDARD_MINIMUMS: Final[dict[str, dict[str, int]]] = {
    "BF": {"presentations": 150, "subjects": 30},
    "PA-01": {"presentations": 50, "subjects": 10},
    "PA-02": {"presentations": 50, "subjects": 10},
    "PA-03": {"presentations": 50, "subjects": 10},
    "PA-04": {"presentations": 30, "subjects": 6},
}

LOW_BRIGHTNESS_THRESHOLD: Final[float] = 45.0
LOW_BLUR_VARIANCE_THRESHOLD: Final[float] = 50.0


class QCValidationError(ValueError):
    """Raised when QC cannot run (e.g. missing collection root)."""


@dataclass(frozen=True, slots=True)
class QCIssue:
    """Single QC finding."""

    code: str
    severity: Severity
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class FrameQualityMetrics:
    """Per-presentation image quality summary."""

    path: str
    frame_count: int
    mean_brightness: float
    mean_blur_variance: float
    width: int
    height: int
    inconsistent_dimensions: bool
    detected_faces: int
    file_hash: str


@dataclass
class CategoryCounts:
    """Presentation and subject counts per PAI."""

    presentations: dict[str, int] = field(default_factory=dict)
    subjects: dict[str, int] = field(default_factory=dict)

    def record(self, pai_id: str, subject_id: str) -> None:
        self.presentations[pai_id] = self.presentations.get(pai_id, 0) + 1

    def finalize_subjects(self, subject_sets: dict[str, set[str]]) -> None:
        for pai_id, subjects in subject_sets.items():
            self.subjects[pai_id] = len(subjects)


@dataclass
class QCReport:
    """Full QC report payload."""

    report_version: str
    collection_root: str
    collection_id: str
    generated_at_utc: str
    decision: Decision
    errors: list[QCIssue]
    warnings: list[QCIssue]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "collection_root": self.collection_root,
            "collection_id": self.collection_id,
            "generated_at_utc": self.generated_at_utc,
            "decision": self.decision,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class DiscoveredPresentation:
    """One presentation folder on disk."""

    pai_id: str
    subject_id: str
    relative_path: str
    presentation_dir: Path
    sequence_name: str


def configure_logging(verbose: bool = False) -> None:
    """Configure module logging for CLI use."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_collection_root(collection_root: Path) -> Path:
    """Resolve collection root path."""
    root = collection_root.expanduser().resolve()
    if not root.is_dir():
        raise QCValidationError(f"Collection root not found: {root}")
    return root


def add_issue(
    issues: list[QCIssue],
    *,
    code: str,
    severity: Severity,
    path: str,
    message: str,
) -> None:
    """Append a QC issue."""
    issues.append(QCIssue(code=code, severity=severity, path=path, message=message))


def validate_sequence_folder_name(
    folder_name: str,
    expected_prefix: str,
) -> bool:
    """Return True if folder matches session_NN or attempt_NN."""
    if expected_prefix == "session":
        return bool(SESSION_FOLDER_PATTERN.match(folder_name))
    return bool(ATTEMPT_FOLDER_PATTERN.match(folder_name))


def discover_presentations(
    collection_root: Path,
    errors: list[QCIssue],
    warnings: list[QCIssue],
) -> tuple[list[DiscoveredPresentation], dict[str, set[str]]]:
    """Walk dataset tree and return valid presentations plus subject sets per PAI."""
    presentations: list[DiscoveredPresentation] = []
    subject_sets: dict[str, set[str]] = {spec.pai_id: set() for spec in PRESENTATION_SPECS.values()}

    for _type_code, spec in sorted(PRESENTATION_SPECS.items()):
        category_root = collection_root / spec.relative_root
        if not category_root.is_dir():
            add_issue(
                warnings,
                code="missing_category_folder",
                severity="warning",
                path=spec.relative_root,
                message=f"Category directory missing for {spec.pai_id}",
            )
            continue

        for subject_dir in sorted(category_root.iterdir()):
            rel_subject = subject_dir.relative_to(collection_root).as_posix()
            if not subject_dir.is_dir():
                add_issue(
                    warnings,
                    code="invalid_folder_entry",
                    severity="warning",
                    path=rel_subject,
                    message="Non-directory entry under category",
                )
                continue

            subject_id = folder_name_to_subject_id(subject_dir.name)
            if subject_id is None:
                add_issue(
                    errors,
                    code="invalid_subject_folder",
                    severity="error",
                    path=rel_subject,
                    message=f"Invalid subject folder name: {subject_dir.name}",
                )
                continue

            for seq_dir in sorted(subject_dir.iterdir()):
                rel_seq = seq_dir.relative_to(collection_root).as_posix()
                if not seq_dir.is_dir():
                    add_issue(
                        warnings,
                        code="invalid_folder_entry",
                        severity="warning",
                        path=rel_seq,
                        message="Non-directory entry under subject folder",
                    )
                    continue

                if not validate_sequence_folder_name(seq_dir.name, spec.sequence_prefix):
                    add_issue(
                        errors,
                        code="invalid_sequence_folder",
                        severity="error",
                        path=rel_seq,
                        message=(
                            f"Invalid sequence folder (expected "
                            f"{spec.sequence_prefix}_NN): {seq_dir.name}"
                        ),
                    )
                    continue

                presentations.append(
                    DiscoveredPresentation(
                        pai_id=spec.pai_id,
                        subject_id=subject_id,
                        relative_path=rel_seq,
                        presentation_dir=seq_dir,
                        sequence_name=seq_dir.name,
                    )
                )
                subject_sets[spec.pai_id].add(subject_id)

    return presentations, subject_sets


def list_frame_paths(presentation_dir: Path) -> list[Path]:
    """List frame_*.jpg paths sorted by index."""
    frames: list[tuple[int, Path]] = []
    for path in presentation_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_FILENAME_PATTERN.match(path.name)
        if match:
            frames.append((int(match.group(1)), path))
    frames.sort(key=lambda item: item[0])
    return [path for _, path in frames]


def check_presentation_frames(
    presentation: DiscoveredPresentation,
    errors: list[QCIssue],
    warnings: list[QCIssue],
) -> list[Path]:
    """Validate frame count, naming, and missing files."""
    rel = presentation.relative_path
    frame_count, missing_indices = count_frames(presentation.presentation_dir)
    frame_paths = list_frame_paths(presentation.presentation_dir)

    if frame_count == 0:
        add_issue(
            errors,
            code="missing_frames",
            severity="error",
            path=rel,
            message="No frame_*.jpg files found",
        )
        return []

    if missing_indices:
        add_issue(
            errors,
            code="non_sequential_frames",
            severity="error",
            path=rel,
            message=f"Non-sequential frame numbering; missing indices: {missing_indices[:10]}",
        )

    extra_files = [
        path.name
        for path in presentation.presentation_dir.iterdir()
        if path.is_file() and not FRAME_FILENAME_PATTERN.match(path.name)
        and path.name.lower() != "meta.json"
    ]
    if extra_files:
        add_issue(
            warnings,
            code="unexpected_files",
            severity="warning",
            path=rel,
            message=f"Unexpected files in presentation folder: {extra_files[:5]}",
        )

    if frame_count < MIN_FRAMES_PROTOCOL:
        add_issue(
            errors,
            code="frame_count_below_protocol",
            severity="error",
            path=rel,
            message=(
                f"Frame count {frame_count} < protocol minimum "
                f"{MIN_FRAMES_PROTOCOL}"
            ),
        )
    elif frame_count < RECOMMENDED_MIN_FRAMES:
        add_issue(
            warnings,
            code="frame_count_below_recommended",
            severity="warning",
            path=rel,
            message=(
                f"Frame count {frame_count} < recommended minimum "
                f"{RECOMMENDED_MIN_FRAMES}"
            ),
        )

    return frame_paths


def analyze_frame_quality(
    frame_paths: list[Path],
    presentation_path: str,
    errors: list[QCIssue],
    warnings: list[QCIssue],
) -> FrameQualityMetrics | None:
    """
    Compute brightness, blur, and dimension metrics for presentation frames.

    Uses OpenCV when available. Marks corrupt/unreadable images as errors.
    """
    if not frame_paths:
        return None

    try:
        import cv2
        import numpy as np
        from deepface import DeepFace
    except ImportError:
        add_issue(
            warnings,
            code="quality_check_skipped",
            severity="warning",
            path=presentation_path,
            message="OpenCV or DeepFace not installed; skipping brightness/blur/face checks",
        )
        return None

    brightness_values: list[float] = []
    blur_values: list[float] = []
    dimensions: set[tuple[int, int]] = set()
    face_counts: list[int] = []
    file_hashes: list[str] = []

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        if image is None:
            add_issue(
                errors,
                code="corrupt_jpeg",
                severity="error",
                path=f"{presentation_path}/{frame_path.name}",
                message="Unreadable or corrupt JPEG",
            )
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(gray)))
        blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        dimensions.add((image.shape[1], image.shape[0]))
        
        # Hash calculation for the first frame as representative for leakage check
        if not file_hashes:
            with open(frame_path, "rb") as f:
                file_hashes.append(hashlib.sha256(f.read()).hexdigest())

        # Face detection count
        try:
            faces = DeepFace.extract_faces(
                img_path=str(frame_path),
                enforce_detection=False,
                detector_backend="opencv",
            )
            # DeepFace returns a list of faces. If it's a dummy return because no face found, confidence is 0.
            # But with enforce_detection=False, if no face is found, it returns the whole image.
            # Let's count faces that have confidence > 0.
            valid_faces = [f for f in faces if f.get("confidence", 0) > 0]
            face_counts.append(len(valid_faces))
        except ValueError:
            face_counts.append(0)

    if not brightness_values:
        return None

    mean_brightness = sum(brightness_values) / len(brightness_values)
    mean_blur = sum(blur_values) / len(blur_values) if blur_values else 0.0
    width, height = next(iter(dimensions))
    inconsistent = len(dimensions) > 1

    if mean_brightness < LOW_BRIGHTNESS_THRESHOLD:
        add_issue(
            warnings,
            code="low_brightness",
            severity="warning",
            path=presentation_path,
            message=f"Mean brightness {mean_brightness:.1f} < {LOW_BRIGHTNESS_THRESHOLD}",
        )

    if mean_blur < LOW_BLUR_VARIANCE_THRESHOLD:
        add_issue(
            warnings,
            code="excessive_blur",
            severity="warning",
            path=presentation_path,
            message=f"Mean blur variance {mean_blur:.1f} < {LOW_BLUR_VARIANCE_THRESHOLD}",
        )

    if inconsistent:
        add_issue(
            warnings,
            code="inconsistent_image_size",
            severity="warning",
            path=presentation_path,
            message=f"Inconsistent frame dimensions: {sorted(dimensions)}",
        )

    mean_faces = sum(face_counts) / len(face_counts) if face_counts else 0
    if mean_faces == 0:
        add_issue(
            errors,
            code="no_face_detected",
            severity="error",
            path=presentation_path,
            message="No face detected in any frame.",
        )
    elif mean_faces > 1.0:
        add_issue(
            errors,
            code="multiple_faces_detected",
            severity="error",
            path=presentation_path,
            message=f"Average of {mean_faces:.1f} faces detected; expected exactly 1.",
        )

    return FrameQualityMetrics(
        path=presentation_path,
        frame_count=len(brightness_values),
        mean_brightness=round(mean_brightness, 2),
        mean_blur_variance=round(mean_blur, 2),
        width=width,
        height=height,
        inconsistent_dimensions=inconsistent,
        detected_faces=round(mean_faces),
        file_hash=file_hashes[0] if file_hashes else "",
    )


def check_collection_minimums(
    category_counts: CategoryCounts,
    subject_sets: dict[str, set[str]],
    warnings: list[QCIssue],
) -> None:
    """Warn when presentation/subject counts are below PAD-ISO-LAB standard minimums."""
    category_counts.finalize_subjects(subject_sets)

    for pai_id, minimums in STANDARD_MINIMUMS.items():
        pres_count = category_counts.presentations.get(pai_id, 0)
        subj_count = category_counts.subjects.get(pai_id, 0)
        if pres_count < minimums["presentations"]:
            add_issue(
                warnings,
                code="below_minimum_presentations",
                severity="warning",
                path=f"collection/{pai_id}",
                message=(
                    f"{pai_id} presentations {pres_count} < standard minimum "
                    f"{minimums['presentations']}"
                ),
            )
        if subj_count < minimums["subjects"]:
            add_issue(
                warnings,
                code="below_minimum_subjects",
                severity="warning",
                path=f"collection/{pai_id}",
                message=(
                    f"{pai_id} subjects {subj_count} < standard minimum "
                    f"{minimums['subjects']}"
                ),
            )


def load_manifest(collection_root: Path) -> dict[str, Any] | None:
    """Load manifest.json if present."""
    manifest_path = collection_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise QCValidationError(f"Invalid manifest JSON: {manifest_path}")
    return data


def check_manifest_consistency(
    collection_root: Path,
    manifest: dict[str, Any] | None,
    discovered_paths: set[str],
    errors: list[QCIssue],
    warnings: list[QCIssue],
) -> None:
    """Validate manifest.json against disk and schema rules."""
    if manifest is None:
        add_issue(
            warnings,
            code="manifest_missing",
            severity="warning",
            path="manifest.json",
            message="manifest.json not found; manifest consistency checks skipped",
        )
        return

    samples = manifest.get("samples")
    if not isinstance(samples, list):
        add_issue(
            errors,
            code="manifest_invalid",
            severity="error",
            path="manifest.json",
            message="manifest samples must be a list",
        )
        return

    seen_sample_ids: set[str] = set()
    seen_paths: set[str] = set()

    for index, sample in enumerate(samples):
        path_label = f"manifest.json:samples[{index}]"
        if not isinstance(sample, dict):
            add_issue(
                errors,
                code="manifest_invalid_sample",
                severity="error",
                path=path_label,
                message="Sample entry must be an object",
            )
            continue

        sample_id = str(sample.get("sample_id", ""))
        rel_path = str(sample.get("path", ""))
        pai_id = str(sample.get("pai_id", ""))
        ground_truth = str(sample.get("ground_truth", ""))
        subject_id = str(sample.get("subject_id", ""))

        if not sample_id:
            add_issue(
                errors,
                code="manifest_missing_field",
                severity="error",
                path=path_label,
                message="Missing sample_id",
            )
        elif sample_id in seen_sample_ids:
            add_issue(
                errors,
                code="duplicate_sample_id",
                severity="error",
                path=path_label,
                message=f"Duplicate sample_id: {sample_id}",
            )
        else:
            seen_sample_ids.add(sample_id)

        if not rel_path:
            add_issue(
                errors,
                code="manifest_missing_field",
                severity="error",
                path=path_label,
                message="Missing path",
            )
        elif rel_path in seen_paths:
            add_issue(
                errors,
                code="duplicate_path",
                severity="error",
                path=path_label,
                message=f"Duplicate path: {rel_path}",
            )
        else:
            seen_paths.add(rel_path)

        full_path = collection_root / rel_path
        if rel_path and not full_path.is_dir():
            add_issue(
                errors,
                code="manifest_path_missing",
                severity="error",
                path=rel_path,
                message="Manifest path does not exist on disk",
            )

        if pai_id not in VALID_PAI_IDS:
            add_issue(
                errors,
                code="invalid_pai_id",
                severity="error",
                path=path_label,
                message=f"Invalid pai_id: {pai_id!r}",
            )

        if ground_truth not in VALID_GROUND_TRUTH:
            add_issue(
                errors,
                code="invalid_ground_truth",
                severity="error",
                path=path_label,
                message=f"Invalid ground_truth: {ground_truth!r}",
            )
        elif pai_id in VALID_PAI_IDS and ground_truth != ground_truth_for_pai(pai_id):
            add_issue(
                errors,
                code="ground_truth_pai_mismatch",
                severity="error",
                path=path_label,
                message=f"ground_truth {ground_truth!r} inconsistent with pai_id {pai_id}",
            )

        if subject_id and not SUBJECT_ID_PATTERN.match(subject_id):
            add_issue(
                errors,
                code="invalid_subject_id",
                severity="error",
                path=path_label,
                message=f"Invalid subject_id format: {subject_id!r}",
            )

    manifest_paths = {str(s.get("path", "")) for s in samples if isinstance(s, dict)}
    orphan_disk = discovered_paths - manifest_paths
    for rel in sorted(orphan_disk):
        add_issue(
            warnings,
            code="presentation_not_in_manifest",
            severity="warning",
            path=rel,
            message="Presentation on disk but not listed in manifest.json",
        )


def compute_decision(errors: list[QCIssue], warnings: list[QCIssue]) -> Decision:
    """Derive PASS / WARNING / FAIL from issue lists."""
    if errors:
        return "FAIL"
    if warnings:
        return "WARNING"
    return "PASS"


def run_qc(
    collection_root: Path,
    *,
    collection_id: str | None = None,
    check_quality: bool = True,
) -> QCReport:
    """Run full dataset QC and return a structured report."""
    root = resolve_collection_root(collection_root)
    collection_id = collection_id or root.name or DEFAULT_COLLECTION_ID

    errors: list[QCIssue] = []
    warnings: list[QCIssue] = []

    presentations, subject_sets = discover_presentations(root, errors, warnings)

    category_counts = CategoryCounts()
    quality_metrics: list[dict[str, Any]] = []
    discovered_paths: set[str] = set()

    for presentation in presentations:
        discovered_paths.add(presentation.relative_path)
        category_counts.record(presentation.pai_id, presentation.subject_id)

        frame_paths = check_presentation_frames(presentation, errors, warnings)

        if check_quality and frame_paths:
            metrics = analyze_frame_quality(
                frame_paths,
                presentation.relative_path,
                errors,
                warnings,
            )
            if metrics is not None:
                quality_metrics.append(asdict(metrics))

    # Cross-presentation leakage check (Duplicate hashes)
    if check_quality and quality_metrics:
        hash_map: dict[str, list[str]] = {}
        for m in quality_metrics:
            h = m.get("file_hash", "")
            if h:
                hash_map.setdefault(h, []).append(m["path"])
        
        for h, paths in hash_map.items():
            if len(paths) > 1:
                # If they cross sessions or subjects, it's a severe leakage error.
                add_issue(
                    errors,
                    code="data_leakage_duplicate_hash",
                    severity="error",
                    path=paths[0],
                    message=f"Identical file hash found across multiple presentations: {paths}",
                )

    check_collection_minimums(category_counts, subject_sets, warnings)

    manifest = None
    try:
        manifest = load_manifest(root)
    except json.JSONDecodeError as error:
        add_issue(
            errors,
            code="manifest_invalid_json",
            severity="error",
            path="manifest.json",
            message=str(error),
        )

    check_manifest_consistency(root, manifest, discovered_paths, errors, warnings)

    category_counts.finalize_subjects(subject_sets)

    summary: dict[str, Any] = {
        "presentations_discovered": len(presentations),
        "category_counts": category_counts.presentations,
        "subject_counts": category_counts.subjects,
        "standard_minimums": STANDARD_MINIMUMS,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "quality_metrics_sample_count": len(quality_metrics),
    }
    if quality_metrics:
        summary["quality_metrics_aggregate"] = {
            "mean_brightness_avg": round(
                sum(row["mean_brightness"] for row in quality_metrics)
                / len(quality_metrics),
                2,
            ),
            "mean_blur_variance_avg": round(
                sum(row["mean_blur_variance"] for row in quality_metrics)
                / len(quality_metrics),
                2,
            ),
        }
        summary["quality_metrics_by_presentation"] = quality_metrics

    decision = compute_decision(errors, warnings)

    return QCReport(
        report_version=REPORT_VERSION,
        collection_root=str(root),
        collection_id=collection_id,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        decision=decision,
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


def write_qc_report(path: Path, report: QCReport) -> Path:
    """Write dataset_qc_report.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def print_console_summary(report: QCReport) -> None:
    """Print human-readable QC summary to stdout."""
    print("")
    print("=" * 60)
    print(f"Face Lock PAD Dataset QC — {report.decision}")
    print("=" * 60)
    print(f"Collection: {report.collection_id}")
    print(f"Root:       {report.collection_root}")
    print(f"Errors:     {len(report.errors)}")
    print(f"Warnings:   {len(report.warnings)}")
    print("")
    print("Category counts (presentations):")
    for pai_id in ("BF", "PA-01", "PA-02", "PA-03", "PA-04"):
        count = report.summary["category_counts"].get(pai_id, 0)
        minimum = STANDARD_MINIMUMS.get(pai_id, {}).get("presentations", "n/a")
        print(f"  {pai_id:6} {count:4}  (minimum {minimum})")
    print("")
    print("Subject counts:")
    for pai_id in ("BF", "PA-01", "PA-02", "PA-03", "PA-04"):
        count = report.summary["subject_counts"].get(pai_id, 0)
        minimum = STANDARD_MINIMUMS.get(pai_id, {}).get("subjects", "n/a")
        print(f"  {pai_id:6} {count:4}  (minimum {minimum})")
    print("")
    if report.errors:
        print("Errors (first 10):")
        for issue in report.errors[:10]:
            print(f"  - [{issue.code}] {issue.path}: {issue.message}")
    if report.warnings:
        print("Warnings (first 10):")
        for issue in report.warnings[:10]:
            print(f"  - [{issue.code}] {issue.path}: {issue.message}")
    print("=" * 60)
    print("")


def run_qc_and_write(
    collection_root: Path,
    output_path: Path | None = None,
    *,
    collection_id: str | None = None,
    check_quality: bool = True,
    dry_run: bool = False,
) -> QCReport:
    """Run QC and optionally write dataset_qc_report.json."""
    root = resolve_collection_root(collection_root)
    output_path = output_path or (root / "dataset_qc_report.json")

    report = run_qc(root, collection_id=collection_id, check_quality=check_quality)
    print_console_summary(report)

    if not dry_run:
        write_qc_report(output_path, report)
        LOGGER.info("Wrote QC report: %s", output_path)
    else:
        LOGGER.info("Dry run: QC report not written")

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    project_root = resolve_project_root()
    default_root = project_root / "eval" / "datasets" / DEFAULT_COLLECTION_ID

    parser = argparse.ArgumentParser(
        description="Quality-control validation for PAD evaluation datasets.",
    )
    parser.add_argument(
        "--collection-root",
        type=Path,
        default=default_root,
        help=f"Dataset root (default: {default_root})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="QC report path (default: <collection-root>/dataset_qc_report.json).",
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default=None,
        help="Collection id for report metadata.",
    )
    parser.add_argument(
        "--no-quality-metrics",
        action="store_true",
        help="Skip brightness/blur/dimension analysis (faster).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing dataset_qc_report.json.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit 0=PASS, 1=WARNING, 2=FAIL."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        report = run_qc_and_write(
            args.collection_root,
            args.output,
            collection_id=args.collection_id,
            check_quality=not args.no_quality_metrics,
            dry_run=args.dry_run,
        )
    except QCValidationError as error:
        LOGGER.error("%s", error)
        return 2

    if report.decision == "PASS":
        return 0
    if report.decision == "WARNING":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
