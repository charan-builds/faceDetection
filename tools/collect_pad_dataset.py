#!/usr/bin/env python3
"""
Face Lock AI — PAD evaluation dataset collection tool.

Captures webcam frame sequences into the folder layout defined in
eval/docs/FACE_LOCK_V1_COLLECTION_GUIDE.md (face_lock_v1 collection).

Does not run PAD, generate manifests, or execute evaluation harnesses.
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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final

from tools.project_paths import resolve_project_root

LOGGER = logging.getLogger("face_lock.collect_pad")

SUBJECT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^subj_[0-9]{3}$")
SUPPORTED_PRESENTATION_TYPES: Final[frozenset[str]] = frozenset(
    {"BF", "PA-01", "PA-02", "PA-03", "PA-04"}
)
FRAME_FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^frame_(\d{4})\.jpg$", re.IGNORECASE
)
WINDOW_NAME: Final[str] = "Face Lock PAD Collection"
MIN_FRAMES_PROTOCOL: Final[int] = 8
DEFAULT_CAPTURE_FPS: Final[float] = 5.0


class CollectionValidationError(ValueError):
    """Raised when user input or output paths are invalid."""


class CameraOpenError(RuntimeError):
    """Raised when the webcam cannot be opened."""


@dataclass(frozen=True, slots=True)
class PresentationTypeSpec:
    """Maps a collector-facing type code to on-disk layout."""

    code: str
    pai_id: str
    relative_root: str
    sequence_prefix: str


PRESENTATION_SPECS: Final[dict[str, PresentationTypeSpec]] = {
    "BF": PresentationTypeSpec(
        code="BF",
        pai_id="BF",
        relative_root="bona_fide",
        sequence_prefix="session",
    ),
    "PA-01": PresentationTypeSpec(
        code="PA-01",
        pai_id="PA-01",
        relative_root="attacks/PA-01_print",
        sequence_prefix="attempt",
    ),
    "PA-02": PresentationTypeSpec(
        code="PA-02",
        pai_id="PA-02",
        relative_root="attacks/PA-02_screen_static",
        sequence_prefix="attempt",
    ),
    "PA-03": PresentationTypeSpec(
        code="PA-03",
        pai_id="PA-03",
        relative_root="attacks/PA-03_screen_video",
        sequence_prefix="attempt",
    ),
    "PA-04": PresentationTypeSpec(
        code="PA-04",
        pai_id="PA-04",
        relative_root="attacks/PA-04_virtual_camera",
        sequence_prefix="attempt",
    ),
}


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """Runtime configuration for the collection session."""

    collection_root: Path
    camera_index: int
    min_frames: int
    max_frames: int
    capture_fps: float
    collector_id: str
    write_meta: bool
    jpeg_quality: int


@dataclass(frozen=True, slots=True)
class PresentationPathInfo:
    """Resolved destination for one presentation clip."""

    subject_id: str
    presentation_type: str
    presentation_dir: Path
    sequence_name: str
    relative_path: str


def configure_logging(verbose: bool = False) -> None:
    """Configure module logging for CLI use."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_subject_id(subject_id: str) -> str:
    """
    Validate manifest-style subject id (e.g. subj_001).

    Raises CollectionValidationError if invalid.
    """
    normalized = subject_id.strip().lower()
    if not SUBJECT_ID_PATTERN.match(normalized):
        raise CollectionValidationError(
            "subject_id must match subj_NNN (three digits), e.g. subj_001"
        )
    return normalized


def validate_presentation_type(presentation_type: str) -> str:
    """Validate presentation type code (BF, PA-01, …)."""
    normalized = presentation_type.strip().upper()
    if normalized not in SUPPORTED_PRESENTATION_TYPES:
        raise CollectionValidationError(
            f"presentation_type must be one of: {', '.join(sorted(SUPPORTED_PRESENTATION_TYPES))}"
        )
    return normalized


def subject_id_to_folder_name(subject_id: str) -> str:
    """Convert subj_007 → subject_007 for directory names."""
    validate_subject_id(subject_id)
    suffix = subject_id.split("_", 1)[1]
    return f"subject_{suffix}"


def resolve_collection_root(collection_root: Path) -> Path:
    """Resolve and ensure collection root exists; guard path escape."""
    root = collection_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def validate_presentation_dir(collection_root: Path, presentation_dir: Path) -> Path:
    """Ensure presentation directory is inside collection root."""
    resolved = presentation_dir.resolve()
    root = collection_root.resolve()
    if not str(resolved).startswith(str(root)):
        raise CollectionValidationError(
            f"Presentation path escapes collection root: {presentation_dir}"
        )
    return resolved


def next_sequence_dir(parent: Path, prefix: str) -> tuple[Path, str]:
    """
    Allocate the next session_XX or attempt_XX directory under parent.

    Returns (absolute_path, folder_name).
    """
    parent.mkdir(parents=True, exist_ok=True)
    highest = 0
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{2}})$")
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            highest = max(highest, int(match.group(1)))
    next_index = highest + 1
    if next_index > 99:
        raise CollectionValidationError(
            f"Sequence limit reached for {prefix} under {parent}"
        )
    folder_name = f"{prefix}_{next_index:02d}"
    return parent / folder_name, folder_name


def build_presentation_path(
    collection_root: Path,
    subject_id: str,
    presentation_type: str,
) -> PresentationPathInfo:
    """
    Build the next presentation directory for a subject and presentation type.

    Follows eval/datasets/face_lock_v1/ layout from FACE_LOCK_V1_COLLECTION_GUIDE.md.
    """
    root = resolve_collection_root(collection_root)
    subject_id = validate_subject_id(subject_id)
    presentation_type = validate_presentation_type(presentation_type)
    spec = PRESENTATION_SPECS[presentation_type]

    subject_dir = root / spec.relative_root / subject_id_to_folder_name(subject_id)
    presentation_dir, sequence_name = next_sequence_dir(subject_dir, spec.sequence_prefix)
    presentation_dir = validate_presentation_dir(root, presentation_dir)

    relative_path = presentation_dir.relative_to(root).as_posix()
    return PresentationPathInfo(
        subject_id=subject_id,
        presentation_type=presentation_type,
        presentation_dir=presentation_dir,
        sequence_name=sequence_name,
        relative_path=relative_path,
    )


def frame_filename(index: int) -> str:
    """Return frame_NNNN.jpg for a 1-based index."""
    if index < 1 or index > 9999:
        raise CollectionValidationError(f"Frame index out of range: {index}")
    return f"frame_{index:04d}.jpg"


def save_frame(image: Any, path: Path, jpeg_quality: int) -> None:
    """Write one BGR frame to disk as JPEG."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(
        str(path),
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        raise OSError(f"Failed to write frame: {path}")


def write_meta_json(
    presentation_dir: Path,
    *,
    collector_id: str,
    frame_count: int,
    camera_index: int,
    subject_id: str,
    presentation_type: str,
) -> Path:
    """Write optional meta.json for a captured presentation."""
    meta = {
        "collector_id": collector_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_count": frame_count,
        "camera_index": camera_index,
        "subject_id": subject_id,
        "presentation_type": presentation_type,
        "pai_id": PRESENTATION_SPECS[presentation_type].pai_id,
    }
    path = presentation_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sample_interval_seconds(capture_fps: float) -> float:
    """Wall-clock seconds between saved frames."""
    if capture_fps <= 0:
        raise CollectionValidationError("capture_fps must be > 0")
    return 1.0 / capture_fps


def effective_capture_fps(saved_frames: int, duration_seconds: float) -> float:
    """Measured save rate over a recording window."""
    if duration_seconds <= 0 or saved_frames <= 0:
        return 0.0
    return saved_frames / duration_seconds


def log_recording_stats(
    *,
    saved_frames: int,
    duration_seconds: float,
    capture_fps: float,
) -> None:
    """Log recording duration, saved count, and measured vs target FPS."""
    effective = effective_capture_fps(saved_frames, duration_seconds)
    LOGGER.info(
        "Recording stats: duration=%.2fs saved_frames=%d effective_fps=%.2f (capture_fps=%.2f)",
        duration_seconds,
        saved_frames,
        effective,
        capture_fps,
    )


def validate_frame_count(count: int, config: CollectionConfig) -> None:
    """Log warnings for frame counts outside recommended bounds."""
    if count < MIN_FRAMES_PROTOCOL:
        LOGGER.warning(
            "Capture has %s frames (< %s protocol minimum). Consider re-capture.",
            count,
            MIN_FRAMES_PROTOCOL,
        )
    elif count < config.min_frames:
        LOGGER.warning(
            "Capture has %s frames (< recommended minimum %s).",
            count,
            config.min_frames,
        )
    elif count > config.max_frames:
        LOGGER.warning(
            "Capture has %s frames (> recommended maximum %s).",
            count,
            config.max_frames,
        )


def draw_overlay(
    frame: Any,
    *,
    subject_id: str,
    presentation_type: str,
    state: str,
    frame_count: int,
    destination: str,
    min_frames: int,
    max_frames: int,
    capture_fps: float,
) -> Any:
    """Draw capture status text on the preview frame."""
    import cv2

    display = frame.copy()
    lines = [
        f"subject: {subject_id}  type: {presentation_type}",
        f"dest: {destination}",
        (
            f"state: {state}  saved: {frame_count}  target: {min_frames}-{max_frames}  "
            f"capture: {capture_fps:.1f} fps"
        ),
        "s=start  e=end/save  q=quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            display,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 26
    return display


def open_camera(camera_index: int) -> Any:
    """Open webcam capture device."""
    import cv2

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise CameraOpenError(
            f"Could not open camera index {camera_index}. "
            "Check device permissions and --camera-index."
        )
    return capture


def prompt_input(prompt: str) -> str:
    """Read a non-empty line from stdin."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Value cannot be empty.")


def prompt_subject_id() -> str:
    """Interactive subject_id prompt with validation."""
    while True:
        try:
            return validate_subject_id(prompt_input("subject_id (e.g. subj_001): "))
        except CollectionValidationError as error:
            print(f"Invalid subject_id: {error}")


def prompt_presentation_type() -> str:
    """Interactive presentation type prompt with validation."""
    options = ", ".join(sorted(SUPPORTED_PRESENTATION_TYPES))
    while True:
        try:
            return validate_presentation_type(
                prompt_input(f"presentation type [{options}]: ")
            )
        except CollectionValidationError as error:
            print(f"Invalid presentation type: {error}")


def save_captured_frames(
    frames: list[Any],
    presentation_dir: Path,
    config: CollectionConfig,
) -> int:
    """Persist captured frames to presentation_dir."""
    if not frames:
        raise CollectionValidationError("No frames to save.")

    to_save = frames[: config.max_frames]
    for index, image in enumerate(to_save, start=1):
        save_frame(image, presentation_dir / frame_filename(index), config.jpeg_quality)

    return len(to_save)


def finalize_presentation_capture(
    buffer: list[Any],
    path_info: PresentationPathInfo,
    config: CollectionConfig,
    *,
    recording_started_at: float,
) -> int:
    """Save buffered frames, optional meta, and log recording statistics."""
    saved = save_captured_frames(buffer, path_info.presentation_dir, config)
    validate_frame_count(saved, config)

    duration_seconds = time.monotonic() - recording_started_at
    log_recording_stats(
        saved_frames=saved,
        duration_seconds=duration_seconds,
        capture_fps=config.capture_fps,
    )

    if config.write_meta:
        meta_path = write_meta_json(
            path_info.presentation_dir,
            collector_id=config.collector_id,
            frame_count=saved,
            camera_index=config.camera_index,
            subject_id=path_info.subject_id,
            presentation_type=path_info.presentation_type,
        )
        LOGGER.info("Wrote %s", meta_path)

    LOGGER.info("Saved %s frames to %s", saved, path_info.presentation_dir)
    return saved


def run_capture_session(
    capture: Any,
    path_info: PresentationPathInfo,
    config: CollectionConfig,
) -> int:
    """
    Run preview and capture loop for one presentation.

    Returns number of frames saved.
    """
    import cv2

    recording = False
    buffer: list[Any] = []
    recording_started_at = 0.0
    next_sample_at = 0.0
    sample_interval = sample_interval_seconds(config.capture_fps)

    LOGGER.info(
        "Presentation directory: %s (%s)",
        path_info.presentation_dir,
        path_info.relative_path,
    )

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise CameraOpenError("Failed to read frame from camera.")

        state = "RECORDING" if recording else "idle"
        display = draw_overlay(
            frame,
            subject_id=path_info.subject_id,
            presentation_type=path_info.presentation_type,
            state=state,
            frame_count=len(buffer),
            destination=path_info.relative_path,
            min_frames=config.min_frames,
            max_frames=config.max_frames,
            capture_fps=config.capture_fps,
        )
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            if recording and buffer:
                LOGGER.info("Quit during recording; discarding in-progress frames.")
            raise KeyboardInterrupt

        if key in (ord("s"), ord("S")):
            if not recording:
                recording = True
                buffer = []
                recording_started_at = time.monotonic()
                next_sample_at = recording_started_at
                path_info.presentation_dir.mkdir(parents=True, exist_ok=True)
                LOGGER.info(
                    "Recording started (capture_fps=%.2f, interval=%.3fs).",
                    config.capture_fps,
                    sample_interval,
                )
            continue

        if key in (ord("e"), ord("E")):
            if not recording:
                LOGGER.info("Press 's' to start capture before 'e'.")
                continue
            recording = False
            if not buffer:
                LOGGER.warning("No frames captured; not saving.")
                continue

            return finalize_presentation_capture(
                buffer,
                path_info,
                config,
                recording_started_at=recording_started_at,
            )

        if recording:
            now = time.monotonic()
            if now >= next_sample_at:
                buffer.append(frame.copy())
                next_sample_at += sample_interval
                while next_sample_at <= now:
                    next_sample_at += sample_interval

            if len(buffer) >= config.max_frames:
                LOGGER.info("Max frames (%s) reached; auto-ending capture.", config.max_frames)
                recording = False
                return finalize_presentation_capture(
                    buffer,
                    path_info,
                    config,
                    recording_started_at=recording_started_at,
                )


def run_collector(
    config: CollectionConfig,
    *,
    subject_id: str | None = None,
    presentation_type: str | None = None,
    interactive: bool = True,
) -> None:
    """
    Main collection loop: preview, capture, save, optionally repeat.

    Parameters
    ----------
    config:
        Collection paths and frame limits.
    subject_id:
        Optional pre-set subject id; otherwise prompted when interactive.
    presentation_type:
        Optional pre-set type; otherwise prompted when interactive.
    interactive:
        When True, prompt for new presentations until quit.
    """
    import cv2

    capture = open_camera(config.camera_index)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    current_subject = validate_subject_id(subject_id) if subject_id else None
    current_type = (
        validate_presentation_type(presentation_type) if presentation_type else None
    )

    try:
        while True:
            if interactive and current_subject is None:
                current_subject = prompt_subject_id()
            if interactive and current_type is None:
                current_type = prompt_presentation_type()
            if current_subject is None or current_type is None:
                raise CollectionValidationError(
                    "subject_id and presentation_type are required."
                )

            path_info = build_presentation_path(
                config.collection_root,
                current_subject,
                current_type,
            )
            print(
                f"\nReady: {path_info.presentation_type} → {path_info.relative_path}\n"
                "Press 's' to start, 'e' to end/save, 'q' to quit.\n"
            )

            try:
                run_capture_session(capture, path_info, config)
            except KeyboardInterrupt:
                LOGGER.info("Collection stopped by user.")
                break

            if not interactive:
                break

            again = input("Capture another presentation? [y/N]: ").strip().lower()
            if again not in ("y", "yes"):
                break
            change = input("Change subject/type? [y/N]: ").strip().lower()
            if change in ("y", "yes"):
                current_subject = None
                current_type = None

    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    project_root = resolve_project_root()
    default_root = project_root / "eval" / "datasets" / "face_lock_v1"

    parser = argparse.ArgumentParser(
        description="Collect PAD evaluation clips for Face Lock AI (face_lock_v1 layout).",
    )
    parser.add_argument(
        "--collection-root",
        type=Path,
        default=default_root,
        help=f"Dataset root (default: {default_root})",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=12,
        help="Recommended minimum frames per presentation (default: 12).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Maximum frames per presentation (default: 16).",
    )
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=DEFAULT_CAPTURE_FPS,
        help=(
            "Save one frame every 1/capture_fps seconds while recording "
            f"(default: {DEFAULT_CAPTURE_FPS}). Preview stays at camera rate."
        ),
    )
    parser.add_argument(
        "--collector-id",
        type=str,
        default="collector_local",
        help="Collector id stored in meta.json.",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Subject id (subj_NNN). Prompted if omitted.",
    )
    parser.add_argument(
        "--presentation-type",
        type=str,
        default=None,
        help="BF, PA-01, PA-02, PA-03, or PA-04. Prompted if omitted.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality 0-100 (default: 92).",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Do not write meta.json after capture.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture one presentation and exit (non-interactive repeat).",
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

    if args.min_frames < MIN_FRAMES_PROTOCOL:
        parser.error(f"--min-frames must be >= {MIN_FRAMES_PROTOCOL}")
    if args.max_frames < args.min_frames:
        parser.error("--max-frames must be >= --min-frames")
    if not 0 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 0 and 100")
    if args.capture_fps <= 0:
        parser.error("--capture-fps must be > 0")

    config = CollectionConfig(
        collection_root=args.collection_root,
        camera_index=args.camera_index,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        capture_fps=args.capture_fps,
        collector_id=args.collector_id,
        write_meta=not args.no_meta,
        jpeg_quality=args.jpeg_quality,
    )

    try:
        run_collector(
            config,
            subject_id=args.subject_id,
            presentation_type=args.presentation_type,
            interactive=not args.once,
        )
    except CollectionValidationError as error:
        LOGGER.error("Validation error: %s", error)
        return 1
    except CameraOpenError as error:
        LOGGER.error("Camera error: %s", error)
        return 2
    except KeyboardInterrupt:
        LOGGER.info("Interrupted.")
        return 0

    LOGGER.info("Collection session finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
