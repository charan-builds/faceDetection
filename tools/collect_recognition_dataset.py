#!/usr/bin/env python3
"""
Face Lock AI — Controlled Real-World Recognition Dataset Collection.

Provides an interactive OpenCV webcam interface to collect consenting participants
under the explicit Face Recognition protocol:
  - session_01 (Enrollment): ~10 frames
  - session_02 (Evaluation): ~20 frames, with variation

Enforces strict face validation (exactly 1 face per frame) before saving.
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
import logging
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Final

from tools.project_paths import resolve_project_root

LOGGER = logging.getLogger("face_lock.collect_recognition")

SUBJECT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^subj_[0-9]{3}$")
WINDOW_NAME: Final[str] = "Face Lock Recognition Collection"
DEFAULT_CAPTURE_FPS: Final[float] = 5.0
ENROLLMENT_FRAMES: Final[int] = 10
EVALUATION_FRAMES: Final[int] = 20

MINIMUM_ENROLLMENT_FRAMES: Final[int] = 5
MINIMUM_EVALUATION_FRAMES: Final[int] = 10


class CollectionValidationError(ValueError):
    """Raised when user input or output paths are invalid."""


class CameraOpenError(RuntimeError):
    """Raised when the webcam cannot be opened."""


class StorageFailureError(RuntimeError):
    """Raised when writing to disk fails."""


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    collection_root: Path
    camera_index: int
    capture_fps: float
    jpeg_quality: int
    dry_run: bool


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_subject_id(subject_id: str) -> str:
    normalized = subject_id.strip().lower()
    if not SUBJECT_ID_PATTERN.match(normalized):
        raise CollectionValidationError("subject_id must match subj_NNN (e.g. subj_001)")
    return normalized


def get_next_subject_id(collection_root: Path) -> str:
    """Scan existing subj_NNN folders and return the next available ID."""
    bona_fide_dir = collection_root / "bona_fide"
    highest = 0
    if bona_fide_dir.is_dir():
        for child in bona_fide_dir.iterdir():
            if child.is_dir() and child.name.startswith("subj_"):
                try:
                    num = int(child.name.split("_")[1])
                    highest = max(highest, num)
                except ValueError:
                    pass
    return f"subj_{highest + 1:03d}"


def count_existing_frames(session_dir: Path) -> int:
    """Count frame_*.jpg files in the directory."""
    if not session_dir.is_dir():
        return 0
    return sum(1 for p in session_dir.iterdir() if p.name.startswith("frame_") and p.name.endswith(".jpg"))


def get_next_session(subject_dir: Path) -> tuple[str | None, int]:
    """
    Determine whether to capture session_01 or session_02, and what the starting index is.
    Returns (session_name, start_index). If None, subject is fully captured.
    """
    s1 = subject_dir / "session_01"
    s2 = subject_dir / "session_02"
    
    s1_frames = count_existing_frames(s1)
    if s1_frames < MINIMUM_ENROLLMENT_FRAMES:
        return "session_01", s1_frames
        
    s2_frames = count_existing_frames(s2)
    if s2_frames < MINIMUM_EVALUATION_FRAMES:
        return "session_02", s2_frames
        
    return None, 0


def hash_frame(frame: Any) -> str:
    """Hash a numpy array for duplicate detection."""
    return hashlib.sha256(frame.tobytes()).hexdigest()


def validate_faces(frames: list[Any]) -> tuple[list[Any], int, int, int]:
    """
    Run DeepFace offline on the captured buffer and drop duplicates.
    Returns (valid_frames, zero_faces_count, multiple_faces_count, duplicate_count).
    """
    try:
        from deepface import DeepFace
    except ImportError:
        LOGGER.warning("DeepFace not installed. Relying only on duplicate dropping.")
        DeepFace = None

    valid_frames = []
    zero_count = 0
    multiple_count = 0
    duplicate_count = 0
    seen_hashes = set()

    for frame in frames:
        h = hash_frame(frame)
        if h in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(h)

        if DeepFace is not None:
            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    enforce_detection=False,
                    detector_backend="opencv",
                )
                valid_faces = [f for f in faces if f.get("confidence", 0) > 0]
                count = len(valid_faces)
                
                if count == 0:
                    zero_count += 1
                elif count > 1:
                    multiple_count += 1
                else:
                    valid_frames.append(frame)
            except ValueError:
                zero_count += 1
        else:
            # Fallback if no DeepFace
            valid_frames.append(frame)

    return valid_frames, zero_count, multiple_count, duplicate_count


def save_frames(frames: list[Any], target_dir: Path, config: CollectionConfig, start_index: int = 0) -> int:
    """Write valid frames to disk, handling storage errors."""
    import cv2
    
    if config.dry_run:
        LOGGER.info(f"DRY RUN: Would have saved {len(frames)} frames to {target_dir}")
        return len(frames)
        
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise StorageFailureError(f"Failed to create directory {target_dir}: {e}")
    
    saved_count = 0
    for i, frame in enumerate(frames, start=start_index + 1):
        path = target_dir / f"frame_{i:04d}.jpg"
        try:
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality]):
                raise StorageFailureError(f"cv2.imwrite returned False for {path}")
            saved_count += 1
        except Exception as e:
            # Cleanup partially saved frames on failure
            LOGGER.error(f"Storage failure during save: {e}")
            for j in range(start_index + 1, start_index + 1 + saved_count):
                bad_path = target_dir / f"frame_{j:04d}.jpg"
                if bad_path.exists():
                    bad_path.unlink()
            raise StorageFailureError("Storage failure occurred. Cleaned up partially written frames.")
            
    return saved_count


def draw_overlay(
    frame: Any,
    subject_id: str,
    session_name: str,
    target_frames: int,
    buffered_frames: int,
    is_recording: bool,
    dry_run: bool,
) -> Any:
    """Draw UI on preview frame."""
    import cv2
    display = frame.copy()
    
    state = "RECORDING" if is_recording else "READY"
    if dry_run:
        state = "[DRY RUN] " + state
        
    lines = [
        f"Subject: {subject_id} | Session: {session_name}",
        f"State: {state} | Buffer: {buffered_frames} / {target_frames}",
        "Press 's' to start/stop, 'q' to quit"
    ]
    
    if session_name == "session_02":
        lines.append("Vary your pose, expression, and distance!")

    y = 30
    for line in lines:
        cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28

    return display


def run_capture_session(
    capture: Any,
    config: CollectionConfig,
    subject_id: str,
    session_name: str,
    target_frames: int,
    start_index: int,
) -> tuple[int, int, int]:
    """
    Handle preview, capture, validation, and save loop for one session.
    Returns (accepted, rejected_zero, rejected_multiple).
    """
    import cv2
    
    recording = False
    buffer = []
    next_sample_at = 0.0
    sample_interval = 1.0 / config.capture_fps

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise CameraOpenError("Failed to read from camera.")

        display = draw_overlay(frame, subject_id, session_name, target_frames, len(buffer), recording, config.dry_run)
        cv2.imshow(WINDOW_NAME, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key in (ord("q"), ord("Q")):
            raise KeyboardInterrupt

        if key in (ord("s"), ord("S")):
            if not recording:
                recording = True
                buffer = []
                next_sample_at = time.monotonic()
                LOGGER.info(f"Started capturing {session_name} for {subject_id}...")
            else:
                recording = False
                LOGGER.info("Capture stopped early by user.")
                break

        if recording:
            now = time.monotonic()
            if now >= next_sample_at:
                buffer.append(frame.copy())
                next_sample_at += sample_interval
                
            if len(buffer) >= target_frames:
                recording = False
                LOGGER.info(f"Target frame count ({target_frames}) reached.")
                break

    if not buffer:
        LOGGER.warning("No frames captured.")
        return 0, 0, 0

    LOGGER.info(f"Validating {len(buffer)} frames with DeepFace...")
    valid_frames, zero_count, multiple_count, dup_count = validate_faces(buffer)
    
    if not valid_frames:
        LOGGER.error(f"All captured frames rejected! (Zero: {zero_count}, Multiple: {multiple_count}, Dups: {dup_count})")
        return 0, zero_count, multiple_count
        
    target_dir = config.collection_root / "bona_fide" / subject_id / session_name
    
    try:
        saved = save_frames(valid_frames, target_dir, config, start_index)
    except StorageFailureError as e:
        LOGGER.error(str(e))
        return 0, 0, 0
    
    LOGGER.info(f"Saved {saved} valid frames to {target_dir}")
    if zero_count > 0 or multiple_count > 0 or dup_count > 0:
        LOGGER.warning(f"Rejected frames -> Zero: {zero_count} | Multiple: {multiple_count} | Duplicates: {dup_count}")
        
    return saved, zero_count, multiple_count


def run_collector(config: CollectionConfig) -> None:
    """Main interactive collection loop."""
    import cv2
    capture = open_camera(config.camera_index)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    participants_enrolled = 0
    participants_completed = 0
    total_accepted = 0
    total_rejected = 0

    try:
        while True:
            suggested_id = get_next_subject_id(config.collection_root)
            print(f"\n--- New Participant ---")
            raw_id = input(f"Enter anonymous subject ID (default: {suggested_id}): ").strip()
            subject_id = validate_subject_id(raw_id) if raw_id else suggested_id
            
            subject_dir = config.collection_root / "bona_fide" / subject_id
            
            while True:
                session_name, start_idx = get_next_session(subject_dir)
                if not session_name:
                    print(f"\nSubject {subject_id} is already fully captured (both sessions present).")
                    break
                    
                target = ENROLLMENT_FRAMES if session_name == "session_01" else EVALUATION_FRAMES
                target = max(1, target - start_idx)  # How many left to reach the ideal count
                print(f"\nReady to capture: {session_name} for {subject_id} (Need ~{target} more frames)")
                
                accepted, rej_zero, rej_mult = run_capture_session(
                    capture, config, subject_id, session_name, target, start_idx
                )
                total_accepted += accepted
                total_rejected += (rej_zero + rej_mult)
                
                # Check completion based on total frames now
                total_frames_now = start_idx + accepted
                
                if session_name == "session_01":
                    if total_frames_now >= MINIMUM_ENROLLMENT_FRAMES:
                        participants_enrolled += 1
                        print(f"[OK] {session_name} completed.")
                    else:
                        print(f"\n[WARNING] INCOMPLETE PARTICIPANT: {session_name} has {total_frames_now} frames (minimum {MINIMUM_ENROLLMENT_FRAMES}).")
                elif session_name == "session_02":
                    if total_frames_now >= MINIMUM_EVALUATION_FRAMES:
                        participants_completed += 1
                        print(f"[OK] {session_name} completed.")
                    else:
                        print(f"\n[WARNING] INCOMPLETE PARTICIPANT: {session_name} has {total_frames_now} frames (minimum {MINIMUM_EVALUATION_FRAMES}).")
                
            again = input("\nCapture another participant? [y/N]: ").strip().lower()
            if again not in ("y", "yes"):
                break

    except KeyboardInterrupt:
        LOGGER.info("Collection interrupted by operator.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("\n--- Collection Summary ---")
    print(f"Participants Enrolled (Session 1): {participants_enrolled}")
    print(f"Participants Evaluated (Session 2): {participants_completed}")
    print(f"Total Accepted Frames: {total_accepted}")
    print(f"Total Rejected Frames: {total_rejected}")
    if not config.dry_run:
        print("\nNext step: Run `python tools/generate_manifest.py --collection-root <path>` to generate the manifest.")


def open_camera(camera_index: int) -> Any:
    import cv2
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise CameraOpenError(f"Could not open camera {camera_index}.")
    return capture


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = resolve_project_root()
    default_root = project_root / "eval" / "datasets" / "face_lock_real"

    parser = argparse.ArgumentParser(description="Collect real-world Face Recognition dataset.")
    parser.add_argument("--collection-root", type=Path, default=default_root, help=f"Root (default: {default_root})")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--capture-fps", type=float, default=DEFAULT_CAPTURE_FPS, help="Capture frame rate")
    parser.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality 0-100")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing to disk.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging()
    
    config = CollectionConfig(
        collection_root=args.collection_root,
        camera_index=args.camera_index,
        capture_fps=args.capture_fps,
        jpeg_quality=args.jpeg_quality,
        dry_run=args.dry_run,
    )
    
    try:
        run_collector(config)
    except Exception as e:
        LOGGER.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
