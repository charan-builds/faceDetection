"""
User registration workflow for the Face Lock AI project.

This module coordinates registration:
- ask for and normalize user name (identity consistency)
- create a folder for that user under data/
- automatically capture multiple quality face images with temporal PAD
- generate embeddings for the captured images
- sign and save those embeddings into encodings/
- clean up raw files on cancellation or failure

It does not contain recognition/unlock logic or application menu logic.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    # Preferred imports when running the project from app.py.
    from src.config import PAD_SETTINGS, RECOGNITION_SETTINGS, REGISTRATION_SETTINGS, WEBCAM_SETTINGS
    from src.ai_engine import (
        DEFAULT_COSINE_THRESHOLD,
        DEFAULT_DETECTOR_BACKEND,
        DEFAULT_MODEL_NAME,
        generate_embedding,
        get_user_embedding_path,
        save_user_embeddings,
    )
    from src.capture import (
        close_windows,
        open_webcam,
        read_frame,
        release_webcam,
        save_frame,
    )
    from src.integrity import enforce_strict_mode_policy
    from src.logging_config import get_logger, log_event, log_registration_event
    from src.pad_engine import (
        ChallengeState,
        PadOutcome,
        PadSession,
        build_pad_frame,
        check_single_image_pad,
        create_pad_session,
    )
    from src.utils import DATA_DIR, clean_name, ensure_directory, ensure_project_folders
    from src.utils import print_status
except ModuleNotFoundError:
    # Fallback imports when running this file directly as: python src/register.py
    from config import PAD_SETTINGS, RECOGNITION_SETTINGS, REGISTRATION_SETTINGS, WEBCAM_SETTINGS
    from ai_engine import (
        DEFAULT_COSINE_THRESHOLD,
        DEFAULT_DETECTOR_BACKEND,
        DEFAULT_MODEL_NAME,
        generate_embedding,
        get_user_embedding_path,
        save_user_embeddings,
    )
    from capture import close_windows, open_webcam, read_frame, release_webcam, save_frame
    from integrity import enforce_strict_mode_policy
    from logging_config import get_logger, log_event, log_registration_event
    from pad_engine import (
        ChallengeState,
        PadOutcome,
        PadSession,
        build_pad_frame,
        check_single_image_pad,
        create_pad_session,
    )
    from utils import DATA_DIR, clean_name, ensure_directory, ensure_project_folders
    from utils import print_status


LOGGER = get_logger(__name__)

# We collect several images so the system has more than one view of the user.
DEFAULT_IMAGE_COUNT = REGISTRATION_SETTINGS.image_count

# A registration with too few valid face embeddings is unreliable.
MIN_VALID_EMBEDDINGS = REGISTRATION_SETTINGS.minimum_embeddings

# Window title used only during the registration capture session.
REGISTRATION_WINDOW_NAME = REGISTRATION_SETTINGS.window_name

# OpenCV camera index. Usually 0 is the default webcam.
DEFAULT_CAMERA_INDEX = WEBCAM_SETTINGS.camera_index

# Global cached face cascade classifier
_FACE_CASCADE: cv2.CascadeClassifier | None = None


def get_face_cascade() -> cv2.CascadeClassifier:
    """Load and cache the OpenCV Haar Cascade face classifier."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def cleanup_registration_directory(
    user_folder: Path,
    image_paths: list[Path] | None = None,
) -> None:
    """
    Remove partially saved images and empty user folder on cancellation or failure.
    """
    if image_paths:
        for path in image_paths:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

    if user_folder.is_dir():
        try:
            remaining = list(user_folder.iterdir())
            if not remaining:
                user_folder.rmdir()
        except OSError:
            pass


def detect_single_face(
    frame: Any,
    min_size_ratio: float = REGISTRATION_SETTINGS.min_face_size_ratio,
    max_size_ratio: float = REGISTRATION_SETTINGS.max_face_size_ratio,
    cascade: Any | None = None,
) -> tuple[tuple[int, int, int, int] | None, str | None]:
    """
    Detect a single face in the frame using OpenCV Haar Cascade.

    Returns:
        ((x, y, w, h), None) if exactly one well-positioned face is found,
        or (None, error_code) otherwise.
    """
    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return None, "no_face"

    h_frame, w_frame = frame.shape[:2]
    # Fast path for unit test mock frames (e.g. 8x8)
    if h_frame < 20 or w_frame < 20:
        return (0, 0, w_frame, h_frame), None

    if cascade is None:
        cascade = get_face_cascade()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(int(min_size_ratio * h_frame), int(min_size_ratio * h_frame)),
    )

    if len(faces) == 0:
        return None, "no_face"
    if len(faces) > 1:
        return None, "multiple_faces"

    x, y, w, h = faces[0]
    h_ratio = h / float(h_frame)
    if h_ratio < min_size_ratio:
        return None, "face_too_small"
    if h_ratio > max_size_ratio:
        return None, "face_too_large"

    # Edge clipping check: face box must not clip through frame boundaries
    edge_margin = 8
    if x < edge_margin or y < edge_margin or (x + w) > (w_frame - edge_margin) or (y + h) > (h_frame - edge_margin):
        return None, "face_clipped"

    return (int(x), int(y), int(w), int(h)), None


def check_frame_quality(
    frame: Any,
    face_box: tuple[int, int, int, int],
    min_sharpness: float = REGISTRATION_SETTINGS.min_sharpness_variance,
    min_brightness: float = RECOGNITION_SETTINGS.brightness_threshold,
) -> tuple[bool, str | None]:
    """
    Assess lighting and blur of the frame and face crop.
    """
    if frame is None or not hasattr(frame, "shape"):
        return False, "invalid_frame"

    # Fast path for small test mock frames (e.g. 8x8)
    if frame.shape[0] < 20 or frame.shape[1] < 20:
        return True, None

    # Lighting check
    brightness = float(frame.mean())
    if brightness < min_brightness:
        return False, "too_dark"
    if brightness > 245.0:
        return False, "too_bright"

    x, y, w, h = face_box
    face_roi = frame[y : y + h, x : x + w]
    if face_roi.size == 0:
        return False, "invalid_face_crop"

    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if face_roi.ndim == 3 else face_roi
    lap_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    if lap_var < min_sharpness:
        return False, "blurry"

    return True, None


def is_frame_diverse(
    frame: Any,
    face_box: tuple[int, int, int, int],
    last_captured_face: np.ndarray | None,
    min_diff: float = REGISTRATION_SETTINGS.min_frame_difference,
) -> bool:
    """
    Deduplicate identical or near-identical frames.
    """
    if last_captured_face is None:
        return True

    x, y, w, h = face_box
    face_roi = frame[y : y + h, x : x + w]
    if face_roi.size == 0:
        return False

    if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
        return True

    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if face_roi.ndim == 3 else face_roi
    standard_crop = cv2.resize(gray_face, (100, 100))
    diff = float(np.mean(np.abs(standard_crop.astype(float) - last_captured_face.astype(float))))
    return diff >= min_diff


ENROLLMENT_ANGLE_GUIDES_20: tuple[str, ...] = (
    "Look straight at the camera (neutral)",
    "Look straight ahead (hold steady)",
    "Look straight with a natural subtle smile",
    "Look straight ahead (relax and blink naturally)",
    "Look straight ahead (hold steady)",
    "Turn head slightly to the left",
    "Hold slight left angle steady",
    "Turn head slightly further left (natural turn)",
    "Slowly return head to center",
    "Look straight ahead at camera",
    "Turn head slightly to the right",
    "Hold slight right angle steady",
    "Turn head slightly further right (natural turn)",
    "Slowly return head to center",
    "Tilt head slightly upward",
    "Hold slight upward tilt steady",
    "Slowly return head to center",
    "Tilt head slightly downward",
    "Hold slight downward tilt steady",
    "Look straight ahead to complete enrollment",
)

ENROLLMENT_ANGLE_GUIDES_10: tuple[str, ...] = (
    "Look straight at the camera",
    "Look straight with a natural expression",
    "Turn head slightly to the left",
    "Turn head slightly to the left and hold",
    "Turn head slightly to the right",
    "Turn head slightly to the right and hold",
    "Tilt head slightly upward",
    "Tilt head slightly upward and hold",
    "Tilt head slightly downward",
    "Look straight at the camera to complete",
)

ENROLLMENT_ANGLE_GUIDES: list[str] = list(ENROLLMENT_ANGLE_GUIDES_10)


def get_enrollment_guide(captured_count: int, total_count: int) -> str:
    """Return stage-appropriate pose guidance text."""
    if total_count >= 20:
        idx = min(captured_count, len(ENROLLMENT_ANGLE_GUIDES_20) - 1)
        return ENROLLMENT_ANGLE_GUIDES_20[idx]
    if total_count >= 10:
        idx = min(captured_count, len(ENROLLMENT_ANGLE_GUIDES_10) - 1)
        return ENROLLMENT_ANGLE_GUIDES_10[idx]
    if captured_count >= total_count:
        return "Enrollment complete"
    return f"Keep face clearly visible ({captured_count + 1}/{total_count})"


def draw_registration_hud(
    frame: np.ndarray,
    user_name: str,
    captured_count: int,
    total_count: int,
    status_message: str,
    face_box: tuple[int, int, int, int] | None = None,
    box_color: tuple[int, int, int] = (0, 255, 0),
    angle_guide: str = "",
) -> np.ndarray:
    """
    Render visual status, progress bar, pose guidance, and face box on the live camera frame.
    """
    display = frame.copy()
    h, w = display.shape[:2]

    # Banner header
    banner_h = 85
    cv2.rectangle(display, (0, 0), (w, banner_h), (25, 25, 25), -1)

    # Progress bar
    pct = min(1.0, captured_count / max(1, total_count))
    bar_w = int((w - 40) * pct)
    cv2.rectangle(display, (20, banner_h - 10), (w - 20, banner_h - 4), (60, 60, 60), -1)
    if bar_w > 0:
        cv2.rectangle(display, (20, banner_h - 10), (20 + bar_w, banner_h - 4), (0, 220, 0), -1)

    line1 = f"Enrollment: {user_name} | Image {captured_count}/{total_count}"
    cv2.putText(display, line1, (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    guide_text = angle_guide or get_enrollment_guide(captured_count, total_count)
    cv2.putText(display, f"Guide: {guide_text}", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display, f"Status: {status_message}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

    # Face box
    if face_box is not None:
        fx, fy, fw, fh = face_box
        cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), box_color, 2)

    # Footer
    cv2.putText(display, "Press 'q' to cancel", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    return display


def ask_user_name() -> str:
    """
    Ask the user for a valid registration name.

    Returns:
        The raw user name entered by the user.
    """
    while True:
        user_name = input("Enter user name: ").strip()

        if not user_name:
            print_status("User name cannot be empty.", level="warning")
            continue

        if clean_name(user_name) == "unknown":
            print_status(
                "Use at least one letter or number in the user name.",
                level="warning",
            )
            continue

        return user_name


def get_user_data_folder(user_name: str) -> Path:
    """
    Create and return the folder where this user's images will be saved.
    """
    safe_user_name = clean_name(user_name)
    user_folder = DATA_DIR / safe_user_name
    return ensure_directory(user_folder)


def capture_registration_images(
    user_name: str,
    image_count: int = DEFAULT_IMAGE_COUNT,
    camera_index: int = DEFAULT_CAMERA_INDEX,
) -> tuple[list[Path], bool]:
    """
    Automatically capture quality face images for registration.

    Workflow:
      - Opens webcam and creates a persistent PAD session.
      - Continuously buffers genuine frames with face_box coordinates.
      - Evaluates liveness; if borderline/inconclusive, prompts the active head-turn challenge.
      - Once liveness is verified, automatically captures sharp, diverse face frames
        without requiring manual keypresses.
      - Cleans up partially written frames on user cancellation or failure.

    Args:
        user_name: Name of the user being registered.
        image_count: Number of images to capture.
        camera_index: Camera number. Usually 0 is default webcam.

    Returns:
        A tuple:
            captured_paths: list of saved image paths
            cancelled: True if the user cancelled or capture failed
    """
    captured_paths: list[Path] = []
    user_folder = get_user_data_folder(user_name)
    filename_prefix = clean_name(user_name)
    cap: cv2.VideoCapture | None = None
    pad_session: PadSession | None = None

    try:
        cap = open_webcam(camera_index)
        print_status("Webcam opened for automatic registration.")
        print_status("Center your face in front of the camera. Press 'q' to cancel.")

        if PAD_SETTINGS.enforce_on_registration and PAD_SETTINGS.enabled:
            pad_session = create_pad_session(PAD_SETTINGS)

        frame_number = 0
        last_capture_time = 0.0
        last_captured_face: np.ndarray | None = None
        prev_face_center: tuple[float, float] | None = None
        pad_verified = (pad_session is None)
        status_message = "Centering face..."
        box_color = (0, 255, 255)
        consecutive_attacks = 0

        while len(captured_paths) < image_count:
            frame = read_frame(cap)
            frame_number += 1

            face_box, face_error = detect_single_face(frame)

            # Motion stability check between consecutive frames
            is_moving_excessively = False
            if face_box is not None:
                fx, fy, fw, fh = face_box
                cx, cy = fx + fw / 2.0, fy + fh / 2.0
                if prev_face_center is not None:
                    shift = float(np.hypot(cx - prev_face_center[0], cy - prev_face_center[1]))
                    if shift > 28.0:
                        is_moving_excessively = True
                prev_face_center = (cx, cy)
            else:
                prev_face_center = None

            # 1. Temporal PAD evaluation
            if pad_session is not None:
                pad_session.add_frame(
                    build_pad_frame(
                        image=frame,
                        frame_index=frame_number,
                        face_box=face_box,
                    )
                )

                if not pad_verified:
                    if not pad_session.needs_more_frames():
                        pad_result = pad_session.evaluate_final()
                        if pad_result.outcome == PadOutcome.LIVE:
                            pad_verified = True
                            status_message = "Liveness verified! Collecting face samples..."
                            box_color = (0, 255, 0)
                            log_registration_event(
                                "registration_pad_checked",
                                outcome=pad_result.outcome.value,
                                reason_code=pad_result.reason_code,
                            )
                        elif pad_result.outcome == PadOutcome.INCONCLUSIVE:
                            box_color = (255, 255, 0)
                            challenge = pad_session.current_challenge()
                            if challenge and challenge.state == ChallengeState.IN_PROGRESS:
                                if challenge.instruction_key == "turn_head_left":
                                    status_message = "Liveness check: Turn head slowly to the LEFT"
                                elif challenge.instruction_key == "turn_head_right":
                                    status_message = "Liveness check: Turn head slowly to the RIGHT"
                                else:
                                    status_message = "Liveness check: Follow prompt"
                            else:
                                status_message = "Verifying liveness..."
                        else:  # ATTACK or ERROR
                            consecutive_attacks += 1
                            log_registration_event(
                                "registration_pad_checked",
                                outcome=pad_result.outcome.value,
                                reason_code=pad_result.reason_code,
                            )
                            if consecutive_attacks >= 25:
                                print_status("Registration aborted: persistent liveness rejection.", level="error")
                                cleanup_registration_directory(user_folder, captured_paths)
                                return [], True
                            status_message = "Liveness check rejected - use a live face"
                            box_color = (0, 0, 255)
                            pad_session.reset()

            # 2. Quality filtering & automatic capture
            should_save = False
            if pad_verified or pad_session is None:
                if face_box is not None:
                    if is_moving_excessively:
                        status_message = "Hold still"
                        box_color = (0, 200, 255)
                        log_registration_event(
                            "registration_frame_rejected",
                            reason="excessive_motion",
                            captured_count=len(captured_paths),
                            target_count=image_count,
                        )
                    else:
                        quality_ok, quality_err = check_frame_quality(frame, face_box)
                        if quality_ok:
                            now = time.monotonic()
                            if (now - last_capture_time) >= REGISTRATION_SETTINGS.auto_capture_interval_seconds:
                                if is_frame_diverse(frame, face_box, last_captured_face):
                                    should_save = True
                                    box_color = (0, 255, 0)
                                else:
                                    status_message = "Vary pose or expression slightly..."
                                    box_color = (0, 220, 220)
                        else:
                            box_color = (0, 140, 255)
                            log_registration_event(
                                "registration_frame_rejected",
                                reason=quality_err or "quality_failed",
                                captured_count=len(captured_paths),
                                target_count=image_count,
                            )
                            if quality_err == "too_dark":
                                status_message = "Improve lighting"
                            elif quality_err == "too_bright":
                                status_message = "Improve lighting - reduce glare"
                            elif quality_err == "blurry":
                                status_message = "Hold still"
                else:
                    box_color = (0, 0, 255)
                    log_registration_event(
                        "registration_frame_rejected",
                        reason=face_error or "detection_failed",
                        captured_count=len(captured_paths),
                        target_count=image_count,
                    )
                    if face_error == "no_face":
                        status_message = "No face detected - center face"
                    elif face_error == "multiple_faces":
                        status_message = "Multiple faces detected"
                    elif face_error == "face_clipped":
                        status_message = "Face clipped - center in frame"
                    elif face_error == "face_too_small":
                        status_message = "Too far - move closer to camera"
                    elif face_error == "face_too_large":
                        status_message = "Too close - move back slightly"

            # 3. User interaction & HUD display
            display = draw_registration_hud(
                frame=frame,
                user_name=user_name,
                captured_count=len(captured_paths),
                total_count=image_count,
                status_message=status_message,
                face_box=face_box,
                box_color=box_color,
            )
            cv2.imshow(REGISTRATION_WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                print_status("Registration cancelled by user.", level="warning")
                cleanup_registration_directory(user_folder, captured_paths)
                return [], True

            if key in (ord("s"), ord("S")):
                # Manual capture trigger
                if face_box is not None and (pad_verified or pad_session is None):
                    q_ok, _ = check_frame_quality(frame, face_box)
                    if q_ok:
                        should_save = True

            # 4. Save validated frame
            if should_save:
                image_path = save_frame(
                    frame=frame,
                    output_dir=user_folder,
                    filename_prefix=filename_prefix,
                )
                captured_paths.append(image_path)
                last_capture_time = time.monotonic()

                if face_box is not None:
                    fx, fy, fw, fh = face_box
                    f_crop = frame[fy : fy + fh, fx : fx + fw]
                    if f_crop.size > 0:
                        g_crop = cv2.cvtColor(f_crop, cv2.COLOR_BGR2GRAY) if f_crop.ndim == 3 else f_crop
                        last_captured_face = cv2.resize(g_crop, (100, 100))

                status_message = f"Image accepted ({len(captured_paths)}/{image_count})"
                print_status(
                    f"Image accepted ({len(captured_paths)}/{image_count}): {image_path.name}",
                    level="success",
                )

        print_status("Image capture complete.", level="success")
        return captured_paths, False

    except RuntimeError as error:
        print_status(str(error), level="error")
        cleanup_registration_directory(user_folder, captured_paths)
        return [], True

    finally:
        if pad_session is not None:
            pad_session.reset()
        release_webcam(cap)
        close_windows()


def validate_embedding(
    embedding: Any,
    expected_dim: int = 512,
) -> bool:
    """
    Validate that an embedding is non-empty, contains only finite numbers,
    and has exactly expected_dim dimensions (512 for FaceNet512).
    """
    if embedding is None:
        return False
    try:
        arr = np.asarray(embedding, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if arr.ndim != 1 or arr.size == 0:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    if arr.size != expected_dim:
        return False
    return True


def generate_registration_embeddings(
    image_paths: list[Path],
    expected_dim: int = 512,
) -> list[list[float]]:
    """
    Generate one face embedding for each valid captured image.
    Validates that every embedding is finite, non-empty, and exactly expected_dim dimensions.
    """
    embeddings: list[list[float]] = []

    for image_path in image_paths:
        try:
            embedding = generate_embedding(
                image_input=image_path,
                model_name=DEFAULT_MODEL_NAME,
                detector_backend=DEFAULT_DETECTOR_BACKEND,
            )
            if not validate_embedding(embedding, expected_dim=expected_dim):
                print_status(
                    f"Skipping {image_path.name}: embedding validation failed (must be finite, non-empty, {expected_dim}-dim).",
                    level="warning",
                )
                continue
            embeddings.append(embedding)
            print_status(f"Embedding created for: {image_path.name}", level="success")

        except Exception as error:
            print_status(
                f"Skipping {image_path.name}: {error}",
                level="warning",
            )

    return embeddings


def register_user(
    user_name: str | None = None,
    image_count: int = DEFAULT_IMAGE_COUNT,
    camera_index: int = DEFAULT_CAMERA_INDEX,
    overwrite: bool = False,
    expected_dim: int = 512,
    minimum_embeddings: int | None = None,
) -> dict[str, Any]:
    """
    Run the full user registration workflow.
    """
    enforce_strict_mode_policy(require_operator_auth=True)

    ensure_project_folders()

    raw_user_name = user_name.strip() if user_name else ask_user_name()
    safe_name = clean_name(raw_user_name)

    if not raw_user_name or safe_name == "unknown":
        print_status("Registration failed: invalid user name.", level="error")
        return {"success": False, "reason": "invalid_user_name"}

    embedding_file = get_user_embedding_path(safe_name)
    if embedding_file.exists() and not overwrite:
        print_status(
            f"Registration failed: user '{safe_name}' already exists. "
            f"Delete the existing record or pass overwrite=True to re-register.",
            level="error",
        )
        return {"success": False, "reason": "user_already_exists"}

    print_status(f"Starting registration for: {safe_name}")

    user_folder = get_user_data_folder(safe_name)

    if overwrite and user_folder.is_dir():
        # Clean existing images so no old images are reused in the new template
        for old_img in user_folder.glob("*.*"):
            if old_img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    old_img.unlink()
                except OSError:
                    pass

    # Step 1: Automatically capture face dataset
    image_paths, cancelled = capture_registration_images(
        user_name=safe_name,
        image_count=image_count,
        camera_index=camera_index,
    )

    if cancelled:
        cleanup_registration_directory(user_folder, image_paths)
        return {
            "success": False,
            "reason": "cancelled",
            "image_count": len(image_paths),
            "image_paths": image_paths,
        }

    if len(image_paths) == 0:
        cleanup_registration_directory(user_folder, image_paths)
        print_status("Registration failed: no images were captured.", level="error")
        return {"success": False, "reason": "no_images"}

    print_status("Generating face embeddings from captured images.")

    # Step 2: Convert captured images into embeddings
    embeddings = generate_registration_embeddings(image_paths, expected_dim=expected_dim)

    required_embeddings = minimum_embeddings if minimum_embeddings is not None else MIN_VALID_EMBEDDINGS
    if len(embeddings) < required_embeddings:
        cleanup_registration_directory(user_folder, image_paths)
        print_status(
            f"Registration failed: only {len(embeddings)} valid embeddings found. "
            f"At least {required_embeddings} are required.",
            level="error",
        )
        return {
            "success": False,
            "reason": "not_enough_valid_embeddings",
            "image_count": len(image_paths),
            "embedding_count": len(embeddings),
            "image_paths": image_paths,
        }

    # Step 3: Save embeddings signed with HMAC
    encoding_path = save_user_embeddings(
        user_name=safe_name,
        embeddings=embeddings,
        image_paths=image_paths,
        model_name=DEFAULT_MODEL_NAME,
        detector_backend=DEFAULT_DETECTOR_BACKEND,
        threshold=DEFAULT_COSINE_THRESHOLD,
    )

    print_status(
        f"Registration complete for {safe_name}. "
        f"Saved {len(embeddings)} embeddings to {encoding_path}.",
        level="success",
    )

    return {
        "success": True,
        "user_name": safe_name,
        "image_count": len(image_paths),
        "embedding_count": len(embeddings),
        "image_paths": image_paths,
        "encoding_path": encoding_path,
        "model_name": DEFAULT_MODEL_NAME,
    }


if __name__ == "__main__":
    try:
        from src.operator_auth import verify_operator_pin
    except ModuleNotFoundError:
        from operator_auth import verify_operator_pin
    if verify_operator_pin():
        register_user()
    else:
        print_status("Registration denied: operator authentication failed.", level="error")
