"""
Webcam capture helpers for the Face Lock AI project.

This module owns camera concerns only:
- opening the webcam
- reading live frames
- saving frames to disk
- releasing the webcam
- closing OpenCV windows

It does not generate embeddings, compare faces, or make access decisions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

try:
    # Preferred imports when running the project from app.py.
    from src.config import WEBCAM_SETTINGS
    from src.logging_config import get_logger, log_event, log_exception
    from src.utils import ensure_directory, generate_image_filename, print_status
except ModuleNotFoundError:
    # Fallback imports when running this file directly as: python src/capture.py
    from config import WEBCAM_SETTINGS
    from logging_config import get_logger, log_event, log_exception
    from utils import ensure_directory, generate_image_filename, print_status


# OpenCV camera index. Usually 0 is the default laptop webcam.
DEFAULT_CAMERA_INDEX = WEBCAM_SETTINGS.camera_index

# Window title used by the manual preview in the __main__ block.
CAPTURE_WINDOW_NAME = WEBCAM_SETTINGS.capture_window_name

LOGGER = get_logger(__name__)


def open_webcam(camera_index: int = DEFAULT_CAMERA_INDEX) -> cv2.VideoCapture:
    """
    Open the webcam and return the OpenCV capture object.

    Args:
        camera_index: Camera number. Usually 0 is the default webcam.

    Returns:
        An opened cv2.VideoCapture object.

    Raises:
        RuntimeError: If the webcam cannot be opened.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        log_event(
            LOGGER,
            "webcam_open_failed",
            level=logging.ERROR,
            camera_index=camera_index,
        )
        raise RuntimeError(f"Could not open webcam with index {camera_index}.")

    log_event(LOGGER, "webcam_opened", camera_index=camera_index)
    return cap


def read_frame(cap: cv2.VideoCapture) -> Any:
    """
    Read one frame from an opened webcam.

    Args:
        cap: An opened cv2.VideoCapture object.

    Returns:
        The captured frame as a NumPy array.

    Raises:
        RuntimeError: If a frame cannot be read from the webcam.
    """
    success, frame = cap.read()

    if not success or frame is None:
        log_event(LOGGER, "webcam_read_failed", level=logging.ERROR)
        raise RuntimeError("Could not read a frame from the webcam.")

    return frame


def save_frame(
    frame: Any,
    output_dir: str | Path,
    filename_prefix: str,
) -> Path:
    """
    Save one frame as an image file inside a folder.

    Args:
        frame: Frame to save.
        output_dir: Folder where the image should be written.
        filename_prefix: Safe name used to build the image filename.

    Returns:
        The saved image path.

    Raises:
        RuntimeError: If the image cannot be written to disk.
    """
    folder = ensure_directory(output_dir)

    image_path = folder / generate_image_filename(filename_prefix)

    success = cv2.imwrite(str(image_path), frame)

    if not success:
        log_event(
            LOGGER,
            "frame_save_failed",
            level=logging.ERROR,
            image_path=image_path,
        )
        raise RuntimeError(f"Could not save the captured frame to {image_path}.")

    log_event(LOGGER, "frame_saved", image_path=image_path)
    return image_path


def release_webcam(cap: cv2.VideoCapture | None) -> None:
    """
    Release the webcam safely.

    Args:
        cap: The capture object to release, or None if it was never opened.
    """
    if cap is None:
        return

    try:
        cap.release()
        log_event(LOGGER, "webcam_released")
    except Exception as error:
        # Cleanup must never crash the workflow that is shutting down.
        log_exception(LOGGER, "webcam_release_failed", error)


def close_windows() -> None:
    """
    Close all OpenCV windows safely.
    """
    try:
        cv2.destroyAllWindows()
    except Exception as error:
        # Cleanup must never crash the workflow that is shutting down.
        log_exception(LOGGER, "windows_close_failed", error)


if __name__ == "__main__":
    # Manual test only. This block runs only when this file is executed directly.
    # It shows a live preview so the camera layer can be checked in isolation.
    preview_cap: cv2.VideoCapture | None = None

    try:
        preview_cap = open_webcam()
        print_status("Webcam preview started. Press 'q' to quit.")

        while True:
            preview_frame = read_frame(preview_cap)
            cv2.imshow(CAPTURE_WINDOW_NAME, preview_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(WEBCAM_SETTINGS.quit_key):
                print_status("Webcam preview stopped by user.")
                break

    except RuntimeError as error:
        print_status(str(error), level="error")

    finally:
        release_webcam(preview_cap)
        close_windows()
