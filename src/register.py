"""
User registration workflow for the Face Lock AI project.

This module coordinates registration only:
- ask for a user name
- create a folder for that user
- capture multiple face images
- generate embeddings for the captured images
- save those embeddings into encodings/

It does not contain recognition/unlock logic or application menu logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

try:
    # Preferred imports when running the project from app.py.
    from src.config import PAD_SETTINGS, REGISTRATION_SETTINGS, WEBCAM_SETTINGS
    from src.ai_engine import (
        DEFAULT_COSINE_THRESHOLD,
        DEFAULT_DETECTOR_BACKEND,
        DEFAULT_MODEL_NAME,
        generate_embedding,
        save_user_embeddings,
    )
    from src.capture import (
        close_windows,
        open_webcam,
        read_frame,
        release_webcam,
        save_frame,
    )
    from src.logging_config import log_registration_event
    from src.pad_engine import PadOutcome, check_single_image_pad
    from src.utils import DATA_DIR, clean_name, ensure_directory, ensure_project_folders
    from src.utils import print_status
except ModuleNotFoundError:
    # Fallback imports when running this file directly as: python src/register.py
    from config import PAD_SETTINGS, REGISTRATION_SETTINGS, WEBCAM_SETTINGS
    from ai_engine import (
        DEFAULT_COSINE_THRESHOLD,
        DEFAULT_DETECTOR_BACKEND,
        DEFAULT_MODEL_NAME,
        generate_embedding,
        save_user_embeddings,
    )
    from capture import close_windows, open_webcam, read_frame, release_webcam, save_frame
    from logging_config import log_registration_event
    from pad_engine import PadOutcome, check_single_image_pad
    from utils import DATA_DIR, clean_name, ensure_directory, ensure_project_folders
    from utils import print_status


# We collect several images so the system has more than one view of the user.
DEFAULT_IMAGE_COUNT = REGISTRATION_SETTINGS.image_count

# A registration with too few valid face embeddings is unreliable.
MIN_VALID_EMBEDDINGS = REGISTRATION_SETTINGS.minimum_embeddings

# Window title used only during the registration capture session.
REGISTRATION_WINDOW_NAME = REGISTRATION_SETTINGS.window_name

# OpenCV camera index. Usually 0 is the default webcam.
DEFAULT_CAMERA_INDEX = WEBCAM_SETTINGS.camera_index


def ask_user_name() -> str:
    """
    Ask the user for a valid registration name.

    Returns:
        The raw user name entered by the user.
    """
    while True:
        # input() pauses the program and waits for keyboard text in the terminal.
        user_name = input("Enter user name: ").strip()

        # Empty names are not useful for folders or encoding files.
        if not user_name:
            print_status("User name cannot be empty.", level="warning")
            continue

        # clean_name converts text into a safe filename/folder name.
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

    Args:
        user_name: Name of the user being registered.

    Returns:
        Path to data/<safe_user_name>/.
    """
    # Convert the name into something safe for folder names.
    safe_user_name = clean_name(user_name)

    # Each user gets their own folder under data/.
    user_folder = DATA_DIR / safe_user_name

    # Create the folder if it does not already exist.
    return ensure_directory(user_folder)


def capture_registration_images(
    user_name: str,
    image_count: int = DEFAULT_IMAGE_COUNT,
    camera_index: int = DEFAULT_CAMERA_INDEX,
) -> tuple[list[Path], bool]:
    """
    Capture multiple images for one user.

    Keyboard controls:
        s -> save current frame
        q -> cancel registration

    Args:
        user_name: Name of the user being registered.
        image_count: Number of images to capture.
        camera_index: Camera number. Usually 0 is the default webcam.

    Returns:
        A tuple:
            captured_paths: list of saved image paths
            cancelled: True if the user cancelled or capture failed
    """
    # Store paths of images saved during this registration.
    captured_paths: list[Path] = []

    # Create the user's image folder before opening the camera.
    user_folder = get_user_data_folder(user_name)

    # Use the safe user name as the filename prefix.
    filename_prefix = clean_name(user_name)

    # cap starts as None so the finally block can safely release it.
    cap: cv2.VideoCapture | None = None

    try:
        # Open the webcam using the capture module.
        cap = open_webcam(camera_index)

        print_status("Webcam opened for registration.")
        print_status("Press 's' to capture an image, or 'q' to cancel.")

        # Keep capturing until we have the requested number of images.
        while len(captured_paths) < image_count:
            # Read one live frame from the webcam.
            frame = read_frame(cap)

            # Display the live webcam frame.
            cv2.imshow(REGISTRATION_WINDOW_NAME, frame)

            # waitKey refreshes the window and checks for keyboard input.
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if PAD_SETTINGS.enforce_on_registration:
                    pad_result = check_single_image_pad(frame=frame)
                    log_registration_event(
                        "registration_pad_checked",
                        outcome=pad_result.outcome.value,
                        reason_code=pad_result.reason_code,
                    )
                    if pad_result.outcome != PadOutcome.LIVE:
                        print_status(
                            "Registration image rejected: liveness check failed. "
                            "Use a live face, not a photo or screen.",
                            level="warning",
                        )
                        continue

                image_path = save_frame(
                    frame=frame,
                    output_dir=user_folder,
                    filename_prefix=filename_prefix,
                )

                captured_paths.append(image_path)

                print_status(
                    f"Captured image {len(captured_paths)}/{image_count}: {image_path}",
                    level="success",
                )

            elif key == ord("q"):
                # Cancel registration cleanly without saving embeddings.
                print_status("Registration cancelled by user.", level="warning")
                return captured_paths, True

        print_status("Image capture complete.", level="success")
        return captured_paths, False

    except RuntimeError as error:
        # RuntimeError is used by capture.py for camera/read/save failures.
        print_status(str(error), level="error")
        return captured_paths, True

    finally:
        # Always release the webcam even if the user cancels or an error happens.
        release_webcam(cap)

        # Always close OpenCV windows after registration capture ends.
        close_windows()


def generate_registration_embeddings(image_paths: list[Path]) -> list[list[float]]:
    """
    Generate one face embedding for each valid captured image.

    Args:
        image_paths: Saved registration image paths.

    Returns:
        A list of embedding vectors.
    """
    # Store embeddings generated from valid face images.
    embeddings: list[list[float]] = []

    for image_path in image_paths:
        try:
            # DeepFace.represent() is wrapped inside ai_engine.generate_embedding().
            embedding = generate_embedding(
                image_input=image_path,
                model_name=DEFAULT_MODEL_NAME,
                detector_backend=DEFAULT_DETECTOR_BACKEND,
            )

            embeddings.append(embedding)
            print_status(f"Embedding created for: {image_path.name}", level="success")

        except Exception as error:
            # A bad frame should not crash the full registration attempt.
            print_status(
                f"Skipping {image_path.name}: {error}",
                level="warning",
            )

    return embeddings


def register_user(
    user_name: str | None = None,
    image_count: int = DEFAULT_IMAGE_COUNT,
    camera_index: int = DEFAULT_CAMERA_INDEX,
) -> dict[str, Any]:
    """
    Run the full user registration workflow.

    Args:
        user_name: Optional user name. If None, the function asks in terminal.
        image_count: Number of registration images to capture.
        camera_index: Camera number. Usually 0 is the default webcam.

    Returns:
        A dictionary describing the registration result.
    """
    # Make sure data/ and encodings/ exist before registration starts.
    ensure_project_folders()

    # Ask for a name only if the caller did not provide one.
    final_user_name = user_name.strip() if user_name else ask_user_name()

    # Validate the final name before creating files.
    if not final_user_name or clean_name(final_user_name) == "unknown":
        print_status("Registration failed: invalid user name.", level="error")
        return {"success": False, "reason": "invalid_user_name"}

    print_status(f"Starting registration for: {final_user_name}")

    # Step 1: Capture face images from webcam.
    image_paths, cancelled = capture_registration_images(
        user_name=final_user_name,
        image_count=image_count,
        camera_index=camera_index,
    )

    # If the user cancelled, do not generate or save embeddings.
    if cancelled:
        return {
            "success": False,
            "reason": "cancelled",
            "image_count": len(image_paths),
            "image_paths": image_paths,
        }

    # If no images were captured, registration cannot continue.
    if len(image_paths) == 0:
        print_status("Registration failed: no images were captured.", level="error")
        return {"success": False, "reason": "no_images"}

    print_status("Generating face embeddings from captured images.")

    # Step 2: Convert captured images into embeddings.
    embeddings = generate_registration_embeddings(image_paths)

    # Reject weak registrations with too few valid face embeddings.
    if len(embeddings) < MIN_VALID_EMBEDDINGS:
        print_status(
            f"Registration failed: only {len(embeddings)} valid embeddings found. "
            f"At least {MIN_VALID_EMBEDDINGS} are required.",
            level="error",
        )
        return {
            "success": False,
            "reason": "not_enough_valid_embeddings",
            "image_count": len(image_paths),
            "embedding_count": len(embeddings),
            "image_paths": image_paths,
        }

    # Step 3: Save embeddings separately from raw images.
    encoding_path = save_user_embeddings(
        user_name=final_user_name,
        embeddings=embeddings,
        image_paths=image_paths,
        model_name=DEFAULT_MODEL_NAME,
        detector_backend=DEFAULT_DETECTOR_BACKEND,
        threshold=DEFAULT_COSINE_THRESHOLD,
    )

    print_status(
        f"Registration complete for {final_user_name}. "
        f"Saved {len(embeddings)} embeddings to {encoding_path}.",
        level="success",
    )

    return {
        "success": True,
        "user_name": final_user_name,
        "image_count": len(image_paths),
        "embedding_count": len(embeddings),
        "image_paths": image_paths,
        "encoding_path": encoding_path,
        "model_name": DEFAULT_MODEL_NAME,
    }


if __name__ == "__main__":
    # Manual test only. This block runs only when this file is executed directly.
    # It does not run when register.py is imported by app.py.
    register_user()
