"""
Shared utility functions for the Face Lock AI project.

This file contains only general helper code:
- folder creation
- path building
- pickle saving/loading
- file/folder checks
- image filename generation
- clean status messages

AI logic, webcam logic, and recognition decisions should stay in their own modules.
"""

from __future__ import annotations

import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    # Preferred import when running the project from app.py.
    from src.config import LOGGING_SETTINGS, PATH_SETTINGS, STORAGE_SETTINGS
except ModuleNotFoundError:
    # Fallback import when running this file directly as: python src/utils.py
    from config import LOGGING_SETTINGS, PATH_SETTINGS, STORAGE_SETTINGS


# Path settings are centralized in src/config.py.
PROJECT_ROOT = PATH_SETTINGS.project_root

# Common project folders used by other modules.
DATA_DIR = PATH_SETTINGS.data_dir
ENCODINGS_DIR = PATH_SETTINGS.encodings_dir
LOGS_DIR = PATH_SETTINGS.logs_dir
MODELS_DIR = PATH_SETTINGS.models_dir

# File format settings used by image and embedding helpers.
DEFAULT_IMAGE_EXTENSION = STORAGE_SETTINGS.default_image_extension
SUPPORTED_IMAGE_EXTENSIONS = STORAGE_SETTINGS.supported_image_extensions
LOGGING_LEVEL = LOGGING_SETTINGS.level

# Internal marker used to detect whether a default value was provided.
_MISSING = object()


def ensure_directory(directory_path: str | Path) -> Path:
    """
    Create a directory safely if it does not already exist.

    Args:
        directory_path: Folder path as a string or Path object.

    Returns:
        The folder path as a Path object.
    """
    # Convert strings into Path objects so the rest of the code is consistent.
    path = Path(directory_path)

    # parents=True creates missing parent folders, exist_ok=True avoids errors
    # when the folder already exists.
    path.mkdir(parents=True, exist_ok=True)

    # Returning the path makes this function easy to reuse in other files.
    return path


def ensure_project_folders() -> None:
    """
    Create the core folders required by the Face Lock project.
    """
    for directory in PATH_SETTINGS.runtime_directories:
        ensure_directory(directory)


def build_path(*parts: str | Path) -> Path:
    """
    Build a path inside the project root.

    Example:
        build_path("data", "charan", "image_001.jpg")

    Args:
        *parts: Folder/file names that should be joined together.

    Returns:
        A Path object pointing inside the project folder.
    """
    # joinpath combines path parts using the correct separator for the OS.
    return PROJECT_ROOT.joinpath(*parts)


def file_exists(file_path: str | Path) -> bool:
    """
    Check whether a path exists and is a file.

    Args:
        file_path: Path to check.

    Returns:
        True if the path exists and is a file, otherwise False.
    """
    return Path(file_path).is_file()


def directory_exists(directory_path: str | Path) -> bool:
    """
    Check whether a path exists and is a directory.

    Args:
        directory_path: Path to check.

    Returns:
        True if the path exists and is a folder, otherwise False.
    """
    return Path(directory_path).is_dir()


def save_pickle(data: Any, file_path: str | Path) -> Path:
    """
    Save Python data to a pickle file.

    Pickle is useful for ML projects because it can store Python objects such as
    lists, dictionaries, NumPy arrays, and face embeddings.

    Args:
        data: Python object to save.
        file_path: Destination .pkl file path.

    Returns:
        The saved file path as a Path object.
    """
    # Convert the target path into a Path object.
    path = Path(file_path)

    # Make sure the parent folder exists before writing the file.
    ensure_directory(path.parent)

    # "wb" means write binary, which is required for pickle files.
    with path.open("wb") as file:
        pickle.dump(data, file)

    return path


def load_pickle(file_path: str | Path, default: Any = _MISSING) -> Any:
    """
    Load Python data from a pickle file.

    Args:
        file_path: Source .pkl file path.
        default: Optional value to return if the file does not exist.

    Returns:
        The loaded Python object, or the default value if provided.

    Raises:
        FileNotFoundError: If the file is missing and no default is provided.
    """
    # Convert the source path into a Path object.
    path = Path(file_path)

    # Handle missing files clearly.
    if not path.is_file():
        if default is not _MISSING:
            return default
        raise FileNotFoundError(f"Pickle file not found: {path}")

    # "rb" means read binary, which is required for pickle files.
    with path.open("rb") as file:
        return pickle.load(file)


def clean_name(value: str) -> str:
    """
    Convert user-provided text into a safe filename-friendly value.

    Args:
        value: Raw text, such as a username.

    Returns:
        A lowercase string safe to use in file and folder names.
    """
    # Remove extra spaces and make the name lowercase for consistency.
    cleaned = value.strip().lower()

    # Replace spaces with underscores.
    cleaned = cleaned.replace(" ", "_")

    # Keep only letters, numbers, underscores, and hyphens.
    cleaned = re.sub(r"[^a-z0-9_-]", "", cleaned)

    # Use a safe fallback if the input had no valid characters.
    return cleaned or "unknown"


def generate_image_filename(
    username: str,
    extension: str = DEFAULT_IMAGE_EXTENSION,
) -> str:
    """
    Generate a clean, unique image filename.

    Example:
        charan_20260527_153045_123456.jpg

    Args:
        username: Name of the registered user.
        extension: Image extension without a dot, such as jpg or png.

    Returns:
        A safe image filename.
    """
    # Clean the username so it is safe for filenames.
    safe_username = clean_name(username)

    # Clean the extension and remove a leading dot if the caller provides one.
    safe_extension = extension.strip().lower().lstrip(".")

    # Fall back to the configured default if the extension is empty.
    safe_extension = safe_extension or DEFAULT_IMAGE_EXTENSION

    # Timestamp makes the filename unique and sortable.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    return f"{safe_username}_{timestamp}.{safe_extension}"


def print_status(message: str, level: str = "info") -> None:
    """
    Print a clean status message for command-line output.

    Args:
        message: Text to show to the user.
        level: Message type: info, success, warning, or error.
    """
    # Map friendly level names to short labels.
    labels = {
        "info": "INFO",
        "success": "OK",
        "warning": "WARNING",
        "error": "ERROR",
    }

    # LOGGING_LEVEL is reserved for the future logging module setup.

    # Use INFO if an unknown level is passed.
    label = labels.get(level.lower(), "INFO")

    print(f"[{label}] {message}")
