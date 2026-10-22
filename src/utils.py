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

import io
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    # Preferred import when running the project from app.py.
    from src.config import (
        DEFAULT_OWNER_NAME,
        ENV_OWNER_NAME,
        LOGGING_SETTINGS,
        PATH_SETTINGS,
        STORAGE_SETTINGS,
        resolve_owner_name,
    )
except ModuleNotFoundError:
    # Fallback import when running this file directly as: python src/utils.py
    from config import (
        DEFAULT_OWNER_NAME,
        ENV_OWNER_NAME,
        LOGGING_SETTINGS,
        PATH_SETTINGS,
        STORAGE_SETTINGS,
        resolve_owner_name,
    )


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
# Re-export for modules that import level from utils (canonical: LOGGING_SETTINGS).
LOGGING_LEVEL = LOGGING_SETTINGS.level

# Internal marker used to detect whether a default value was provided.
_MISSING = object()


# ---------------------------------------------------------------------------
# Secure pickle deserialization (SEC-002)
# ---------------------------------------------------------------------------

# Exhaustive set of (module, name) pairs that the safe unpickler allows.
# Face Lock .pkl files contain only plain Python primitives and lists of floats.
_SAFE_TYPES: frozenset[tuple[str, str]] = frozenset(
    {
        # Built-in containers
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "tuple"),
        # Built-in scalars
        ("builtins", "str"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "bool"),
        # datetime.datetime is included because embedding records store
        # an ISO-formatted creation timestamp via datetime.now().isoformat().
        # The value is serialised as a plain str, but older pickle files may
        # have serialised the datetime object itself.
        ("datetime", "datetime"),
    }
)


class RestrictedUnpicklingError(ValueError):
    """
    Raised when a pickle stream tries to instantiate a type that is not on
    the explicit allow-list.

    Inherits from ValueError so callers that catch ValueError (e.g. schema
    validation in ai_engine) continue to behave correctly.
    """


class _SafeUnpickler(pickle.Unpickler):
    """
    A restricted ``pickle.Unpickler`` that only allows pure-data types.

    Any attempt to deserialise an object whose ``(module, name)`` pair is not
    in ``_SAFE_TYPES`` raises ``RestrictedUnpicklingError`` immediately, before
    any code is executed.  This blocks every known pickle RCE gadget (os.system,
    subprocess, reduce-based gadgets, importlib tricks, etc.).

    Permitted types
    ---------------
    - ``builtins``: dict, list, tuple, str, int, float, bool
    - ``datetime``: datetime  (legacy compatibility)

    Blocked examples
    ----------------
    - ``os.system``, ``subprocess.Popen``  → arbitrary command execution
    - ``__builtin__.eval``                 → code injection
    - Any user-defined / third-party class → import-based gadgets
    """

    def find_class(self, module: str, name: str) -> Any:
        """
        Intercept every GLOBAL / STACK_GLOBAL opcode in the pickle stream.

        Called once for each class reference encountered during unpickling.
        Raises ``RestrictedUnpicklingError`` if the type is not explicitly
        allowed.
        """
        if (module, name) in _SAFE_TYPES:
            # Delegate to the standard resolver only for permitted types.
            return super().find_class(module, name)

        raise RestrictedUnpicklingError(
            f"Pickle load blocked: type '{module}.{name}' is not permitted. "
            "Only plain Python data types are allowed in Face Lock embedding files."
        )


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

    Security:
        Only load pickles created by this application from a trusted ``encodings/``
        directory. Untrusted pickle files can execute arbitrary code on load.

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
    Load Python data from a pickle file using a restricted unpickler.

    Security (SEC-002):
        Only plain Python data types are permitted during deserialization:
        dict, list, tuple, str, int, float, bool, and datetime.datetime.
        Any attempt to load a class outside this allow-list raises
        ``RestrictedUnpicklingError``, preventing arbitrary code execution
        even when a malicious ``.pkl`` file is placed in ``encodings/``.

    Args:
        file_path: Source .pkl file path.
        default: Optional value to return if the file does not exist.

    Returns:
        The loaded Python object, or the default value if provided.

    Raises:
        FileNotFoundError: If the file is missing and no default is provided.
        RestrictedUnpicklingError: If the pickle stream references a blocked type.
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
        # _SafeUnpickler replaces the unrestricted pickle.load() call.
        # It raises RestrictedUnpicklingError for any non-allowed type,
        # which prevents RCE gadgets from executing during deserialization.
        return _SafeUnpickler(file).load()


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

    # Use INFO if an unknown level is passed.
    label = labels.get(level.lower(), "INFO")

    print(f"[{label}] {message}")
