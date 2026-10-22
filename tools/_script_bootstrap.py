"""Bootstrap repository root when a tool is run as ``python tools/<script>.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def ensure_repository_root_on_sys_path(script_file: str | Path) -> Path:
    """
    Execute ``tools/_repo_path.py`` so imports of ``src``, ``tools``, and ``eval`` work.

    Must be called before any ``from tools...`` or ``from src...`` imports.

    Returns:
        Repository root path.
    """
    tools_dir = Path(script_file).resolve().parent
    repo_path_file = tools_dir / "_repo_path.py"
    spec = importlib.util.spec_from_file_location("face_lock_repo_path", repo_path_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load repository path bootstrap: {repo_path_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tools_dir.parent
