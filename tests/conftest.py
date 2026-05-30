"""
Shared pytest fixtures for the Face Lock AI project.

Mocking strategy (summary):
- Webcam: patch ``capture.open_webcam`` / ``read_frame``; return synthetic frames (NumPy).
- DeepFace: patch ``ai_engine._get_deepface`` or ``generate_embedding`` at the call site.
- Filesystem: prefer ``tmp_path`` for pickles and directories; avoid writing to real data/.
- Logging: reset ``logging_config._IS_CONFIGURED`` and handlers between tests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.config import PathSettings
from src import logging_config as logging_config_module


@pytest.fixture
def isolated_path_settings(tmp_path: Path) -> PathSettings:
    """
    Path settings rooted in a temporary directory for tests that touch logs/files.
    """
    return PathSettings(project_root=tmp_path)


@pytest.fixture
def patch_path_settings(monkeypatch: pytest.MonkeyPatch, isolated_path_settings: PathSettings):
    """
    Point logging (and any module that reads PATH_SETTINGS at runtime) at tmp_path.
    """
    monkeypatch.setattr(logging_config_module, "PATH_SETTINGS", isolated_path_settings)
    return isolated_path_settings


@pytest.fixture(autouse=True)
def reset_face_lock_logging() -> None:
    """
    Prevent handler duplication and cross-test log pollution.
    """
    logging_config_module._IS_CONFIGURED = False

    app_logger = logging.getLogger(logging_config_module.APP_LOGGER_NAME)
    app_logger.handlers.clear()
    app_logger.setLevel(logging.NOTSET)
    app_logger.propagate = False

    yield

    logging_config_module._IS_CONFIGURED = False
    for handler in list(app_logger.handlers):
        app_logger.removeHandler(handler)
        handler.close()
    app_logger.handlers.clear()


@pytest.fixture
def configured_logger(patch_path_settings, isolated_path_settings: PathSettings):
    """
    Return the application logger after a forced setup into tmp_path/logs.
    """
    logger = logging_config_module.setup_logging(force=True)
    return logger
