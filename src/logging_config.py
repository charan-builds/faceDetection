"""
Centralized logging setup for the Face Lock AI project.

This module owns logging configuration only:
- console logging
- rotating file logging
- structured formatting
- reusable module-specific logger access

Feature modules should call get_logger(__name__) and log meaningful events, but
they should not create handlers or configure logging themselves.
"""

from __future__ import annotations

import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

try:
    # Preferred import when running the project from app.py.
    from src.config import LOGGING_SETTINGS, PATH_SETTINGS
except ModuleNotFoundError:
    # Fallback import when running this file directly as: python src/logging_config.py
    from config import LOGGING_SETTINGS, PATH_SETTINGS


APP_LOGGER_NAME = LOGGING_SETTINGS.logger_name
_IS_CONFIGURED = False


def _level_from_name(level_name: str, default: int = logging.INFO) -> int:
    """
    Convert a configured level name into a logging module level value.
    """
    level = logging.getLevelName(level_name.upper())
    return level if isinstance(level, int) else default


def _minimum_configured_level() -> int:
    """
    Return the most verbose configured level needed by any handler.
    """
    return min(
        _level_from_name(LOGGING_SETTINGS.level),
        _level_from_name(LOGGING_SETTINGS.console_level),
        _level_from_name(LOGGING_SETTINGS.file_level, logging.DEBUG),
    )


def _build_formatter() -> logging.Formatter:
    """
    Create the formatter shared by console and file handlers.
    """
    return logging.Formatter(
        fmt=LOGGING_SETTINGS.message_format,
        datefmt=LOGGING_SETTINGS.date_format,
    )


def _create_console_handler(formatter: logging.Formatter) -> logging.Handler:
    """
    Create a console handler for developer-facing terminal output.
    """
    handler = logging.StreamHandler()
    handler.setLevel(_level_from_name(LOGGING_SETTINGS.console_level))
    handler.setFormatter(formatter)
    return handler


def _create_file_handler(formatter: logging.Formatter) -> RotatingFileHandler:
    """
    Create a rotating file handler for persistent application logs.
    """
    log_dir = PATH_SETTINGS.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / LOGGING_SETTINGS.file_name

    handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=LOGGING_SETTINGS.max_bytes,
        backupCount=LOGGING_SETTINGS.backup_count,
        encoding=LOGGING_SETTINGS.encoding,
    )
    handler.setLevel(_level_from_name(LOGGING_SETTINGS.file_level, logging.DEBUG))
    handler.setFormatter(formatter)
    return handler


def setup_logging(force: bool = False) -> Logger:
    """
    Configure the application logger once and return it.

    Args:
        force: If True, remove and rebuild handlers. Useful for tests.

    Returns:
        The root Face Lock application logger.
    """
    global _IS_CONFIGURED

    logger = logging.getLogger(APP_LOGGER_NAME)

    if _IS_CONFIGURED and not force:
        return logger

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
    elif logger.handlers:
        _IS_CONFIGURED = True
        return logger

    logger.setLevel(_minimum_configured_level())
    logger.propagate = False

    formatter = _build_formatter()
    logger.addHandler(_create_console_handler(formatter))
    logger.addHandler(_create_file_handler(formatter))

    _IS_CONFIGURED = True
    logger.debug(
        "Logging configured log_file=%s max_bytes=%s backup_count=%s",
        Path(PATH_SETTINGS.logs_dir) / LOGGING_SETTINGS.file_name,
        LOGGING_SETTINGS.max_bytes,
        LOGGING_SETTINGS.backup_count,
    )
    return logger


def get_logger(module_name: str | None = None) -> Logger:
    """
    Return a reusable logger for a module.

    Args:
        module_name: Usually __name__ from the caller.

    Returns:
        A configured logger under the Face Lock logger namespace.
    """
    setup_logging()

    if not module_name:
        return logging.getLogger(APP_LOGGER_NAME)

    cleaned_name = module_name.removeprefix("src.")
    return logging.getLogger(f"{APP_LOGGER_NAME}.{cleaned_name}")


def log_event(
    logger: Logger,
    event_name: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """
    Log a structured application event using key=value fields.
    """
    field_text = " ".join(f"{key}={_format_field_value(value)}" for key, value in fields.items())
    message = f"event={event_name}"

    if field_text:
        message = f"{message} {field_text}"

    logger.log(level, message)


def log_exception(logger: Logger, event_name: str, error: Exception, **fields: Any) -> None:
    """
    Log an exception with traceback and structured context.
    """
    field_text = " ".join(f"{key}={_format_field_value(value)}" for key, value in fields.items())
    message = (
        f"event={event_name} "
        f"error_type={type(error).__name__} "
        f"error={_format_field_value(error)}"
    )

    if field_text:
        message = f"{message} {field_text}"

    logger.exception(message)


def log_registration_event(
    event_name: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """
    Convenience helper for registration workflow audit events.
    """
    log_event(get_logger("registration"), event_name, level=level, **fields)


def log_recognition_event(
    event_name: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """
    Convenience helper for recognition workflow audit events.
    """
    log_event(get_logger("recognition"), event_name, level=level, **fields)


def _format_field_value(value: Any) -> str:
    """
    Format structured log values so spaces do not break key=value scanning.
    """
    text = str(value)

    if text == "" or any(character.isspace() for character in text):
        return repr(text)

    return text


if __name__ == "__main__":
    demo_logger = get_logger(__name__)
    log_event(demo_logger, "logging_demo_started", level=logging.INFO)
