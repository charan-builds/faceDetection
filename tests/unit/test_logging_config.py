"""
Unit tests for src.logging_config.

Uses isolated tmp_path logs via patch_path_settings; resets handlers via conftest.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from src import logging_config as lc


@pytest.mark.unit
class TestSetupLogging:
    def test_setup_logging_creates_console_and_file_handlers(
        self, configured_logger: logging.Logger, isolated_path_settings
    ) -> None:
        handler_types = {type(handler) for handler in configured_logger.handlers}

        assert logging.StreamHandler in handler_types
        assert RotatingFileHandler in handler_types

    def test_setup_logging_writes_log_file_under_project_logs_dir(
        self, configured_logger: logging.Logger, isolated_path_settings
    ) -> None:
        log_path = isolated_path_settings.logs_dir / lc.LOGGING_SETTINGS.file_name

        configured_logger.info("test log line")

        for handler in configured_logger.handlers:
            handler.flush()

        assert log_path.is_file()
        assert "test log line" in log_path.read_text(encoding="utf-8")

    def test_setup_logging_force_rebuilds_handlers_without_duplication(
        self, patch_path_settings
    ) -> None:
        first = lc.setup_logging(force=True)
        first_handler_count = len(first.handlers)

        second = lc.setup_logging(force=True)

        assert second is first
        assert len(second.handlers) == first_handler_count


@pytest.mark.unit
class TestGetLogger:
    def test_get_logger_strips_src_prefix(self, patch_path_settings) -> None:
        logger = lc.get_logger("src.recognize")

        assert logger.name == f"{lc.APP_LOGGER_NAME}.recognize"

    def test_get_logger_returns_app_logger_when_name_missing(
        self, patch_path_settings
    ) -> None:
        logger = lc.get_logger(None)

        assert logger.name == lc.APP_LOGGER_NAME


@pytest.fixture
def capture_app_logs(
    configured_logger: logging.Logger, caplog: pytest.LogCaptureFixture
) -> pytest.LogCaptureFixture:
    """
    Attach caplog to the non-propagating Face Lock logger.
    """
    configured_logger.addHandler(caplog.handler)
    configured_logger.setLevel(logging.DEBUG)
    return caplog


@pytest.mark.unit
class TestStructuredLogging:
    def test_log_event_emits_key_value_message(
        self, configured_logger: logging.Logger, capture_app_logs: pytest.LogCaptureFixture
    ) -> None:
        lc.log_event(configured_logger, "unit_test_event", user_name="alice", count=2)

        assert "event=unit_test_event" in capture_app_logs.text
        assert "user_name=alice" in capture_app_logs.text
        assert "count=2" in capture_app_logs.text

    def test_log_event_quotes_values_with_whitespace(
        self, configured_logger: logging.Logger, capture_app_logs: pytest.LogCaptureFixture
    ) -> None:
        lc.log_event(configured_logger, "spacing_test", detail="bad lighting")

        assert "detail='bad lighting'" in capture_app_logs.text

    def test_log_exception_includes_error_context(
        self, configured_logger: logging.Logger, capture_app_logs: pytest.LogCaptureFixture
    ) -> None:
        try:
            raise ValueError("camera unavailable")
        except ValueError as error:
            lc.log_exception(configured_logger, "unit_test_failure", error, camera_index=0)

        assert "event=unit_test_failure" in capture_app_logs.text
        assert "error_type=ValueError" in capture_app_logs.text
        assert "camera_index=0" in capture_app_logs.text
