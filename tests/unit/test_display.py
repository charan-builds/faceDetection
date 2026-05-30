"""
Unit tests for presentation helpers in src.display.

No webcam, no cv2.imshow / waitKey, and no DeepFace.
Frames are in-memory NumPy arrays; window calls are mocked when needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.config import AI_SETTINGS, DISPLAY_SETTINGS, WEBCAM_SETTINGS
from src.display import (
    COLOR_DENIED,
    COLOR_GRANTED,
    COLOR_INFO,
    COLOR_SHADOW,
    COLOR_WARNING,
    build_status_footer,
    draw_bounding_box,
    draw_pad_hint,
    draw_status_overlay,
    draw_text,
    draw_warning,
    get_challenge_prompt_text,
    get_status_color,
    is_quit_key,
    prepare_bounding_box,
    read_display_key,
    show_frame,
)
from src.pad_engine import ChallengeState, ChallengeType, PadChallenge


@pytest.fixture
def blank_frame() -> np.ndarray:
    """Small BGR frame in memory (not shown in any window)."""
    return np.zeros((120, 160, 3), dtype=np.uint8)


@pytest.fixture
def granted_status() -> dict[str, Any]:
    return {
        "access_granted": True,
        "primary_text": "ACCESS GRANTED",
        "secondary_text": "User: alice",
        "detail_text": "distance=0.000 threshold=0.30",
    }


@pytest.fixture
def denied_status() -> dict[str, Any]:
    return {
        "access_granted": False,
        "primary_text": "ACCESS DENIED",
        "secondary_text": "Closest user: bob",
        "detail_text": "distance=0.850 threshold=0.30",
    }


@pytest.fixture
def waiting_status() -> dict[str, Any]:
    return {
        "access_granted": False,
        "primary_text": "WAITING",
        "secondary_text": "Looking for a registered face",
        "detail_text": "",
    }


def _puttext_texts(mock_puttext: Any) -> list[str]:
    """Return the text argument from each cv2.putText call."""
    return [call.args[1] for call in mock_puttext.call_args_list]


def _puttext_colors(mock_puttext: Any) -> list[tuple[int, int, int]]:
    """Return the color argument from each cv2.putText call."""
    return [call.args[5] for call in mock_puttext.call_args_list]


@pytest.mark.unit
class TestGetStatusColor:
    def test_access_granted_uses_granted_color(self, granted_status: dict[str, Any]) -> None:
        assert get_status_color(granted_status) == COLOR_GRANTED

    def test_access_denied_uses_denied_color(self, denied_status: dict[str, Any]) -> None:
        assert get_status_color(denied_status) == COLOR_DENIED

    def test_waiting_uses_info_color(self, waiting_status: dict[str, Any]) -> None:
        assert get_status_color(waiting_status) == COLOR_INFO

    def test_waiting_is_case_insensitive(self) -> None:
        status = {"access_granted": False, "primary_text": "waiting"}
        assert get_status_color(status) == COLOR_INFO

    def test_unknown_status_falls_back_to_denied_color(self) -> None:
        """Missing or unexpected primary_text should not look like a grant."""
        assert get_status_color({}) == COLOR_DENIED
        assert get_status_color({"primary_text": "PROCESSING"}) == COLOR_DENIED

    def test_granted_wins_over_waiting_primary_text(self) -> None:
        status = {"access_granted": True, "primary_text": "WAITING"}
        assert get_status_color(status) == COLOR_GRANTED


@pytest.mark.unit
class TestBuildStatusFooter:
    def test_footer_includes_model_frame_skip_and_cooldown(self) -> None:
        footer = build_status_footer(frame_skip=10, cooldown_seconds=1.5)

        assert AI_SETTINGS.model_name in footer
        assert "every 10 frames" in footer
        assert "cooldown 1.5s" in footer

    def test_footer_formats_cooldown_to_one_decimal(self) -> None:
        footer = build_status_footer(frame_skip=3, cooldown_seconds=2.0)
        assert "cooldown 2.0s" in footer


@pytest.mark.unit
class TestPrepareBoundingBox:
    def test_valid_facial_area_returns_xywh_tuple(self) -> None:
        box = prepare_bounding_box({"x": 10, "y": 20, "w": 30, "h": 40})
        assert box == (10, 20, 30, 40)

    def test_none_or_empty_input_returns_none(self) -> None:
        assert prepare_bounding_box(None) is None
        assert prepare_bounding_box({}) is None

    def test_missing_keys_returns_none(self) -> None:
        assert prepare_bounding_box({"x": 1, "y": 2}) is None

    def test_non_positive_size_returns_none(self) -> None:
        assert prepare_bounding_box({"x": 0, "y": 0, "w": 0, "h": 10}) is None
        assert prepare_bounding_box({"x": 0, "y": 0, "w": 10, "h": -1}) is None

    def test_invalid_types_returns_none(self) -> None:
        assert prepare_bounding_box({"x": "bad", "y": 0, "w": 10, "h": 10}) is None

    def test_numeric_strings_are_coerced(self) -> None:
        box = prepare_bounding_box({"x": "5", "y": "6", "w": "7", "h": "8"})
        assert box == (5, 6, 7, 8)


@pytest.mark.unit
class TestDrawTextAndWarning:
    def test_draw_text_calls_puttext_twice_for_shadow_and_foreground(
        self, blank_frame: np.ndarray
    ) -> None:
        with patch("src.display.cv2.putText") as mock_puttext:
            draw_text(blank_frame, "HELLO", (10, 20), COLOR_GRANTED)

        assert mock_puttext.call_count == 2
        colors = _puttext_colors(mock_puttext)
        assert colors[0] == COLOR_SHADOW
        assert colors[1] == COLOR_GRANTED

    def test_draw_warning_uses_warning_color(self, blank_frame: np.ndarray) -> None:
        with patch("src.display.draw_text") as mock_draw_text:
            draw_warning(blank_frame, "Low light", line_number=1)

        mock_draw_text.assert_called_once()
        assert mock_draw_text.call_args.kwargs["color"] == COLOR_WARNING
        assert mock_draw_text.call_args.kwargs["text"] == "Low light"
        assert mock_draw_text.call_args.kwargs["position"] == (
            DISPLAY_SETTINGS.text_margin_x,
            DISPLAY_SETTINGS.primary_text_y + 32,
        )


@pytest.mark.unit
class TestDrawStatusOverlay:
    def test_overlay_draws_primary_secondary_detail_and_footer_text(
        self, blank_frame: np.ndarray, denied_status: dict[str, Any]
    ) -> None:
        with patch("src.display.cv2.putText") as mock_puttext:
            draw_status_overlay(
                frame=blank_frame,
                status=denied_status,
                frame_skip=10,
                cooldown_seconds=1.5,
            )

        texts = _puttext_texts(mock_puttext)
        assert "ACCESS DENIED" in texts
        assert denied_status["secondary_text"] in texts
        assert denied_status["detail_text"] in texts
        assert build_status_footer(10, 1.5) in texts
        assert f"Press {WEBCAM_SETTINGS.quit_key} to quit" in texts

    def test_overlay_omits_detail_line_when_detail_text_empty(
        self, blank_frame: np.ndarray, waiting_status: dict[str, Any]
    ) -> None:
        with patch("src.display.cv2.putText") as mock_puttext:
            draw_status_overlay(
                frame=blank_frame,
                status=waiting_status,
                frame_skip=5,
                cooldown_seconds=0.5,
            )

        texts = _puttext_texts(mock_puttext)
        assert waiting_status["secondary_text"] in texts
        assert not any("distance=" in text for text in texts)

    def test_overlay_uses_granted_color_for_primary_when_access_granted(
        self, blank_frame: np.ndarray, granted_status: dict[str, Any]
    ) -> None:
        with patch("src.display.cv2.putText") as mock_puttext:
            draw_status_overlay(
                frame=blank_frame,
                status=granted_status,
                frame_skip=10,
                cooldown_seconds=1.5,
            )

        texts = _puttext_texts(mock_puttext)
        colors = _puttext_colors(mock_puttext)
        primary_index = texts.index("ACCESS GRANTED")
        # Each draw_text issues shadow then foreground; foreground is the next call.
        assert colors[primary_index + 1] == COLOR_GRANTED

    def test_invalid_frame_is_skipped_without_puttext(self, denied_status: dict[str, Any]) -> None:
        with patch("src.display.cv2.putText") as mock_puttext:
            with patch("src.display.log_event") as mock_log_event:
                draw_status_overlay(
                    frame=None,
                    status=denied_status,
                    frame_skip=10,
                    cooldown_seconds=1.5,
                )

        mock_puttext.assert_not_called()
        mock_log_event.assert_called_once()
        assert mock_log_event.call_args.kwargs["reason"] == "invalid_frame"

    def test_frame_without_shape_is_treated_as_invalid(
        self, denied_status: dict[str, Any]
    ) -> None:
        invalid_frame = object()

        with patch("src.display.cv2.putText") as mock_puttext:
            with patch("src.display.log_event"):
                draw_status_overlay(
                    frame=invalid_frame,
                    status=denied_status,
                    frame_skip=10,
                    cooldown_seconds=1.5,
                )

        mock_puttext.assert_not_called()

    def test_missing_primary_text_defaults_to_waiting_label(
        self, blank_frame: np.ndarray
    ) -> None:
        with patch("src.display.cv2.putText") as mock_puttext:
            draw_status_overlay(
                frame=blank_frame,
                status={"secondary_text": "Scanning"},
                frame_skip=1,
                cooldown_seconds=0.0,
            )

        assert "WAITING" in _puttext_texts(mock_puttext)


@pytest.mark.unit
class TestDrawBoundingBox:
    def test_draw_bounding_box_skips_when_box_is_none(self, blank_frame: np.ndarray) -> None:
        with patch("src.display.cv2.rectangle") as mock_rectangle:
            draw_bounding_box(blank_frame, None)

        mock_rectangle.assert_not_called()

    def test_draw_bounding_box_draws_rectangle_for_valid_box(
        self, blank_frame: np.ndarray
    ) -> None:
        with patch("src.display.cv2.rectangle") as mock_rectangle:
            draw_bounding_box(blank_frame, (10, 20, 30, 40), color=COLOR_WARNING)

        mock_rectangle.assert_called_once_with(
            blank_frame,
            (10, 20),
            (40, 60),
            COLOR_WARNING,
            2,
        )


@pytest.mark.unit
class TestWindowHelpers:
    """cv2.imshow / waitKey are mocked — no real windows."""

    def test_show_frame_delegates_to_imshow(self, blank_frame: np.ndarray) -> None:
        with patch("src.display.cv2.imshow") as mock_imshow:
            show_frame("Test Window", blank_frame)

        mock_imshow.assert_called_once_with("Test Window", blank_frame)

    def test_read_display_key_delegates_to_waitkey(self) -> None:
        with patch("src.display.cv2.waitKey", return_value=ord("q")) as mock_wait:
            key = read_display_key(delay_ms=1)

        mock_wait.assert_called_once_with(1)
        assert key == ord("q")

    def test_is_quit_key_matches_configured_key(self) -> None:
        assert is_quit_key(ord("q")) is True
        assert is_quit_key(ord("x")) is False

    def test_is_quit_key_returns_false_for_empty_quit_key(self) -> None:
        assert is_quit_key(ord("q"), quit_key="") is False


@pytest.mark.unit
class TestPadDisplay:
    def test_get_challenge_prompt_text(self) -> None:
        assert "left" in get_challenge_prompt_text("turn_head_left").lower()

    def test_draw_pad_hint_calls_puttext(self, blank_frame: np.ndarray) -> None:
        challenge = PadChallenge(
            challenge_id="c1",
            challenge_type=ChallengeType.HEAD_TURN_LEFT,
            instruction_key="turn_head_left",
            state=ChallengeState.IN_PROGRESS,
            started_at_ms=0.0,
            deadline_ms=1000.0,
        )
        with patch("src.display.cv2.putText") as mock_puttext:
            draw_pad_hint(blank_frame, challenge)
        assert mock_puttext.called


@pytest.mark.unit
class TestInMemoryDrawing:
    """
    Optional sanity check: putText mutates an array without opening a window.
    """

    def test_draw_text_modifies_frame_pixels(self, blank_frame: np.ndarray) -> None:
        before = int(blank_frame.sum())
        draw_text(blank_frame, "X", (15, 25), COLOR_INFO)
        assert int(blank_frame.sum()) > before
