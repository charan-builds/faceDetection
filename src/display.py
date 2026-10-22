"""
OpenCV display helpers for the Face Lock AI project.

This module owns presentation concerns only:
- drawing readable text
- drawing recognition status overlays
- choosing UI colors
- showing OpenCV frames
- reading display-window key input
- preparing optional face bounding boxes

It does not generate embeddings, compare faces, or make access decisions.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2

try:
    # Preferred imports when running the project from app.py.
    from src.config import (
        AI_SETTINGS,
        DISPLAY_SETTINGS,
        RECOGNITION_SETTINGS,
        WEBCAM_SETTINGS,
    )
    from src.logging_config import get_logger, log_event
    from src.pad_engine import ChallengeState, PadChallenge
except ModuleNotFoundError:
    # Fallback imports when running this file directly as: python src/display.py
    from config import AI_SETTINGS, DISPLAY_SETTINGS, RECOGNITION_SETTINGS, WEBCAM_SETTINGS
    from logging_config import get_logger, log_event
    from pad_engine import ChallengeState, PadChallenge


BgrColor = tuple[int, int, int]

LOGGER = get_logger(__name__)

RECOGNITION_WINDOW_NAME = RECOGNITION_SETTINGS.window_name

COLOR_GRANTED = DISPLAY_SETTINGS.color_granted
COLOR_DENIED = DISPLAY_SETTINGS.color_denied
COLOR_INFO = DISPLAY_SETTINGS.color_info
COLOR_WARNING = DISPLAY_SETTINGS.color_warning
COLOR_SHADOW = DISPLAY_SETTINGS.color_shadow


def get_status_color(status: dict[str, Any]) -> BgrColor:
    """
    Choose a display color from a recognition status dictionary.
    Supports explicit states: VERIFIED, DETECTING FACE, CHECKING LIVENESS,
    VERIFYING IDENTITY, NOT VERIFIED, TRY AGAIN, as well as legacy
    ACCESS GRANTED, ACCESS DENIED, and WAITING.
    """
    if status.get("access_granted"):
        return COLOR_GRANTED

    state = str(status.get("state", "")).upper()
    primary_text = str(status.get("primary_text", "")).upper()
    effective = state or primary_text

    if effective in ("VERIFIED",):
        return COLOR_GRANTED

    if effective in ("WAITING", "DETECTING FACE"):
        return COLOR_INFO

    if (
        effective in ("CHECKING LIVENESS", "VERIFYING IDENTITY", "TRY AGAIN")
        or "LIVENESS VERIFIED" in effective
    ):
        return COLOR_WARNING

    return COLOR_DENIED


def resolve_display_state(status: dict[str, Any]) -> str:
    """
    Resolve the explicit UI state label for the live preview.
    States: DETECTING FACE, CHECKING LIVENESS, VERIFYING IDENTITY,
            VERIFIED, NOT VERIFIED, TRY AGAIN.
    """
    state = status.get("state")
    if state:
        state_str = str(state).upper()
        if state_str == "VERIFIED":
            if status.get("access_granted") and status.get("liveness_passed") is True:
                return "VERIFIED"
            return "NOT VERIFIED"
        return state_str

    if status.get("access_granted") and status.get("liveness_passed") is True:
        return "VERIFIED"

    if "LIVENESS VERIFIED" in primary or "LIVENESS VERIFIED" in str(state or "").upper():
        return "LIVENESS VERIFIED — LOOK AT CAMERA"

    if primary == "WAITING":
        if "liveness" in secondary or "challenge" in secondary:
            return "CHECKING LIVENESS"
        if "identity" in secondary or "matching" in secondary:
            return "VERIFYING IDENTITY"
        return "DETECTING FACE"

    if primary in ("ACCESS GRANTED", "VERIFIED"):
        if status.get("liveness_passed") is True:
            return "VERIFIED"
        return "NOT VERIFIED"

    if primary in ("ACCESS DENIED", "NOT VERIFIED"):
        if "try again" in secondary or "timeout" in secondary or "inconclusive" in secondary:
            return "TRY AGAIN"
        return "NOT VERIFIED"

    return primary or "DETECTING FACE"


def draw_text(
    frame: Any,
    text: str,
    position: tuple[int, int],
    color: BgrColor,
    scale: float = 0.8,
    thickness: int = 2,
) -> None:
    """
    Draw readable text with a shadow on an OpenCV frame.
    """
    # Draw a black shadow first so text stays readable on bright backgrounds.
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        COLOR_SHADOW,
        thickness + 2,
        cv2.LINE_AA,
    )

    # Draw the real colored text on top of the shadow.
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_warning(
    frame: Any,
    text: str,
    line_number: int = 0,
) -> None:
    """
    Draw a warning line near the top of the frame.
    """
    y_position = DISPLAY_SETTINGS.primary_text_y + (line_number * 32)
    draw_text(
        frame=frame,
        text=text,
        position=(DISPLAY_SETTINGS.text_margin_x, y_position),
        color=COLOR_WARNING,
        scale=DISPLAY_SETTINGS.secondary_scale,
        thickness=2,
    )


_CHALLENGE_PROMPTS = {
    "turn_head_left": "Turn your head slowly to the left",
    "turn_head_right": "Turn your head slowly to the right",
    "challenge_prompt": "Complete the liveness check",
    "challenge_timeout": "Liveness timed out — try again",
}


def get_challenge_prompt_text(instruction_key: str) -> str:
    """
    Map a PAD instruction key to user-visible text.
    """
    return _CHALLENGE_PROMPTS.get(instruction_key, "Verifying liveness...")


def draw_pad_hint(frame: Any, challenge: PadChallenge | None) -> None:
    """
    Draw the active PAD challenge instruction on a live frame.
    """
    if frame is None or challenge is None:
        return
    if getattr(challenge, "state", None) != ChallengeState.IN_PROGRESS:
        return

    text = get_challenge_prompt_text(challenge.instruction_key)
    draw_warning(frame=frame, text=text, line_number=2)


def build_status_footer(frame_skip: int, cooldown_seconds: float) -> str:
    """
    Build the footer text that describes model and recognition cadence.
    """
    return (
        f"Model: {AI_SETTINGS.model_name} | "
        f"every {frame_skip} frames | cooldown {cooldown_seconds:.1f}s"
    )


def draw_status_overlay(
    frame: Any,
    status: dict[str, Any],
    frame_skip: int,
    cooldown_seconds: float,
    face_box: tuple[int, int, int, int] | None = None,
) -> None:
    """
    Draw recognition status text and optional face bounding box on a live webcam frame.
    """
    if frame is None or not hasattr(frame, "shape"):
        log_event(
            LOGGER,
            "display_overlay_skipped",
            level=logging.WARNING,
            reason="invalid_frame",
        )
        return

    margin_x = DISPLAY_SETTINGS.text_margin_x
    color = get_status_color(status)

    # Draw face bounding box when detected
    if face_box is not None:
        draw_bounding_box(frame, face_box, color=color, thickness=2)

    # Prefer explicit state if provided, otherwise fallback to primary_text
    display_title = str(status.get("state") or status.get("primary_text", "WAITING"))
    # Fail-closed guard: only allow VERIFIED if both liveness and access granted
    if display_title == "VERIFIED" and not (status.get("access_granted") and status.get("liveness_passed") is True):
        display_title = "NOT VERIFIED"
        color = COLOR_DENIED

    draw_text(
        frame=frame,
        text=display_title,
        position=(margin_x, DISPLAY_SETTINGS.primary_text_y),
        color=color,
        scale=DISPLAY_SETTINGS.primary_scale,
        thickness=2,
    )

    draw_text(
        frame=frame,
        text=str(status.get("secondary_text", "")),
        position=(margin_x, DISPLAY_SETTINGS.secondary_text_y),
        color=COLOR_INFO,
        scale=DISPLAY_SETTINGS.secondary_scale,
        thickness=2,
    )

    if status.get("detail_text"):
        draw_text(
            frame=frame,
            text=str(status["detail_text"]),
            position=(margin_x, DISPLAY_SETTINGS.detail_text_y),
            color=COLOR_INFO,
            scale=DISPLAY_SETTINGS.detail_scale,
            thickness=1,
        )

    liveness_passed = status.get("liveness_passed")
    if liveness_passed is True:
        draw_text(
            frame=frame,
            text="Liveness: verified",
            position=(margin_x, DISPLAY_SETTINGS.detail_text_y + 24),
            color=COLOR_GRANTED,
            scale=DISPLAY_SETTINGS.detail_scale,
            thickness=1,
        )
    elif liveness_passed is False:
        draw_text(
            frame=frame,
            text="Liveness: failed",
            position=(margin_x, DISPLAY_SETTINGS.detail_text_y + 24),
            color=COLOR_DENIED,
            scale=DISPLAY_SETTINGS.detail_scale,
            thickness=1,
        )

    draw_text(
        frame=frame,
        text=build_status_footer(
            frame_skip=frame_skip,
            cooldown_seconds=cooldown_seconds,
        ),
        position=(margin_x, frame.shape[0] - DISPLAY_SETTINGS.footer_offset_y),
        color=COLOR_WARNING,
        scale=DISPLAY_SETTINGS.footer_scale,
        thickness=1,
    )

    draw_text(
        frame=frame,
        text=f"Press {WEBCAM_SETTINGS.quit_key} to quit",
        position=(margin_x, frame.shape[0] - DISPLAY_SETTINGS.quit_offset_y),
        color=COLOR_WARNING,
        scale=DISPLAY_SETTINGS.footer_scale,
        thickness=1,
    )


def prepare_bounding_box(
    facial_area: dict[str, Any] | None,
) -> tuple[int, int, int, int] | None:
    """
    Convert a DeepFace facial_area dictionary into an OpenCV rectangle tuple.

    Returns:
        (x, y, width, height), or None when the area is unavailable.
    """
    if not facial_area:
        return None

    try:
        x = int(facial_area["x"])
        y = int(facial_area["y"])
        width = int(facial_area["w"])
        height = int(facial_area["h"])
    except (KeyError, TypeError, ValueError):
        return None

    if width <= 0 or height <= 0:
        return None

    return x, y, width, height


def draw_bounding_box(
    frame: Any,
    box: tuple[int, int, int, int] | None,
    color: BgrColor = COLOR_WARNING,
    thickness: int = 2,
) -> None:
    """
    Draw a prepared bounding box on a frame when one is available.
    """
    if box is None:
        return

    x, y, width, height = box
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)


def show_frame(window_name: str, frame: Any) -> None:
    """
    Show one frame in an OpenCV window.
    """
    cv2.imshow(window_name, frame)


def read_display_key(delay_ms: int = 1) -> int:
    """
    Read a keyboard key from the active OpenCV display window.
    """
    return cv2.waitKey(delay_ms) & 0xFF


def is_quit_key(key: int, quit_key: str = WEBCAM_SETTINGS.quit_key) -> bool:
    """
    Check whether an OpenCV key code matches the configured quit key.
    """
    if not quit_key:
        return False

    return key == ord(quit_key[0])
