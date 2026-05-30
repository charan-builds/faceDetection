"""
Live face recognition workflow for the Face Lock AI project.

This module coordinates recognition only:
- open the webcam
- run PAD (liveness) before identity matching
- generate a live face embedding
- load saved trusted embeddings
- compare embeddings
- display ACCESS GRANTED or ACCESS DENIED

It does not contain registration logic, app menu logic, or DeepFace.verify().
"""

from __future__ import annotations

import logging
import time
from typing import Any

try:
    from src.config import PAD_SETTINGS, RECOGNITION_SETTINGS, WEBCAM_SETTINGS
    from src.ai_engine import (
        DEFAULT_COSINE_THRESHOLD,
        DEFAULT_DETECTOR_BACKEND,
        DEFAULT_MODEL_NAME,
        find_best_match,
        generate_embedding,
        load_all_user_embeddings,
    )
    from src.capture import close_windows, open_webcam, read_frame, release_webcam
    from src.display import (
        RECOGNITION_WINDOW_NAME,
        draw_pad_hint,
        draw_status_overlay,
        is_quit_key,
        read_display_key,
        show_frame,
    )
    from src.logging_config import get_logger, log_event, log_exception
    from src.pad_engine import (
        MSG_CHALLENGE_PROMPT,
        MSG_CHALLENGE_TIMEOUT,
        MSG_LIVENESS_FAILED,
        MSG_LIVENESS_INCONCLUSIVE,
        PadOutcome,
        PadResult,
        PadSession,
        build_pad_frame,
        check_single_image_pad,
        create_pad_session,
    )
    from src.utils import print_status
except ModuleNotFoundError:
    from config import PAD_SETTINGS, RECOGNITION_SETTINGS, WEBCAM_SETTINGS
    from ai_engine import (
        DEFAULT_COSINE_THRESHOLD,
        DEFAULT_DETECTOR_BACKEND,
        DEFAULT_MODEL_NAME,
        find_best_match,
        generate_embedding,
        load_all_user_embeddings,
    )
    from capture import close_windows, open_webcam, read_frame, release_webcam
    from display import (
        RECOGNITION_WINDOW_NAME,
        draw_pad_hint,
        draw_status_overlay,
        is_quit_key,
        read_display_key,
        show_frame,
    )
    from logging_config import get_logger, log_event, log_exception
    from pad_engine import (
        MSG_CHALLENGE_PROMPT,
        MSG_CHALLENGE_TIMEOUT,
        MSG_LIVENESS_FAILED,
        MSG_LIVENESS_INCONCLUSIVE,
        PadOutcome,
        PadResult,
        PadSession,
        build_pad_frame,
        check_single_image_pad,
        create_pad_session,
    )
    from utils import print_status


DEFAULT_FRAME_SKIP = RECOGNITION_SETTINGS.frame_skip
DEFAULT_COOLDOWN_SECONDS = RECOGNITION_SETTINGS.cooldown_seconds
LOW_LIGHT_THRESHOLD = RECOGNITION_SETTINGS.brightness_threshold
DEFAULT_CAMERA_INDEX = WEBCAM_SETTINGS.camera_index

LOGGER = get_logger(__name__)

_PAD_USER_MESSAGES = {
    MSG_LIVENESS_FAILED: "Liveness check failed",
    MSG_LIVENESS_INCONCLUSIVE: "Could not verify liveness",
    MSG_CHALLENGE_PROMPT: "Follow on-screen liveness prompt",
    MSG_CHALLENGE_TIMEOUT: "Liveness challenge timed out",
}


def load_trusted_records() -> list[dict[str, Any]]:
    """
    Load all registered user embeddings from encodings/.
    """
    try:
        records = load_all_user_embeddings()
    except Exception as error:
        log_exception(LOGGER, "trusted_embeddings_load_failed", error)
        print_status(f"Could not load trusted embeddings: {error}", level="error")
        return []

    if len(records) == 0:
        log_event(
            LOGGER,
            "trusted_embeddings_missing",
            level=logging.WARNING,
        )
        print_status("No trusted users found. Register a user first.", level="warning")
    else:
        log_event(
            LOGGER,
            "trusted_embeddings_loaded",
            record_count=len(records),
        )
        print_status(f"Loaded {len(records)} trusted user record(s).", level="success")

    return records


def estimate_brightness(frame: Any) -> float:
    return float(frame.mean())


def is_bad_lighting(frame: Any, threshold: float = LOW_LIGHT_THRESHOLD) -> bool:
    return estimate_brightness(frame) < threshold


def format_distance(distance: float | None) -> str:
    if distance is None:
        return "n/a"
    return f"{distance:.3f}"


def build_denied_result(reason: str, detail: str = "") -> dict[str, Any]:
    return {
        "access_granted": False,
        "primary_text": "ACCESS DENIED",
        "secondary_text": reason,
        "detail_text": detail,
        "user_name": None,
        "cosine_distance": None,
        "liveness_passed": False,
    }


def status_from_pad_result(pad_result: PadResult) -> dict[str, Any]:
    """
    Map a PAD failure into the recognition status dictionary.
    """
    reason = _PAD_USER_MESSAGES.get(
        pad_result.user_message_key,
        "Liveness check failed",
    )
    detail = (
        f"pad={pad_result.outcome.value} "
        f"code={pad_result.reason_code} "
        f"model={pad_result.model_id}"
    )
    return {
        "access_granted": False,
        "primary_text": "ACCESS DENIED",
        "secondary_text": reason,
        "detail_text": detail,
        "user_name": None,
        "cosine_distance": None,
        "liveness_passed": False,
        "pad_outcome": pad_result.outcome.value,
        "pad_reason_code": pad_result.reason_code,
    }


def classify_recognition_error(error: Exception) -> dict[str, Any]:
    message = str(error).lower()

    if "expected exactly one face" in message:
        return build_denied_result(
            reason="Multiple faces detected",
            detail="Use one face in the camera frame.",
        )

    if "no usable face" in message or "face" in message and "detected" in message:
        return build_denied_result(
            reason="No face detected",
            detail="Improve lighting and face the camera.",
        )

    return build_denied_result(
        reason="Recognition failed",
        detail="Check lighting, camera angle, and registration quality.",
    )


def run_pad_check(
    frame: Any,
    frame_number: int,
    pad_session: PadSession | None = None,
) -> PadResult:
    """
    Run PAD on the current frame (session clip or single-frame check).
    """
    if pad_session is not None:
        pad_session.add_frame(
            build_pad_frame(image=frame, frame_index=frame_number),
        )
        return pad_session.evaluate_final()

    return check_single_image_pad(image=frame, frame_index=frame_number)


def recognize_identity(
    frame: Any,
    trusted_records: list[dict[str, Any]],
    threshold: float = DEFAULT_COSINE_THRESHOLD,
) -> dict[str, Any]:
    """
  Match identity only — caller must ensure PAD already passed.
    """
    try:
        live_embedding = generate_embedding(
            image_input=frame,
            model_name=DEFAULT_MODEL_NAME,
            detector_backend=DEFAULT_DETECTOR_BACKEND,
        )
        match = find_best_match(
            candidate_embedding=live_embedding,
            known_records=trusted_records,
            threshold=threshold,
        )
    except Exception as error:
        log_exception(LOGGER, "recognition_inference_failed", error)
        return classify_recognition_error(error)

    distance = match.get("cosine_distance")
    distance_text = format_distance(distance)
    threshold_text = f"{threshold:.2f}"

    if match["is_match"]:
        user_name = match.get("user_name") or "unknown"
        return {
            "access_granted": True,
            "primary_text": "ACCESS GRANTED",
            "secondary_text": f"User: {user_name}",
            "detail_text": f"distance={distance_text} threshold={threshold_text}",
            "user_name": user_name,
            "cosine_distance": distance,
            "liveness_passed": True,
            "pad_outcome": PadOutcome.LIVE.value,
        }

    closest_user = match.get("user_name") or "unknown"
    log_event(
        LOGGER,
        "recognition_denied",
        level=logging.WARNING,
        closest_user=closest_user,
        cosine_distance=distance_text,
        threshold=threshold_text,
    )
    return {
        "access_granted": False,
        "primary_text": "ACCESS DENIED",
        "secondary_text": f"Closest user: {closest_user}",
        "detail_text": f"distance={distance_text} threshold={threshold_text}",
        "user_name": None,
        "cosine_distance": distance,
        "liveness_passed": True,
        "pad_outcome": PadOutcome.LIVE.value,
    }


def recognize_frame(
    frame: Any,
    trusted_records: list[dict[str, Any]],
    threshold: float = DEFAULT_COSINE_THRESHOLD,
    frame_number: int = 1,
    pad_session: PadSession | None = None,
    pad_result: PadResult | None = None,
) -> dict[str, Any]:
    """
    Run one recognition attempt on one webcam frame.

    PAD runs before embedding generation when enabled.
    """
    if len(trusted_records) == 0:
        return build_denied_result(
            reason="No registered users",
            detail="Register a trusted user first.",
        )

    if is_bad_lighting(frame):
        brightness = estimate_brightness(frame)
        log_event(
            LOGGER,
            "recognition_low_lighting",
            level=logging.WARNING,
            brightness=f"{brightness:.1f}",
            threshold=LOW_LIGHT_THRESHOLD,
        )
        return build_denied_result(
            reason="Bad lighting",
            detail="Move to a brighter area.",
        )

    if PAD_SETTINGS.enabled:
        if pad_result is None:
            pad_result = run_pad_check(
                frame=frame,
                frame_number=frame_number,
                pad_session=pad_session,
            )

        if pad_result.outcome != PadOutcome.LIVE:
            log_event(
                LOGGER,
                "recognition_pad_denied",
                level=logging.WARNING,
                pad_outcome=pad_result.outcome.value,
                reason_code=pad_result.reason_code,
            )
            return status_from_pad_result(pad_result)

    return recognize_identity(
        frame=frame,
        trusted_records=trusted_records,
        threshold=threshold,
    )


def should_run_inference(
    frame_number: int,
    next_allowed_time: float,
    frame_skip: int,
) -> bool:
    safe_frame_skip = max(1, frame_skip)
    is_selected_frame = frame_number % safe_frame_skip == 0
    cooldown_finished = time.monotonic() >= next_allowed_time
    return is_selected_frame and cooldown_finished


def _waiting_status(pad_hint: str = "") -> dict[str, Any]:
    return {
        "access_granted": False,
        "primary_text": "WAITING",
        "secondary_text": pad_hint or "Looking for a registered face",
        "detail_text": "",
        "user_name": None,
        "cosine_distance": None,
        "liveness_passed": None,
    }


def recognize_live(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    threshold: float = DEFAULT_COSINE_THRESHOLD,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
) -> dict[str, Any]:
    """
    Run live webcam recognition until the user presses q.
    """
    trusted_records = load_trusted_records()
    last_status = _waiting_status()
    cap: Any | None = None
    frame_number = 0
    next_allowed_time = 0.0
    pad_session: PadSession | None = None

    if PAD_SETTINGS.enabled:
        pad_session = create_pad_session(PAD_SETTINGS)
        log_event(
            LOGGER,
            "pad_session_started",
            session_id=pad_session.session_id,
            tier=PAD_SETTINGS.assurance_tier,
            mode=PAD_SETTINGS.mode,
            model_id=PAD_SETTINGS.passive_model_id,
        )

    try:
        cap = open_webcam(camera_index)
        log_event(
            LOGGER,
            "recognition_started",
            camera_index=camera_index,
            frame_skip=max(1, frame_skip),
            cooldown_seconds=cooldown_seconds,
            threshold=threshold,
            pad_enabled=PAD_SETTINGS.enabled,
        )
        print_status("Live recognition started. Press 'q' to quit.")
        if PAD_SETTINGS.enabled:
            print_status("Liveness detection is active.", level="info")

        while True:
            frame = read_frame(cap)
            frame_number += 1

            if pad_session is not None:
                pad_session.add_frame(
                    build_pad_frame(image=frame, frame_index=frame_number),
                )

            if should_run_inference(
                frame_number=frame_number,
                next_allowed_time=next_allowed_time,
                frame_skip=frame_skip,
            ):
                if pad_session is not None:
                    pad_result = pad_session.evaluate_final()

                    if pad_result.outcome == PadOutcome.LIVE:
                        last_status = recognize_identity(
                            frame=frame,
                            trusted_records=trusted_records,
                            threshold=threshold,
                        )
                        pad_session.reset()
                    elif pad_result.outcome == PadOutcome.INCONCLUSIVE:
                        hint = _PAD_USER_MESSAGES.get(
                            pad_result.user_message_key,
                            "Verifying liveness...",
                        )
                        last_status = _waiting_status(pad_hint=hint)
                    else:
                        last_status = status_from_pad_result(pad_result)
                        pad_session.reset()
                else:
                    last_status = recognize_frame(
                        frame=frame,
                        trusted_records=trusted_records,
                        threshold=threshold,
                        frame_number=frame_number,
                    )

                next_allowed_time = time.monotonic() + cooldown_seconds

                if last_status.get("access_granted"):
                    log_event(
                        LOGGER,
                        "recognition_granted",
                        user_name=last_status["user_name"],
                        cosine_distance=format_distance(
                            last_status.get("cosine_distance")
                        ),
                    )
                    print_status(
                        f"ACCESS GRANTED: {last_status['user_name']}",
                        level="success",
                    )
                elif last_status.get("primary_text") == "ACCESS DENIED":
                    print_status(last_status["secondary_text"], level="warning")

            draw_status_overlay(
                frame=frame,
                status=last_status,
                frame_skip=max(1, frame_skip),
                cooldown_seconds=cooldown_seconds,
            )

            if pad_session is not None and pad_session.current_challenge() is not None:
                draw_pad_hint(frame, pad_session.current_challenge())

            show_frame(RECOGNITION_WINDOW_NAME, frame)
            key = read_display_key()

            if is_quit_key(key):
                log_event(LOGGER, "recognition_stopped_by_user")
                print_status("Live recognition stopped by user.")
                break

    except RuntimeError as error:
        log_exception(LOGGER, "recognition_runtime_error", error)
        print_status(str(error), level="error")
        last_status = build_denied_result(
            reason="Camera error",
            detail=str(error),
        )

    finally:
        release_webcam(cap)
        close_windows()

    return last_status


if __name__ == "__main__":
    recognize_live()
