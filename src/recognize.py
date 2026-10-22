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
        load_face_model,
    )
    from src.capture import close_windows, open_webcam, read_frame, release_webcam
    from src.integrity import enforce_strict_mode_policy
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
        ChallengeState,
        MSG_CHALLENGE_PROMPT,
        MSG_CHALLENGE_TIMEOUT,
        MSG_LIVENESS_FAILED,
        MSG_LIVENESS_INCONCLUSIVE,
        REASON_ACTIVE_PASSED,
        PadOutcome,
        PadResult,
        PadSession,
        build_pad_frame,
        check_single_image_pad,
        create_pad_session,
    )
    from src.register import detect_single_face
    from src.utils import clean_name, print_status
except ModuleNotFoundError:
    from config import PAD_SETTINGS, RECOGNITION_SETTINGS, WEBCAM_SETTINGS
    from ai_engine import (
        DEFAULT_COSINE_THRESHOLD,
        DEFAULT_DETECTOR_BACKEND,
        DEFAULT_MODEL_NAME,
        find_best_match,
        generate_embedding,
        load_all_user_embeddings,
        load_face_model,
    )
    from capture import close_windows, open_webcam, read_frame, release_webcam
    from integrity import enforce_strict_mode_policy
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
        ChallengeState,
        MSG_CHALLENGE_PROMPT,
        MSG_CHALLENGE_TIMEOUT,
        MSG_LIVENESS_FAILED,
        MSG_LIVENESS_INCONCLUSIVE,
        REASON_ACTIVE_PASSED,
        PadOutcome,
        PadResult,
        PadSession,
        build_pad_frame,
        check_single_image_pad,
        create_pad_session,
    )
    from register import detect_single_face
    from utils import clean_name, print_status


DEFAULT_FRAME_SKIP = RECOGNITION_SETTINGS.frame_skip
DEFAULT_COOLDOWN_SECONDS = RECOGNITION_SETTINGS.cooldown_seconds
DEFAULT_RECENTER_TIMEOUT_SECONDS = 4.0
DEFAULT_GRANT_EXPIRY_SECONDS = 2.5
LOW_LIGHT_THRESHOLD = RECOGNITION_SETTINGS.brightness_threshold
DEFAULT_CAMERA_INDEX = WEBCAM_SETTINGS.camera_index

LOGGER = get_logger(__name__)


def extract_face_crop(
    frame: Any,
    face_box: tuple[int, int, int, int],
    margin: float = 0.25,
) -> Any:
    """
    Extract a padded bounding-box crop around the detected face.
    Provides context margin for DeepFace eye landmark alignment while
    preventing peripheral background artifacts from triggering false extra faces.
    """
    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return frame
    x, y, w, h = face_box
    H, W = frame.shape[:2]
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else frame

_PAD_USER_MESSAGES = {
    MSG_LIVENESS_FAILED: "Liveness check failed",
    MSG_LIVENESS_INCONCLUSIVE: "Could not verify liveness",
    MSG_CHALLENGE_PROMPT: "Follow on-screen liveness prompt",
    MSG_CHALLENGE_TIMEOUT: "Liveness challenge timed out",
}


def load_trusted_records(encodings_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Load registered user embeddings from encodings/, strictly filtering
    for the single authorized owner to enforce 1:1 verification.
    """
    try:
        all_records = (
            load_all_user_embeddings(encodings_dir=encodings_dir)
            if encodings_dir is not None
            else load_all_user_embeddings()
        )
        owner_canonical = clean_name(RECOGNITION_SETTINGS.owner_name)
        records = [
            r for r in all_records
            if clean_name(r.get("user_name") or "") == owner_canonical
        ]
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
        if len(all_records) > 0:
            print_status(f"Found {len(all_records)} records, but none matched owner '{RECOGNITION_SETTINGS.owner_name}'.", level="warning")
        else:
            print_status("No trusted users found. Register the owner first.", level="warning")
    else:
        log_event(
            LOGGER,
            "trusted_embeddings_loaded",
            record_count=len(records),
        )
        print_status(f"Loaded trusted owner record: {RECOGNITION_SETTINGS.owner_name}", level="success")

    return records


def estimate_brightness(frame: Any) -> float:
    return float(frame.mean())


def is_bad_lighting(frame: Any, threshold: float = LOW_LIGHT_THRESHOLD) -> bool:
    return estimate_brightness(frame) < threshold


def format_distance(distance: float | None) -> str:
    if distance is None:
        return "n/a"
    return f"{distance:.3f}"


def build_denied_result(
    reason: str,
    detail: str = "",
    state: str = "NOT VERIFIED",
) -> dict[str, Any]:
    return {
        "access_granted": False,
        "state": state,
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
    state = "TRY AGAIN" if pad_result.outcome == PadOutcome.INCONCLUSIVE else "NOT VERIFIED"
    return {
        "access_granted": False,
        "state": state,
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


def check_recenter_criteria(
    face_box: tuple[int, int, int, int] | None,
    face_error: str | None,
    frame_shape: tuple[int, ...],
    start_centroid_x: float | None = None,
    max_shift: float | None = None,
    min_face_size_ratio: float = 0.15,
    max_face_size_ratio: float = 0.85,
    min_aspect_ratio: float = 0.70,
    max_aspect_ratio: float = 1.30,
) -> tuple[bool, str]:
    """
    Evaluate whether a detected face meets conservative centering and pose criteria
    to be considered re-centered and facing the camera after an active head-turn challenge.

    Measurements Used:
    1. Single Face Detection: Requires exactly one face detected (face_box is not None, face_error is None).
    2. Bounding Box Geometry: Checks width/height ratio (aspect ratio) in [min_aspect_ratio, max_aspect_ratio]
       (typically [0.70, 1.30] for human faces detected by OpenCV frontal Haar cascade).
    3. Face Scale: Relative height h / frame_height must be in [0.15, 0.85].
    4. Frame Centering: Centroid (cx, cy) must lie within the central area of the frame
       (0.20 <= cx / W <= 0.80 and 0.15 <= cy / H <= 0.85).
    5. Baseline Centroid Recovery: When start_centroid_x (from pre-turn baseline) is provided,
       horizontal displacement |cx - start_centroid_x| must return within an allowable drift:
       if max_shift is known, allowed_drift = min(14.0, max(8.0, 0.40 * max_shift)),
       otherwise allowed_drift = 12.0 px.

    Architectural Limitations on Pose Estimation:
    OpenCV's Haar frontal cascade (haarcascade_frontalface_default.xml) detects 2D image
    features and does NOT compute 3D facial landmarks or continuous 3D head pose angles
    (yaw, pitch, roll). While centroid recovery to the pre-turn baseline and bounding box
    aspect ratio symmetry indicate the user has shifted their head back toward the center
    camera axis, this criterion serves as a conservative heuristic proxy and cannot
    mathematically guarantee zero degrees yaw without 3D landmark regression.

    Returns:
        (is_frontal_and_centered, reason_code)
    """
    if face_box is None:
        return False, face_error or "no_face"

    if face_error is not None:
        return False, face_error

    if not frame_shape or len(frame_shape) < 2:
        return False, "invalid_frame_shape"

    h_frame, w_frame = frame_shape[:2]
    if h_frame <= 0 or w_frame <= 0:
        return False, "invalid_frame_dimensions"

    x, y, w, h = face_box
    if w <= 0 or h <= 0:
        return False, "invalid_box_size"

    # Aspect ratio check
    aspect_ratio = float(w) / float(h)
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False, "aspect_ratio_out_of_bounds"

    # Face scale check
    h_ratio = float(h) / float(h_frame)
    if h_ratio < min_face_size_ratio or h_ratio > max_face_size_ratio:
        return False, "face_size_out_of_bounds"

    # Frame centering check
    cx = float(x + w / 2.0)
    cy = float(y + h / 2.0)
    norm_cx = cx / float(w_frame)
    norm_cy = cy / float(h_frame)

    if not (0.20 <= norm_cx <= 0.80) or not (0.15 <= norm_cy <= 0.85):
        return False, "off_center_in_frame"

    # Relative centroid return
    if start_centroid_x is not None:
        delta_from_start = abs(cx - float(start_centroid_x))
        if max_shift is not None and max_shift > 0:
            allowed_drift = max(14.0, min(0.06 * float(w_frame), 0.45 * float(max_shift)))
        else:
            allowed_drift = max(14.0, 0.05 * float(w_frame))

        if delta_from_start > allowed_drift:
            return False, "head_still_turned"

    return True, "frontal_centered"


def run_pad_check(
    frame: Any,
    frame_number: int,
    pad_session: PadSession | None = None,
    face_box: tuple[int, int, int, int] | None = None,
) -> PadResult:
    """
    Run PAD on the current frame (session clip or single-frame check).
    """
    if face_box is None:
        face_box, _ = detect_single_face(frame)

    if pad_session is not None:
        pad_session.add_frame(
            build_pad_frame(
                image=frame,
                frame_index=frame_number,
                face_box=face_box,
            ),
        )
        return pad_session.evaluate_final()

    return check_single_image_pad(image=frame, frame_index=frame_number)


def recognize_identity(
    frame: Any,
    trusted_records: list[dict[str, Any]],
    threshold: float = DEFAULT_COSINE_THRESHOLD,
    face_box: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """
    Match identity only — caller must ensure PAD already passed.
    Uses padded face crop if face_box is available, falling back to full frame
    if needed, to prevent false multi-face detections and DeepFace disagreement.
    """
    t0_attempt = time.perf_counter()
    t_embed_ms = 0.0
    t_match_ms = 0.0
    live_embedding = None
    extraction_method = "full_frame"

    try:
        if face_box is not None:
            crop = extract_face_crop(frame, face_box)
            if crop is not frame and crop.size > 0:
                t0_embed = time.perf_counter()
                try:
                    live_embedding = generate_embedding(
                        image_input=crop,
                        model_name=DEFAULT_MODEL_NAME,
                        detector_backend=DEFAULT_DETECTOR_BACKEND,
                    )
                    t_embed_ms = (time.perf_counter() - t0_embed) * 1000.0
                    extraction_method = "crop"
                except Exception as crop_error:
                    log_event(
                        LOGGER,
                        "recognition_crop_fallback",
                        level=logging.DEBUG,
                        error=str(crop_error),
                    )
                    live_embedding = None

        if live_embedding is None:
            t0_embed = time.perf_counter()
            live_embedding = generate_embedding(
                image_input=frame,
                model_name=DEFAULT_MODEL_NAME,
                detector_backend=DEFAULT_DETECTOR_BACKEND,
            )
            t_embed_ms = (time.perf_counter() - t0_embed) * 1000.0
            extraction_method = "full_frame"

        t0_match = time.perf_counter()
        match = find_best_match(
            candidate_embedding=live_embedding,
            known_records=trusted_records,
            threshold=threshold,
        )
        t_match_ms = (time.perf_counter() - t0_match) * 1000.0
    except Exception as error:
        log_exception(LOGGER, "recognition_inference_failed", error)
        res = classify_recognition_error(error)
        res["timing_profile"] = {
            "embed_ms": round(t_embed_ms, 2),
            "match_ms": round(t_match_ms, 2),
            "total_ms": round((time.perf_counter() - t0_attempt) * 1000.0, 2),
            "method": extraction_method,
        }
        return res

    distance = match.get("cosine_distance")
    distance_text = format_distance(distance)
    threshold_text = f"{threshold:.2f}"
    t_total_ms = (time.perf_counter() - t0_attempt) * 1000.0

    timing_profile = {
        "embed_ms": round(t_embed_ms, 2),
        "match_ms": round(t_match_ms, 2),
        "total_ms": round(t_total_ms, 2),
        "method": extraction_method,
    }

    if match["is_match"]:
        user_name = match.get("user_name") or "unknown"
        return {
            "access_granted": True,
            "state": "VERIFIED",
            "primary_text": "ACCESS GRANTED",
            "secondary_text": f"User: {user_name}",
            "detail_text": f"distance={distance_text} threshold={threshold_text}",
            "user_name": user_name,
            "cosine_distance": distance,
            "liveness_passed": True,
            "pad_outcome": PadOutcome.LIVE.value,
            "timing_profile": timing_profile,
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
        "state": "NOT VERIFIED",
        "primary_text": "ACCESS DENIED",
        "secondary_text": "Face not recognized",
        "detail_text": "Try again or re-register",
        "user_name": None,
        "cosine_distance": distance,
        "liveness_passed": True,
        "pad_outcome": PadOutcome.LIVE.value,
        "timing_profile": timing_profile,
    }


def recognize_frame(
    frame: Any,
    trusted_records: list[dict[str, Any]],
    threshold: float = DEFAULT_COSINE_THRESHOLD,
    frame_number: int = 1,
    pad_session: PadSession | None = None,
    pad_result: PadResult | None = None,
    face_box: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """
    Run one recognition attempt on one webcam frame.

    PAD runs before embedding generation when enabled.
    """
    t0_frame = time.perf_counter()
    t_detect_ms = 0.0
    t_pad_ms = 0.0

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

    if face_box is None:
        t0_det = time.perf_counter()
        face_box, _ = detect_single_face(frame)
        t_detect_ms = (time.perf_counter() - t0_det) * 1000.0

    if PAD_SETTINGS.enabled:
        t0_pad = time.perf_counter()
        if pad_result is None:
            pad_result = run_pad_check(
                frame=frame,
                frame_number=frame_number,
                pad_session=pad_session,
                face_box=face_box,
            )
        t_pad_ms = (time.perf_counter() - t0_pad) * 1000.0

        if pad_result.outcome != PadOutcome.LIVE:
            log_event(
                LOGGER,
                "recognition_pad_denied",
                level=logging.WARNING,
                pad_outcome=pad_result.outcome.value,
                reason_code=pad_result.reason_code,
            )
            res = status_from_pad_result(pad_result)
            res["timing_profile"] = {
                "detect_ms": round(t_detect_ms, 2),
                "pad_ms": round(t_pad_ms, 2),
                "total_ms": round((time.perf_counter() - t0_frame) * 1000.0, 2),
            }
            return res

        if pad_result.reason_code == REASON_ACTIVE_PASSED or pad_result.active_passed is True:
            log_event(
                LOGGER,
                "recognition_active_challenge_passed_frame",
                frame_number=frame_number,
            )
            res = {
                "access_granted": False,
                "state": "LIVENESS VERIFIED — LOOK AT CAMERA",
                "primary_text": "LIVENESS VERIFIED — LOOK AT CAMERA",
                "secondary_text": "Look directly at the camera",
                "detail_text": "Re-centering required before identity extraction",
                "user_name": None,
                "cosine_distance": None,
                "liveness_passed": True,
                "pad_outcome": pad_result.outcome.value,
                "pad_reason_code": pad_result.reason_code,
            }
            res["timing_profile"] = {
                "detect_ms": round(t_detect_ms, 2),
                "pad_ms": round(t_pad_ms, 2),
                "total_ms": round((time.perf_counter() - t0_frame) * 1000.0, 2),
            }
            return res

    res = recognize_identity(
        frame=frame,
        trusted_records=trusted_records,
        threshold=threshold,
        face_box=face_box,
    )
    t_total_ms = (time.perf_counter() - t0_frame) * 1000.0
    id_profile = res.get("timing_profile", {})
    timing_profile = {
        "detect_ms": round(t_detect_ms, 2),
        "pad_ms": round(t_pad_ms, 2),
        "embed_ms": id_profile.get("embed_ms", 0.0),
        "match_ms": id_profile.get("match_ms", 0.0),
        "total_ms": round(t_total_ms, 2),
        "method": id_profile.get("method", "full_frame"),
        "reason_code": (
            res.get("pad_reason_code")
            or ("match_granted" if res.get("access_granted") else "match_denied")
        ),
    }
    res["timing_profile"] = timing_profile
    log_event(LOGGER, "recognition_cycle_profile", level=logging.DEBUG, **timing_profile)
    return res


def should_run_inference(
    frame_number: int,
    next_allowed_time: float,
    frame_skip: int,
) -> bool:
    safe_frame_skip = max(1, frame_skip)
    is_selected_frame = frame_number % safe_frame_skip == 0
    cooldown_finished = time.monotonic() >= next_allowed_time
    return is_selected_frame and cooldown_finished


def _waiting_status(
    pad_hint: str = "",
    state: str = "DETECTING FACE",
) -> dict[str, Any]:
    return {
        "access_granted": False,
        "state": state,
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
    enforce_strict_mode_policy(require_operator_auth=False)
    trusted_records = load_trusted_records()
    last_status = _waiting_status()
    cap: Any | None = None
    frame_number = 0
    next_allowed_time = 0.0
    pad_session: PadSession | None = None

    awaiting_recenter = False
    recenter_deadline = 0.0
    recenter_completion_frame = 0
    recenter_start_cx: float | None = None
    recenter_max_shift: float | None = None

    last_inferred_frame_number = -1
    last_logged_decision_state: str | None = None
    grant_timestamp = 0.0

    try:
        load_face_model(DEFAULT_MODEL_NAME)
    except Exception as warm_err:
        log_event(LOGGER, "model_warmup_warning", level=logging.DEBUG, error=str(warm_err))

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
            t0_read = time.perf_counter()
            frame = read_frame(cap)
            t_read_ms = (time.perf_counter() - t0_read) * 1000.0
            frame_number += 1

            t0_detect = time.perf_counter()
            face_box, face_error = detect_single_face(frame)
            t_detect_ms = (time.perf_counter() - t0_detect) * 1000.0

            # Stale-result expiry: Require fresh face presence and expire promptly on face loss or TTL
            if last_status.get("access_granted"):
                if grant_timestamp == 0.0:
                    grant_timestamp = time.monotonic()
                if face_box is None and frame_number > 1:
                    last_status = _waiting_status(pad_hint="Looking for registered face", state="DETECTING FACE")
                    grant_timestamp = 0.0
                    last_logged_decision_state = None
                elif time.monotonic() - grant_timestamp > DEFAULT_GRANT_EXPIRY_SECONDS:
                    last_status = _waiting_status(pad_hint="Looking for registered face", state="DETECTING FACE")
                    grant_timestamp = 0.0
                    last_logged_decision_state = None

            if is_bad_lighting(frame):
                if pad_session is not None:
                    pad_session.reset()
                awaiting_recenter = False
                face_box = None
                if (
                    should_run_inference(
                        frame_number=frame_number,
                        next_allowed_time=next_allowed_time,
                        frame_skip=frame_skip,
                    )
                    and frame_number > last_inferred_frame_number
                ):
                    last_inferred_frame_number = frame_number
                    last_status = build_denied_result(
                        reason="Bad lighting",
                        detail="Move to a brighter area.",
                        state="TRY AGAIN",
                    )
                    next_allowed_time = time.monotonic() + cooldown_seconds
            elif awaiting_recenter:
                now = time.monotonic()

                if now > recenter_deadline:
                    log_event(
                        LOGGER,
                        "recognition_recenter_timeout",
                        level=logging.WARNING,
                        frame_number=frame_number,
                        deadline=recenter_deadline,
                    )
                    last_status = build_denied_result(
                        reason="Re-centering timed out",
                        detail="Face camera and try again.",
                        state="TRY AGAIN",
                    )
                    last_status["liveness_passed"] = False
                    if pad_session is not None:
                        pad_session.reset()
                    awaiting_recenter = False
                    next_allowed_time = now + cooldown_seconds
                elif frame_number > recenter_completion_frame:
                    is_frontal, recenter_reason = check_recenter_criteria(
                        face_box=face_box,
                        face_error=face_error,
                        frame_shape=frame.shape,
                        start_centroid_x=recenter_start_cx,
                        max_shift=recenter_max_shift,
                    )
                    if is_frontal:
                        log_event(
                            LOGGER,
                            "recognition_recenter_passed",
                            frame_number=frame_number,
                            reason=recenter_reason,
                        )
                        t0_attempt = time.perf_counter()
                        last_status = recognize_identity(
                            frame=frame,
                            trusted_records=trusted_records,
                            threshold=threshold,
                            face_box=face_box,
                        )
                        profile = {
                            "read_ms": round(t_read_ms, 2),
                            "detect_ms": round(t_detect_ms, 2),
                            "pad_ms": 0.0,
                            "embed_ms": last_status.get("timing_profile", {}).get("embed_ms", 0.0),
                            "match_ms": last_status.get("timing_profile", {}).get("match_ms", 0.0),
                            "total_ms": round((time.perf_counter() - t0_attempt) * 1000.0, 2),
                            "reason_code": (
                                "match_granted" if last_status.get("access_granted") else "match_denied"
                            ),
                            "state": last_status.get("state", "UNKNOWN"),
                        }
                        last_status["timing_profile"] = profile
                        log_event(LOGGER, "recognition_cycle_profile", level=logging.DEBUG, **profile)
                        if pad_session is not None:
                            pad_session.reset()
                        awaiting_recenter = False
                        next_allowed_time = time.monotonic() + cooldown_seconds
                    else:
                        secondary = (
                            "Center face in front of camera"
                            if face_box is None
                            else "Look directly at the camera"
                        )
                        detail = (
                            "Waiting for face..."
                            if face_box is None
                            else f"Re-centering ({recenter_reason})..."
                        )
                        last_status = {
                            "access_granted": False,
                            "state": "LIVENESS VERIFIED — LOOK AT CAMERA",
                            "primary_text": "LIVENESS VERIFIED — LOOK AT CAMERA",
                            "secondary_text": secondary,
                            "detail_text": detail,
                            "user_name": None,
                            "cosine_distance": None,
                            "liveness_passed": True,
                        }
            else:
                if pad_session is not None:
                    pad_session.add_frame(
                        build_pad_frame(
                            image=frame,
                            frame_index=frame_number,
                            face_box=face_box,
                        ),
                    )

                if (
                    should_run_inference(
                        frame_number=frame_number,
                        next_allowed_time=next_allowed_time,
                        frame_skip=frame_skip,
                    )
                    and frame_number > last_inferred_frame_number
                ):
                    last_inferred_frame_number = frame_number
                    t0_inference = time.perf_counter()
                    t_pad_ms = 0.0

                    if pad_session is not None:
                        t0_pad = time.perf_counter()
                        pad_result = pad_session.evaluate_final()
                        t_pad_ms = (time.perf_counter() - t0_pad) * 1000.0

                        if pad_result.outcome == PadOutcome.LIVE:
                            if (
                                pad_result.reason_code == REASON_ACTIVE_PASSED
                                or pad_result.active_passed is True
                            ):
                                awaiting_recenter = True
                                recenter_completion_frame = frame_number
                                recenter_deadline = (
                                    time.monotonic() + DEFAULT_RECENTER_TIMEOUT_SECONDS
                                )
                                recenter_start_cx = pad_session.active_start_centroid_x()
                                recenter_max_shift = pad_session.active_max_delta_x()
                                last_status = {
                                    "access_granted": False,
                                    "state": "LIVENESS VERIFIED — LOOK AT CAMERA",
                                    "primary_text": "LIVENESS VERIFIED — LOOK AT CAMERA",
                                    "secondary_text": "Look directly at the camera",
                                    "detail_text": "Re-centering face...",
                                    "user_name": None,
                                    "cosine_distance": None,
                                    "liveness_passed": True,
                                    "pad_outcome": pad_result.outcome.value,
                                    "pad_reason_code": pad_result.reason_code,
                                }
                                log_event(
                                    LOGGER,
                                    "pad_challenge_recenter_started",
                                    frame_number=frame_number,
                                    session_id=pad_session.session_id,
                                    start_cx=recenter_start_cx,
                                    max_shift=recenter_max_shift,
                                )
                            else:
                                last_status = recognize_identity(
                                    frame=frame,
                                    trusted_records=trusted_records,
                                    threshold=threshold,
                                    face_box=face_box,
                                )
                                pad_session.reset()
                        elif pad_result.outcome == PadOutcome.INCONCLUSIVE:
                            hint = _PAD_USER_MESSAGES.get(
                                pad_result.user_message_key,
                                "Verifying liveness...",
                            )
                            last_status = _waiting_status(
                                pad_hint=hint,
                                state="CHECKING LIVENESS",
                            )
                        else:
                            last_status = status_from_pad_result(pad_result)
                            pad_session.reset()
                    else:
                        last_status = recognize_frame(
                            frame=frame,
                            trusted_records=trusted_records,
                            threshold=threshold,
                            frame_number=frame_number,
                            face_box=face_box,
                        )

                    profile = {
                        "read_ms": round(t_read_ms, 2),
                        "detect_ms": round(t_detect_ms, 2),
                        "pad_ms": round(t_pad_ms, 2),
                        "embed_ms": last_status.get("timing_profile", {}).get("embed_ms", 0.0),
                        "match_ms": last_status.get("timing_profile", {}).get("match_ms", 0.0),
                        "total_ms": round((time.perf_counter() - t0_inference) * 1000.0, 2),
                        "reason_code": (
                            last_status.get("pad_reason_code")
                            or ("match_granted" if last_status.get("access_granted") else "match_denied")
                        ),
                        "state": last_status.get("state", "UNKNOWN"),
                    }
                    last_status["timing_profile"] = profile
                    log_event(LOGGER, "recognition_cycle_profile", level=logging.DEBUG, **profile)
                    next_allowed_time = time.monotonic() + cooldown_seconds
                elif time.monotonic() >= next_allowed_time:
                    if face_box is None:
                        if face_error == "multiple_faces":
                            last_status = build_denied_result(
                                reason="Multiple faces detected",
                                detail="Only 1 person allowed in camera frame.",
                                state="NOT VERIFIED",
                            )
                        elif last_status.get("state") not in ("VERIFIED", "NOT VERIFIED", "TRY AGAIN"):
                            last_status = _waiting_status(
                                pad_hint="Center face in front of camera",
                                state="DETECTING FACE",
                            )
                    else:
                        if pad_session is not None and pad_session.needs_more_frames():
                            challenge = pad_session.current_challenge()
                            hint = "Follow on-screen challenge" if challenge else "Checking liveness..."
                            last_status = _waiting_status(
                                pad_hint=hint,
                                state="CHECKING LIVENESS",
                            )
                        elif last_status.get("state") not in ("VERIFIED", "NOT VERIFIED", "TRY AGAIN"):
                            last_status = _waiting_status(
                                pad_hint="Verifying identity...",
                                state="VERIFYING IDENTITY",
                            )

            # De-duplicated decision logging: log transition only once to prevent display loop flooding
            current_decision = (
                "GRANTED" if last_status.get("access_granted")
                else ("DENIED" if last_status.get("primary_text") == "ACCESS DENIED" else None)
            )
            if current_decision is not None and current_decision != last_logged_decision_state:
                last_logged_decision_state = current_decision
                if last_status.get("access_granted"):
                    grant_timestamp = time.monotonic()
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
            elif current_decision is None:
                last_logged_decision_state = None

            draw_status_overlay(
                frame=frame,
                status=last_status,
                frame_skip=max(1, frame_skip),
                cooldown_seconds=cooldown_seconds,
                face_box=face_box,
            )

            if (
                pad_session is not None
                and pad_session.current_challenge() is not None
                and getattr(pad_session.current_challenge(), "state", None) == ChallengeState.IN_PROGRESS
            ):
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
