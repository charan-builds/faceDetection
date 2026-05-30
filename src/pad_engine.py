"""
Presentation Attack Detection (PAD) for the Face Lock AI project.

Owns liveness / anti-spoof decisions only:
- frame clip buffering
- passive scoring (heuristic or mock)
- optional active challenge (head movement)
- fusion into PadResult

Does not generate face embeddings, open menus, or call DeepFace.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

try:
    from src.config import PAD_SETTINGS
    from src.logging_config import get_logger, log_event
except ModuleNotFoundError:
    from config import PAD_SETTINGS
    from logging_config import get_logger, log_event


LOGGER = get_logger(__name__)

# --- Reason codes and user-facing message keys ---

REASON_PAD_DISABLED = "pad_disabled"
REASON_PASSIVE_LIVE = "passive_live"
REASON_PASSIVE_ATTACK = "passive_attack"
REASON_PASSIVE_INCONCLUSIVE = "passive_inconclusive"
REASON_ACTIVE_PASSED = "active_challenge_passed"
REASON_ACTIVE_FAILED = "active_challenge_failed"
REASON_ACTIVE_TIMEOUT = "challenge_timeout"
REASON_TOO_FEW_FRAMES = "too_few_frames"
REASON_PAD_ERROR = "pad_error"
REASON_UNKNOWN_MODEL = "unknown_model"

MSG_VERIFYING = "verifying_liveness"
MSG_LIVENESS_FAILED = "liveness_failed"
MSG_LIVENESS_INCONCLUSIVE = "liveness_inconclusive"
MSG_CHALLENGE_PROMPT = "challenge_prompt"
MSG_CHALLENGE_TIMEOUT = "challenge_timeout"


class PadOutcome(str, Enum):
    """Presentation attack detection decision."""

    LIVE = "LIVE"
    ATTACK = "ATTACK"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"


class PadMode(str, Enum):
    """How passive and active checks are combined."""

    PASSIVE_ONLY = "passive_only"
    ACTIVE_REQUIRED = "active_required"
    PASSIVE_THEN_ACTIVE_ON_BORDERLINE = "passive_then_active_on_borderline"


class ConfidenceBand(str, Enum):
  HIGH = "HIGH"
  MEDIUM = "MEDIUM"
  LOW = "LOW"


class ChallengeState(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    PASSED = "PASSED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class ChallengeType(str, Enum):
    NONE = "none"
    HEAD_TURN_LEFT = "head_turn_left"
    HEAD_TURN_RIGHT = "head_turn_right"


class PadConfigurationError(ValueError):
    """Invalid PAD configuration."""


@dataclass(frozen=True)
class PadFrame:
    """One observation in a PAD clip."""

    frame_index: int
    timestamp_ms: float
    image: Any
    face_box: tuple[int, int, int, int] | None = None
    capture_device_id: str | None = None


@dataclass(frozen=True)
class PassiveScoreResult:
    live_probability: float
    scorer_id: str
    scorer_version: str


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    model_version: str


@dataclass(frozen=True)
class PadChallenge:
    challenge_id: str
    challenge_type: ChallengeType
    instruction_key: str
    state: ChallengeState
    started_at_ms: float
    deadline_ms: float


@dataclass(frozen=True)
class PadResult:
    outcome: PadOutcome
    reason_code: str
    user_message_key: str
    model_id: str
    model_version: str
    frames_analyzed: int
    latency_ms: float
    passive_score: float | None = None
    confidence_band: ConfidenceBand | None = None
    active_passed: bool | None = None
    challenge_id: str | None = None
    session_id: str | None = None


@dataclass
class _ClipBuilder:
    settings: Any
    _frames: list[PadFrame] = field(default_factory=list)

    def add(self, pad_frame: PadFrame) -> None:
        stride = max(1, self.settings.sample_every_n_frames)
        if self._frames and (pad_frame.frame_index % stride) != 0:
            return
        self._frames.append(pad_frame)
        max_buffer = max(1, self.settings.max_frames_buffer)
        if len(self._frames) > max_buffer:
            self._frames = self._frames[-max_buffer:]

    def clear(self) -> None:
        self._frames.clear()

    def count(self) -> int:
        return len(self._frames)

    def is_ready(self) -> bool:
        return self.count() >= max(1, self.settings.min_frames_passive)

    def frames(self) -> list[PadFrame]:
        return list(self._frames)


class PassiveScorerProtocol(Protocol):
    def score(self, frames: Sequence[PadFrame]) -> PassiveScoreResult:
        ...

    def metadata(self) -> ModelInfo:
        ...


@dataclass
class MockPassiveScorer:
    """Deterministic scorer for tests and demo bypass scenarios."""

    mode: str = "always_live"
    live_probability: float = 0.95
    model_id: str = "mock_always_live"
    model_version: str = "1.0.0"

    def score(self, frames: Sequence[PadFrame]) -> PassiveScoreResult:
        if self.mode == "always_attack":
            probability = 0.05
        elif self.mode == "borderline":
            probability = 0.55
        elif self.mode == "by_frame_count":
            probability = 0.85 if len(frames) >= 5 else 0.35
        else:
            probability = self.live_probability

        return PassiveScoreResult(
            live_probability=probability,
            scorer_id=self.model_id,
            scorer_version=self.model_version,
        )

    def metadata(self) -> ModelInfo:
        return ModelInfo(model_id=self.model_id, model_version=self.model_version)


@dataclass
class HeuristicPassiveScorer:
    """
    Lightweight RGB heuristic PAD (no ONNX).

    Uses Laplacian variance, channel correlation, and temporal motion on the
    frame sequence. Not a certified anti-spoof model.
    """

    model_id: str = "heuristic_rgb_v1"
    model_version: str = "1.0.0"

    def score(self, frames: Sequence[PadFrame]) -> PassiveScoreResult:
        if len(frames) == 0:
            return PassiveScoreResult(0.0, self.model_id, self.model_version)

        scores: list[float] = []
        for pad_frame in frames:
            scores.append(self._score_single_frame(pad_frame.image))

        motion_bonus = self._motion_score(frames)
        combined = float(np.clip(np.mean(scores) * 0.75 + motion_bonus * 0.25, 0.0, 1.0))

        return PassiveScoreResult(
            live_probability=combined,
            scorer_id=self.model_id,
            scorer_version=self.model_version,
        )

    def metadata(self) -> ModelInfo:
        return ModelInfo(model_id=self.model_id, model_version=self.model_version)

    def _extract_roi(self, image: Any) -> np.ndarray:
        if image is None or not hasattr(image, "shape"):
            return np.zeros((32, 32), dtype=np.uint8)

        height, width = image.shape[:2]
        top = height // 4
        bottom = height - height // 4
        left = width // 4
        right = width - width // 4
        roi = image[top:bottom, left:right]

        if roi.size == 0:
            return np.zeros((32, 32), dtype=np.uint8)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        return gray

    def _score_single_frame(self, image: Any) -> float:
        gray = self._extract_roi(image)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Very flat regions (some prints/screens) score lower; live faces tend higher.
        texture_score = np.clip(laplacian_var / 120.0, 0.0, 1.0)

        if len(image.shape) == 3:
            channel_std = [float(image[:, :, index].std()) for index in range(3)]
            std_spread = max(channel_std) - min(channel_std)
            color_score = np.clip(std_spread / 25.0, 0.0, 1.0)
        else:
            color_score = 0.5

        return float(np.clip(texture_score * 0.7 + color_score * 0.3, 0.0, 1.0))

    def _motion_score(self, frames: Sequence[PadFrame]) -> float:
        if len(frames) < 2:
            return 0.4

        centroids: list[float] = []
        for pad_frame in frames[-6:]:
            gray = self._extract_roi(pad_frame.image)
            moments = cv2.moments(gray)
            if moments["m00"] <= 0:
                continue
            centroids.append(float(moments["m10"] / moments["m00"]))

        if len(centroids) < 2:
            return 0.4

        movement = float(np.std(centroids))
        return float(np.clip(movement / 15.0, 0.0, 1.0))


@dataclass
class _ActiveChallengeController:
    settings: Any
    challenge: PadChallenge | None = None
    _start_centroid_x: float | None = None
    _max_delta_x: float = 0.0

    def start(self, challenge_type: ChallengeType) -> PadChallenge:
        now_ms = time.monotonic() * 1000.0
        instruction = (
            "turn_head_left"
            if challenge_type == ChallengeType.HEAD_TURN_LEFT
            else "turn_head_right"
        )
        self.challenge = PadChallenge(
            challenge_id=str(uuid.uuid4()),
            challenge_type=challenge_type,
            instruction_key=instruction,
            state=ChallengeState.IN_PROGRESS,
            started_at_ms=now_ms,
            deadline_ms=now_ms + self.settings.challenge_timeout_seconds * 1000.0,
        )
        self._start_centroid_x = None
        self._max_delta_x = 0.0
        return self.challenge

    def update(self, pad_frame: PadFrame) -> PadChallenge | None:
        if self.challenge is None:
            return None

        now_ms = time.monotonic() * 1000.0
        if now_ms > self.challenge.deadline_ms:
            self.challenge = PadChallenge(
                challenge_id=self.challenge.challenge_id,
                challenge_type=self.challenge.challenge_type,
                instruction_key=MSG_CHALLENGE_TIMEOUT,
                state=ChallengeState.EXPIRED,
                started_at_ms=self.challenge.started_at_ms,
                deadline_ms=self.challenge.deadline_ms,
            )
            return self.challenge

        centroid_x = self._centroid_x(pad_frame.image)
        if centroid_x is not None:
            if self._start_centroid_x is None:
                self._start_centroid_x = centroid_x
            else:
                self._max_delta_x = max(
                    self._max_delta_x,
                    abs(centroid_x - self._start_centroid_x),
                )

        required = float(self.settings.head_turn_min_shift_px)
        if self._max_delta_x >= required:
            self.challenge = PadChallenge(
                challenge_id=self.challenge.challenge_id,
                challenge_type=self.challenge.challenge_type,
                instruction_key=self.challenge.instruction_key,
                state=ChallengeState.PASSED,
                started_at_ms=self.challenge.started_at_ms,
                deadline_ms=self.challenge.deadline_ms,
            )

        return self.challenge

    def _centroid_x(self, image: Any) -> float | None:
        if image is None or not hasattr(image, "shape"):
            return None
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )
        moments = cv2.moments(gray)
        if moments["m00"] <= 0:
            return None
        return float(moments["m10"] / moments["m00"])


def parse_pad_mode(mode: str) -> PadMode:
    try:
        return PadMode(mode.lower())
    except ValueError as error:
        raise PadConfigurationError(f"Unknown PAD mode: {mode}") from error


def validate_pad_settings(settings: Any) -> list[str]:
    errors: list[str] = []
    reject = settings.passive_reject_threshold
    low = settings.borderline_low
    high = settings.borderline_high
    pass_threshold = settings.passive_pass_threshold

    if not (0.0 <= reject < low <= high < pass_threshold <= 1.0):
        errors.append(
            "Thresholds must satisfy 0 <= reject < borderline_low "
            "<= borderline_high < pass <= 1"
        )

    if settings.min_frames_passive < 1:
        errors.append("min_frames_passive must be >= 1")

    if settings.max_frames_buffer < settings.min_frames_passive:
        errors.append("max_frames_buffer must be >= min_frames_passive")

    return errors


def resolve_passive_scorer(model_id: str) -> PassiveScorerProtocol:
    if model_id == "mock_always_live":
        return MockPassiveScorer(mode="always_live", model_id=model_id)
    if model_id == "mock_always_attack":
        return MockPassiveScorer(mode="always_attack", model_id=model_id)
    if model_id == "mock_borderline":
        return MockPassiveScorer(mode="borderline", model_id=model_id)
    if model_id == "heuristic_rgb_v1":
        return HeuristicPassiveScorer()

    raise PadConfigurationError(f"Unknown passive PAD model id: {model_id}")


def _confidence_band(score: float, settings: Any) -> ConfidenceBand:
    if score >= settings.passive_pass_threshold:
        return ConfidenceBand.HIGH
    if score <= settings.passive_reject_threshold:
        return ConfidenceBand.LOW
    return ConfidenceBand.MEDIUM


def _outcome_from_passive(score: float, settings: Any) -> PadOutcome:
    if score >= settings.passive_pass_threshold:
        return PadOutcome.LIVE
    if score <= settings.passive_reject_threshold:
        return PadOutcome.ATTACK
    return PadOutcome.INCONCLUSIVE


def build_pad_frame(
    image: Any,
    frame_index: int,
    timestamp_ms: float | None = None,
    face_box: tuple[int, int, int, int] | None = None,
) -> PadFrame:
    return PadFrame(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms if timestamp_ms is not None else time.monotonic() * 1000.0,
        image=image,
        face_box=face_box,
    )


@dataclass
class PadSession:
    """Collects frames and produces a PadResult for one unlock or check."""

    settings: Any = field(default_factory=lambda: PAD_SETTINGS)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _clip: _ClipBuilder = field(init=False)
    _scorer: PassiveScorerProtocol = field(init=False)
    _active: _ActiveChallengeController = field(init=False)
    _pending_challenge_type: ChallengeType | None = None

    def __post_init__(self) -> None:
        errors = validate_pad_settings(self.settings)
        if errors:
            raise PadConfigurationError("; ".join(errors))

        self._clip = _ClipBuilder(settings=self.settings)
        self._scorer = resolve_passive_scorer(self.settings.passive_model_id)
        self._active = _ActiveChallengeController(settings=self.settings)

    def add_frame(self, pad_frame: PadFrame) -> None:
        self._clip.add(pad_frame)

        if self._active.challenge and self._active.challenge.state == ChallengeState.IN_PROGRESS:
            self._active.update(pad_frame)

    def needs_more_frames(self) -> bool:
        return not self._clip.is_ready()

    def reset(self) -> None:
        self._clip.clear()
        self._active.challenge = None
        self._pending_challenge_type = None

    def current_challenge(self) -> PadChallenge | None:
        return self._active.challenge

    def evaluate_final(self) -> PadResult:
        started = time.monotonic()

        if not self.settings.enabled:
            return self._result(
                outcome=PadOutcome.LIVE,
                reason_code=REASON_PAD_DISABLED,
                user_message_key=MSG_VERIFYING,
                passive_score=None,
                confidence_band=None,
                active_passed=None,
                frames_analyzed=self._clip.count(),
                started=started,
            )

        if not self._clip.is_ready():
            return self._result(
                outcome=PadOutcome.INCONCLUSIVE,
                reason_code=REASON_TOO_FEW_FRAMES,
                user_message_key=MSG_LIVENESS_INCONCLUSIVE,
                passive_score=None,
                confidence_band=ConfidenceBand.LOW,
                active_passed=None,
                frames_analyzed=self._clip.count(),
                started=started,
            )

        try:
            passive = self._scorer.score(self._clip.frames())
        except Exception as error:
            log_event(LOGGER, "pad_error", level=logging.ERROR, error_type=type(error).__name__)
            return self._result(
                outcome=PadOutcome.ERROR,
                reason_code=REASON_PAD_ERROR,
                user_message_key=MSG_LIVENESS_INCONCLUSIVE,
                passive_score=None,
                confidence_band=None,
                active_passed=None,
                frames_analyzed=self._clip.count(),
                started=started,
            )

        score = passive.live_probability
        band = _confidence_band(score, self.settings)
        mode = parse_pad_mode(self.settings.mode)

        log_event(
            LOGGER,
            "pad_passive_evaluated",
            level=logging.DEBUG,
            session_id=self.session_id,
            confidence_band=band.value,
            scorer_id=passive.scorer_id,
            log_score=self.settings.log_passive_scores,
            live_probability=f"{score:.3f}" if self.settings.log_passive_scores else "redacted",
        )

        if mode == PadMode.ACTIVE_REQUIRED:
            return self._finalize_with_active(score, band, passive, started, force=True)

        passive_outcome = _outcome_from_passive(score, self.settings)

        if mode == PadMode.PASSIVE_ONLY:
            return self._finalize_passive_only(passive_outcome, score, band, passive, started)

        # Borderline → active challenge when inconclusive or medium band
        if passive_outcome == PadOutcome.INCONCLUSIVE or (
            self.settings.borderline_low <= score <= self.settings.borderline_high
        ):
            return self._finalize_with_active(score, band, passive, started, force=False)

        return self._finalize_passive_only(passive_outcome, score, band, passive, started)

    def _finalize_passive_only(
        self,
        outcome: PadOutcome,
        score: float,
        band: ConfidenceBand,
        passive: PassiveScoreResult,
        started: float,
    ) -> PadResult:
        if outcome == PadOutcome.LIVE:
            reason = REASON_PASSIVE_LIVE
            message = MSG_VERIFYING
        elif outcome == PadOutcome.ATTACK:
            reason = REASON_PASSIVE_ATTACK
            message = MSG_LIVENESS_FAILED
        else:
            reason = REASON_PASSIVE_INCONCLUSIVE
            message = MSG_LIVENESS_INCONCLUSIVE

        return self._result(
            outcome=outcome,
            reason_code=reason,
            user_message_key=message,
            passive_score=score,
            confidence_band=band,
            active_passed=None,
            frames_analyzed=self._clip.count(),
            started=started,
            model_id=passive.scorer_id,
            model_version=passive.scorer_version,
        )

    def _finalize_with_active(
        self,
        score: float,
        band: ConfidenceBand,
        passive: PassiveScoreResult,
        started: float,
        force: bool,
    ) -> PadResult:
        challenge_type = self._resolve_challenge_type()
        if self._active.challenge is None or self._active.challenge.state in {
            ChallengeState.PENDING,
            ChallengeState.EXPIRED,
            ChallengeState.FAILED,
        }:
            challenge = self._active.start(challenge_type)
            log_event(
                LOGGER,
                "pad_challenge_started",
                session_id=self.session_id,
                challenge_id=challenge.challenge_id,
                instruction_key=challenge.instruction_key,
            )
            if force and self._clip.count() > 0:
                self._active.update(self._clip.frames()[-1])

        challenge = self._active.challenge
        if challenge is None:
            return self._result(
                outcome=PadOutcome.ERROR,
                reason_code=REASON_PAD_ERROR,
                user_message_key=MSG_LIVENESS_INCONCLUSIVE,
                passive_score=score,
                confidence_band=band,
                active_passed=False,
                frames_analyzed=self._clip.count(),
                started=started,
                model_id=passive.scorer_id,
                model_version=passive.scorer_version,
            )

        if challenge.state == ChallengeState.PASSED:
            log_event(
                LOGGER,
                "pad_challenge_completed",
                session_id=self.session_id,
                challenge_id=challenge.challenge_id,
                passed=True,
            )
            return self._result(
                outcome=PadOutcome.LIVE,
                reason_code=REASON_ACTIVE_PASSED,
                user_message_key=MSG_VERIFYING,
                passive_score=score,
                confidence_band=band,
                active_passed=True,
                frames_analyzed=self._clip.count(),
                started=started,
                model_id=passive.scorer_id,
                model_version=passive.scorer_version,
                challenge_id=challenge.challenge_id,
            )

        if challenge.state in {ChallengeState.EXPIRED, ChallengeState.FAILED}:
            log_event(
                LOGGER,
                "pad_challenge_completed",
                session_id=self.session_id,
                challenge_id=challenge.challenge_id,
                passed=False,
            )
            return self._result(
                outcome=PadOutcome.ATTACK,
                reason_code=REASON_ACTIVE_TIMEOUT,
                user_message_key=MSG_CHALLENGE_TIMEOUT,
                passive_score=score,
                confidence_band=band,
                active_passed=False,
                frames_analyzed=self._clip.count(),
                started=started,
                model_id=passive.scorer_id,
                model_version=passive.scorer_version,
                challenge_id=challenge.challenge_id,
            )

        # IN_PROGRESS or just started — caller should feed more frames before calling again
        return self._result(
            outcome=PadOutcome.INCONCLUSIVE,
            reason_code=REASON_PASSIVE_INCONCLUSIVE,
            user_message_key=MSG_CHALLENGE_PROMPT,
            passive_score=score,
            confidence_band=band,
            active_passed=False,
            frames_analyzed=self._clip.count(),
            started=started,
            model_id=passive.scorer_id,
            model_version=passive.scorer_version,
            challenge_id=challenge.challenge_id,
        )

    def _resolve_challenge_type(self) -> ChallengeType:
        challenge_name = str(self.settings.challenge_type).lower()
        if challenge_name == "head_turn_right":
            return ChallengeType.HEAD_TURN_RIGHT
        return ChallengeType.HEAD_TURN_LEFT

    def _result(
        self,
        outcome: PadOutcome,
        reason_code: str,
        user_message_key: str,
        passive_score: float | None,
        confidence_band: ConfidenceBand | None,
        active_passed: bool | None,
        frames_analyzed: int,
        started: float,
        model_id: str | None = None,
        model_version: str | None = None,
        challenge_id: str | None = None,
    ) -> PadResult:
        meta = self._scorer.metadata()
        latency_ms = (time.monotonic() - started) * 1000.0
        result = PadResult(
            outcome=outcome,
            reason_code=reason_code,
            user_message_key=user_message_key,
            model_id=model_id or meta.model_id,
            model_version=model_version or meta.model_version,
            frames_analyzed=frames_analyzed,
            latency_ms=latency_ms,
            passive_score=passive_score,
            confidence_band=confidence_band,
            active_passed=active_passed,
            challenge_id=challenge_id,
            session_id=self.session_id,
        )
        log_event(
            LOGGER,
            "pad_decision",
            level=logging.INFO if outcome == PadOutcome.LIVE else logging.WARNING,
            session_id=self.session_id,
            outcome=outcome.value,
            reason_code=reason_code,
            model_version=result.model_version,
            latency_ms=f"{latency_ms:.1f}",
        )
        return result


def create_pad_session(settings: Any | None = None) -> PadSession:
    return PadSession(settings=settings or PAD_SETTINGS)


def run_pad_gate(frames: Sequence[PadFrame], settings: Any | None = None) -> PadResult:
    session = create_pad_session(settings=settings)
    for pad_frame in frames:
        session.add_frame(pad_frame)
    return session.evaluate_final()


def check_single_image_pad(
    image: Any,
    settings: Any | None = None,
    frame_index: int = 1,
) -> PadResult:
    """Run PAD on a single frame (registration or simple recognition path)."""
    pad_settings = settings or PAD_SETTINGS
    burst = max(1, pad_settings.registration_pad_burst_frames)
    frames = [
        build_pad_frame(image=image, frame_index=frame_index + offset)
        for offset in range(burst)
    ]
    return run_pad_gate(frames, settings=pad_settings)


__all__ = [
    "PadOutcome",
    "PadMode",
    "ConfidenceBand",
    "PadFrame",
    "PadResult",
    "PadChallenge",
    "PadSession",
    "PadConfigurationError",
    "MockPassiveScorer",
    "HeuristicPassiveScorer",
    "create_pad_session",
    "run_pad_gate",
    "check_single_image_pad",
    "build_pad_frame",
    "validate_pad_settings",
    "resolve_passive_scorer",
]
