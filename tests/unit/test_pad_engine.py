"""
Unit tests for src.pad_engine (Phases 0–2 scaffold).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.config import PAD_SETTINGS
from src.pad_engine import (
    MSG_CHALLENGE_PROMPT,
    PadConfigurationError,
    PadOutcome,
    PadSession,
    build_pad_frame,
    check_single_image_pad,
    create_pad_session,
    resolve_passive_scorer,
    run_pad_gate,
    validate_pad_settings,
)


pytest.importorskip("numpy")


@pytest.fixture
def bright_frame() -> np.ndarray:
    return np.full((80, 100, 3), 200, dtype=np.uint8)


@pytest.fixture
def dark_frame() -> np.ndarray:
    return np.full((80, 100, 3), 5, dtype=np.uint8)


@pytest.fixture
def mock_live_settings():
    return replace(
        PAD_SETTINGS,
        enabled=True,
        mode="passive_only",
        passive_model_id="mock_always_live",
        min_frames_passive=2,
        max_frames_buffer=10,
    )


@pytest.fixture
def mock_attack_settings():
    return replace(
        PAD_SETTINGS,
        enabled=True,
        mode="passive_only",
        passive_model_id="mock_always_attack",
        min_frames_passive=2,
    )


@pytest.mark.unit
@pytest.mark.pad
class TestValidatePadSettings:
    def test_default_settings_are_valid(self) -> None:
        assert validate_pad_settings(PAD_SETTINGS) == []

    def test_invalid_threshold_order_reports_errors(self) -> None:
        bad = replace(
            PAD_SETTINGS,
            passive_reject_threshold=0.9,
            borderline_low=0.2,
            borderline_high=0.3,
            passive_pass_threshold=0.4,
        )
        assert len(validate_pad_settings(bad)) > 0


@pytest.mark.unit
@pytest.mark.pad
class TestMockPassiveScorer:
    def test_mock_always_live(self, bright_frame: np.ndarray, mock_live_settings) -> None:
        frames = [
            build_pad_frame(bright_frame, frame_index=1),
            build_pad_frame(bright_frame, frame_index=2),
        ]
        result = run_pad_gate(frames, settings=mock_live_settings)
        assert result.outcome == PadOutcome.LIVE

    def test_mock_always_attack(self, bright_frame: np.ndarray, mock_attack_settings) -> None:
        frames = [
            build_pad_frame(bright_frame, frame_index=1),
            build_pad_frame(bright_frame, frame_index=2),
        ]
        result = run_pad_gate(frames, settings=mock_attack_settings)
        assert result.outcome == PadOutcome.ATTACK

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(PadConfigurationError):
            resolve_passive_scorer("unknown_model_xyz")


@pytest.mark.unit
@pytest.mark.pad
class TestPadSession:
    def test_disabled_pad_bypasses(self, bright_frame: np.ndarray) -> None:
        settings = replace(PAD_SETTINGS, enabled=False)
        result = check_single_image_pad(bright_frame, settings=settings)
        assert result.outcome == PadOutcome.LIVE
        assert result.reason_code == "pad_disabled"

    def test_too_few_frames_is_inconclusive(
        self, bright_frame: np.ndarray, mock_live_settings
    ) -> None:
        settings = replace(mock_live_settings, min_frames_passive=5)
        session = create_pad_session(settings)
        session.add_frame(build_pad_frame(bright_frame, frame_index=1))
        result = session.evaluate_final()
        assert result.outcome == PadOutcome.INCONCLUSIVE

    def test_session_reset_clears_buffer(
        self, bright_frame: np.ndarray, mock_live_settings
    ) -> None:
        session = create_pad_session(mock_live_settings)
        session.add_frame(build_pad_frame(bright_frame, 1))
        session.add_frame(build_pad_frame(bright_frame, 2))
        session.evaluate_final()
        session.reset()
        assert session.needs_more_frames() is True


@pytest.mark.unit
@pytest.mark.pad
class TestHeuristicScorer:
    def test_heuristic_returns_probability(self, bright_frame: np.ndarray) -> None:
        settings = replace(
            PAD_SETTINGS,
            enabled=True,
            mode="passive_only",
            passive_model_id="heuristic_rgb_v1",
            min_frames_passive=2,
        )
        frames = [
            build_pad_frame(bright_frame, frame_index=index)
            for index in range(1, 4)
        ]
        result = run_pad_gate(frames, settings=settings)
        assert result.passive_score is not None
        assert 0.0 <= result.passive_score <= 1.0


@pytest.mark.unit
@pytest.mark.pad
class TestActiveChallengeMode:
    def test_borderline_starts_challenge_prompt(
        self, bright_frame: np.ndarray
    ) -> None:
        settings = replace(
            PAD_SETTINGS,
            enabled=True,
            mode="passive_then_active_on_borderline",
            passive_model_id="mock_borderline",
            min_frames_passive=2,
        )
        frames = [
            build_pad_frame(bright_frame, frame_index=1),
            build_pad_frame(bright_frame, frame_index=2),
        ]
        result = run_pad_gate(frames, settings=settings)
        assert result.outcome == PadOutcome.INCONCLUSIVE
        assert result.user_message_key == MSG_CHALLENGE_PROMPT
        assert result.challenge_id is not None
