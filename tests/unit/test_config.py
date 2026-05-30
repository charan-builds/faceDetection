"""
Unit tests for src.config.

Config is pure data + path resolution; no I/O beyond Path resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    AI_SETTINGS,
    CONFIG,
    DISPLAY_SETTINGS,
    LOGGING_SETTINGS,
    PAD_SETTINGS,
    PATH_SETTINGS,
    RECOGNITION_SETTINGS,
    WEBCAM_SETTINGS,
    resolve_project_root,
)


@pytest.mark.unit
class TestResolveProjectRoot:
    def test_resolve_project_root_points_to_repo_root(self) -> None:
        root = resolve_project_root()

        assert root.is_dir()
        assert (root / "src" / "config.py").is_file()
        assert (root / "app.py").is_file()


@pytest.mark.unit
class TestPathSettings:
    def test_runtime_directories_are_under_project_root(self) -> None:
        root = PATH_SETTINGS.project_root

        for directory in PATH_SETTINGS.runtime_directories:
            assert directory.is_relative_to(root)

    def test_runtime_directory_names_match_convention(self) -> None:
        assert PATH_SETTINGS.data_dir.name == "data"
        assert PATH_SETTINGS.encodings_dir.name == "encodings"
        assert PATH_SETTINGS.logs_dir.name == "logs"
        assert PATH_SETTINGS.models_dir.name == "models"

    def test_custom_project_root_overrides_paths(self, tmp_path: Path) -> None:
        from src.config import PathSettings

        custom = PathSettings(project_root=tmp_path)

        assert custom.data_dir == tmp_path / "data"
        assert custom.encodings_dir == tmp_path / "encodings"


@pytest.mark.unit
class TestAppConfigDefaults:
    def test_singleton_aliases_reference_grouped_config(self) -> None:
        assert CONFIG.ai is AI_SETTINGS
        assert CONFIG.paths is PATH_SETTINGS
        assert CONFIG.recognition is RECOGNITION_SETTINGS
        assert CONFIG.display is DISPLAY_SETTINGS
        assert CONFIG.webcam is WEBCAM_SETTINGS
        assert CONFIG.logging is LOGGING_SETTINGS
        assert CONFIG.pad is PAD_SETTINGS

    def test_pad_settings_defaults(self) -> None:
        assert PAD_SETTINGS.enabled is True
        assert PAD_SETTINGS.passive_model_id == "heuristic_rgb_v1"
        assert (
            PAD_SETTINGS.passive_reject_threshold
            < PAD_SETTINGS.borderline_low
            <= PAD_SETTINGS.borderline_high
            < PAD_SETTINGS.passive_pass_threshold
        )
        assert PAD_SETTINGS.enforce_on_registration is True

    def test_ai_settings_use_cosine_metric(self) -> None:
        assert AI_SETTINGS.distance_metric == "cosine"
        assert 0.0 < AI_SETTINGS.cosine_threshold < 1.0

    def test_recognition_settings_are_sensible(self) -> None:
        assert RECOGNITION_SETTINGS.frame_skip >= 1
        assert RECOGNITION_SETTINGS.cooldown_seconds > 0
        assert RECOGNITION_SETTINGS.brightness_threshold > 0

    def test_display_colors_are_bgr_triplets(self) -> None:
        for color in (
            DISPLAY_SETTINGS.color_granted,
            DISPLAY_SETTINGS.color_denied,
            DISPLAY_SETTINGS.color_info,
            DISPLAY_SETTINGS.color_warning,
            DISPLAY_SETTINGS.color_shadow,
        ):
            assert len(color) == 3
            assert all(0 <= channel <= 255 for channel in color)

    def test_webcam_quit_key_is_single_character(self) -> None:
        assert len(WEBCAM_SETTINGS.quit_key) == 1
