"""
Unit tests for src.utils.

Filesystem helpers use pytest's tmp_path; no project data/ or encodings/ writes.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from src.utils import (
    build_path,
    clean_name,
    directory_exists,
    ensure_directory,
    file_exists,
    generate_image_filename,
    load_pickle,
    print_status,
    save_pickle,
)


@pytest.mark.unit
class TestCleanName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Alice Smith", "alice_smith"),
            ("  BOB  ", "bob"),
            ("user@#1!", "user1"),
            ("!!!", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_clean_name_normalizes_input(self, raw: str, expected: str) -> None:
        assert clean_name(raw) == expected


@pytest.mark.unit
class TestEnsureDirectory:
    def test_ensure_directory_creates_nested_path(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c"

        result = ensure_directory(target)

        assert result == target
        assert target.is_dir()

    def test_ensure_directory_is_idempotent(self, tmp_path: Path) -> None:
        target = tmp_path / "repeat"
        ensure_directory(target)
        ensure_directory(target)

        assert target.is_dir()


@pytest.mark.unit
class TestPathHelpers:
    def test_file_and_directory_exist_checks(self, tmp_path: Path) -> None:
        file_path = tmp_path / "sample.txt"
        file_path.write_text("x", encoding="utf-8")

        assert file_exists(file_path) is True
        assert directory_exists(file_path) is False
        assert directory_exists(tmp_path) is True
        assert file_exists(tmp_path / "missing.txt") is False

    def test_build_path_joins_under_project_root(self) -> None:
        result = build_path("data", "user", "face.jpg")

        assert result.name == "face.jpg"
        assert result.parts[-3:] == ("data", "user", "face.jpg")


@pytest.mark.unit
class TestPickleHelpers:
    def test_save_and_load_pickle_round_trip(self, tmp_path: Path) -> None:
        payload = {"user_name": "alice", "embeddings": [[0.1, 0.2]]}
        file_path = tmp_path / "nested" / "record.pkl"

        save_pickle(payload, file_path)
        loaded = load_pickle(file_path)

        assert loaded == payload

    def test_load_pickle_returns_default_when_missing(self, tmp_path: Path) -> None:
        assert load_pickle(tmp_path / "missing.pkl", default=[]) == []

    def test_load_pickle_raises_when_missing_and_no_default(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pickle(tmp_path / "missing.pkl")

    def test_save_pickle_writes_binary_pickle(self, tmp_path: Path) -> None:
        file_path = tmp_path / "raw.pkl"
        save_pickle({"ok": True}, file_path)

        with file_path.open("rb") as handle:
            assert pickle.load(handle) == {"ok": True}


@pytest.mark.unit
class TestGenerateImageFilename:
    def test_generate_image_filename_uses_safe_username_and_extension(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FixedDatetime:
            @classmethod
            def now(cls):
                from datetime import datetime

                return datetime(2026, 5, 30, 12, 0, 0, 123456)

        monkeypatch.setattr("src.utils.datetime", FixedDatetime)

        filename = generate_image_filename("Alice Smith", extension=".png")

        assert filename == "alice_smith_20260530_120000_123456.png"


@pytest.mark.unit
class TestPrintStatus:
    def test_print_status_formats_known_levels(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_status("hello", level="success")

        captured = capsys.readouterr().out
        assert "[OK] hello" in captured

    def test_print_status_falls_back_to_info_for_unknown_level(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        print_status("hello", level="verbose")

        captured = capsys.readouterr().out
        assert "[INFO] hello" in captured
