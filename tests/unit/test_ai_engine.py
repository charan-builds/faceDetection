"""
Unit tests for pure math and comparison logic in src.ai_engine.

No DeepFace, TensorFlow, webcam, or real face embeddings are used.
Vectors are small, hand-picked, and fully predictable.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

from src.ai_engine import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_DISTANCE_METRIC,
    compare_embeddings,
    cosine_distance,
    cosine_similarity,
    euclidean_distance,
    find_best_match,
)
from src.config import AI_SETTINGS


# --- Deterministic test vectors (not real face embeddings) ---

# Unit vector along X axis.
VECTOR_A = [1.0, 0.0, 0.0]

# Same direction as VECTOR_A -> identical match.
VECTOR_A_COPY = [1.0, 0.0, 0.0]

# Unit vector along Y axis -> orthogonal to A (maximally different direction).
VECTOR_B = [0.0, 1.0, 0.0]

# Unit vector at ~45° between A and B in the X-Y plane (similarity = 0.7, distance = 0.3).
VECTOR_NEAR_A = [0.7, 0.7141428428557129, 0.0]

# Slightly farther from A than VECTOR_NEAR_A (distance just above 0.30).
VECTOR_PAST_THRESHOLD = [0.69, 0.723573427377772, 0.0]


@pytest.fixture
def threshold() -> float:
    """Default project cosine distance threshold."""
    return AI_SETTINGS.cosine_threshold


@pytest.fixture
def make_user_record() -> Callable[..., dict[str, Any]]:
    """
    Build a minimal known-user record like those loaded from encodings/.
    """

    def _factory(user_name: str, embeddings: list[list[float]]) -> dict[str, Any]:
        return {
            "user_name": user_name,
            "model_name": "test-model",
            "embeddings": embeddings,
        }

    return _factory


@pytest.mark.unit
class TestCosineSimilarity:
    def test_identical_vectors_have_similarity_one(self) -> None:
        """Same direction and length -> perfect similarity."""
        assert cosine_similarity(VECTOR_A, VECTOR_A_COPY) == pytest.approx(1.0)

    def test_orthogonal_vectors_have_similarity_zero(self) -> None:
        """Perpendicular unit vectors share no directional overlap."""
        assert cosine_similarity(VECTOR_A, VECTOR_B) == pytest.approx(0.0)

    def test_completely_opposite_vectors_have_similarity_minus_one(self) -> None:
        """Opposite direction is the strongest possible disagreement."""
        opposite = [-1.0, 0.0, 0.0]
        assert cosine_similarity(VECTOR_A, opposite) == pytest.approx(-1.0)

    def test_empty_vector_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Embedding cannot be empty"):
            cosine_similarity([], VECTOR_A)

    def test_zero_vector_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Embedding norm cannot be zero"):
            cosine_similarity([0.0, 0.0, 0.0], VECTOR_A)

    def test_mismatched_dimensions_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="same number of dimensions"):
            cosine_similarity(VECTOR_A, [1.0, 0.0])

    def test_non_numeric_values_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="only numeric"):
            cosine_similarity(["a", "b", "c"], VECTOR_A)

    def test_two_dimensional_embedding_raises_value_error(self) -> None:
        """Embeddings must be 1-D vectors, not matrices."""
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="one-dimensional"):
            cosine_similarity(matrix, VECTOR_A)


@pytest.mark.unit
class TestCosineDistance:
    def test_identical_vectors_have_zero_distance(self) -> None:
        assert cosine_distance(VECTOR_A, VECTOR_A_COPY) == pytest.approx(0.0)

    def test_orthogonal_vectors_have_distance_one(self) -> None:
        assert cosine_distance(VECTOR_A, VECTOR_B) == pytest.approx(1.0)

    def test_distance_is_one_minus_similarity(self) -> None:
        similarity = cosine_similarity(VECTOR_NEAR_A, VECTOR_A)
        distance = cosine_distance(VECTOR_NEAR_A, VECTOR_A)
        assert distance == pytest.approx(1.0 - similarity)


@pytest.mark.unit
class TestEuclideanDistance:
    def test_identical_vectors_have_zero_euclidean_distance(self) -> None:
        assert euclidean_distance(VECTOR_A, VECTOR_A_COPY) == pytest.approx(0.0)

    def test_orthogonal_unit_vectors_have_unit_euclidean_distance(self) -> None:
        assert euclidean_distance(VECTOR_A, VECTOR_B) == pytest.approx(1.41421356, rel=1e-5)

    def test_empty_vector_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Embedding cannot be empty"):
            euclidean_distance([], VECTOR_A)


@pytest.mark.unit
class TestCompareEmbeddings:
    def test_identical_embeddings_match(self, threshold: float) -> None:
        result = compare_embeddings(VECTOR_A, VECTOR_A_COPY, threshold=threshold)

        assert result["is_match"] is True
        assert result["cosine_distance"] == pytest.approx(0.0)
        assert result["cosine_similarity"] == pytest.approx(1.0)
        assert result["threshold"] == threshold
        assert result["distance_metric"] == DEFAULT_DISTANCE_METRIC

    def test_orthogonal_embeddings_do_not_match_default_threshold(
        self, threshold: float
    ) -> None:
        result = compare_embeddings(VECTOR_A, VECTOR_B, threshold=threshold)

        assert result["is_match"] is False
        assert result["cosine_distance"] == pytest.approx(1.0)

    def test_distance_exactly_at_threshold_counts_as_match(self) -> None:
        """
        Access rule is distance <= threshold, so equality must grant access.
        """
        custom_threshold = 0.30
        result = compare_embeddings(VECTOR_A, VECTOR_NEAR_A, threshold=custom_threshold)

        assert result["cosine_distance"] == pytest.approx(0.30, abs=1e-6)
        assert result["is_match"] is True

    def test_distance_just_above_threshold_denies_match(self) -> None:
        custom_threshold = 0.30
        result = compare_embeddings(
            VECTOR_A, VECTOR_PAST_THRESHOLD, threshold=custom_threshold
        )

        assert result["cosine_distance"] > custom_threshold
        assert result["is_match"] is False

    def test_invalid_embedding_propagates_value_error(self) -> None:
        with pytest.raises(ValueError):
            compare_embeddings([], VECTOR_A)


@pytest.mark.unit
class TestFindBestMatch:
    def test_no_registered_embeddings_returns_fail_closed(self, threshold: float) -> None:
        result = find_best_match(
            candidate_embedding=VECTOR_A,
            known_records=[],
            threshold=threshold,
        )

        assert result["is_match"] is False
        assert result["user_name"] is None
        assert result["cosine_distance"] is None
        assert result["reason"] == "no_registered_embeddings"

    def test_closest_user_wins_among_multiple_candidates(
        self, make_user_record: Callable[..., dict[str, Any]], threshold: float
    ) -> None:
        """
        User B is closer to the candidate than User A, so B should be returned.
        """
        records = [
            make_user_record("alice", [VECTOR_B]),
            make_user_record("bob", [VECTOR_A]),
        ]

        result = find_best_match(
            candidate_embedding=VECTOR_A,
            known_records=records,
            threshold=threshold,
        )

        assert result["user_name"] == "bob"
        assert result["cosine_distance"] == pytest.approx(0.0)
        assert result["is_match"] is True

    def test_best_embedding_per_user_is_considered(
        self, make_user_record: Callable[..., dict[str, Any]], threshold: float
    ) -> None:
        """
        One user may store several embeddings; the closest of any image should win.
        """
        records = [
            make_user_record(
                "alice",
                [
                    VECTOR_B,  # far from candidate
                    VECTOR_A,  # exact match
                ],
            ),
            make_user_record("bob", [VECTOR_NEAR_A]),
        ]

        result = find_best_match(
            candidate_embedding=VECTOR_A,
            known_records=records,
            threshold=threshold,
        )

        assert result["user_name"] == "alice"
        assert result["is_match"] is True
        assert result["cosine_distance"] == pytest.approx(0.0)

    def test_no_match_when_closest_face_exceeds_threshold(
        self, make_user_record: Callable[..., dict[str, Any]]
    ) -> None:
        """
        A closest user is still reported, but access is denied when distance > threshold.
        """
        strict_threshold = 0.10
        records = [make_user_record("alice", [VECTOR_B])]

        result = find_best_match(
            candidate_embedding=VECTOR_A,
            known_records=records,
            threshold=strict_threshold,
        )

        assert result["user_name"] == "alice"
        assert result["cosine_distance"] == pytest.approx(1.0)
        assert result["is_match"] is False

    def test_three_users_picks_global_minimum_distance(
        self, make_user_record: Callable[..., dict[str, Any]], threshold: float
    ) -> None:
        records = [
            make_user_record("far", [VECTOR_B]),
            make_user_record("medium", [VECTOR_NEAR_A]),
            make_user_record("close", [VECTOR_A]),
        ]

        result = find_best_match(
            candidate_embedding=VECTOR_A,
            known_records=records,
            threshold=threshold,
        )

        assert result["user_name"] == "close"
        assert result["is_match"] is True

    def test_invalid_candidate_embedding_raises_value_error(
        self, make_user_record: Callable[..., dict[str, Any]]
    ) -> None:
        records = [make_user_record("alice", [VECTOR_A])]

        with pytest.raises(ValueError, match="Embedding cannot be empty"):
            find_best_match(
                candidate_embedding=[],
                known_records=records,
                threshold=DEFAULT_COSINE_THRESHOLD,
            )

    def test_invalid_stored_embedding_raises_value_error(
        self, make_user_record: Callable[..., dict[str, Any]]
    ) -> None:
        records = [make_user_record("alice", [[]])]

        with pytest.raises(ValueError, match="Embedding cannot be empty"):
            find_best_match(
                candidate_embedding=VECTOR_A,
                known_records=records,
                threshold=DEFAULT_COSINE_THRESHOLD,
            )


# ---------------------------------------------------------------------------
# load_face_model warm-up — unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadFaceModel:
    """
    Tests for src.ai_engine.load_face_model.

    DeepFace is never imported here.  We patch _get_deepface at the call-site
    to return a controllable mock, so these tests are instant and offline.
    """

    def test_returns_model_object_on_success(self) -> None:
        """A successful build_model call returns whatever DeepFace gives back."""
        from unittest.mock import MagicMock, patch

        mock_model = MagicMock(name="FaceModel")
        mock_deepface = MagicMock()
        mock_deepface.build_model.return_value = mock_model

        with patch("src.ai_engine._get_deepface", return_value=mock_deepface):
            from src.ai_engine import load_face_model

            result = load_face_model("Facenet512")

        assert result is mock_model

    def test_calls_build_model_with_task_argument(self) -> None:
        """Modern DeepFace receives model_name and task='facial_recognition'."""
        from unittest.mock import MagicMock, patch

        mock_deepface = MagicMock()
        mock_deepface.build_model.return_value = MagicMock()

        with patch("src.ai_engine._get_deepface", return_value=mock_deepface):
            from src.ai_engine import load_face_model

            load_face_model("Facenet512")

        mock_deepface.build_model.assert_called_once_with(
            model_name="Facenet512", task="facial_recognition"
        )

    def test_falls_back_when_task_argument_raises_type_error(self) -> None:
        """Older DeepFace raises TypeError for unknown kwargs; fallback omits task."""
        from unittest.mock import MagicMock, call, patch

        fallback_model = MagicMock(name="FallbackModel")
        mock_deepface = MagicMock()
        mock_deepface.build_model.side_effect = [TypeError("unexpected keyword"), fallback_model]

        with patch("src.ai_engine._get_deepface", return_value=mock_deepface):
            from src.ai_engine import load_face_model

            result = load_face_model("Facenet512")

        assert result is fallback_model
        assert mock_deepface.build_model.call_count == 2
        # Second call must be the positional-only form
        assert mock_deepface.build_model.call_args_list[1] == call("Facenet512")

    def test_raises_runtime_error_on_unexpected_exception(self) -> None:
        """Any non-TypeError from build_model is re-raised as RuntimeError."""
        from unittest.mock import MagicMock, patch

        mock_deepface = MagicMock()
        mock_deepface.build_model.side_effect = OSError("weights file missing")

        with patch("src.ai_engine._get_deepface", return_value=mock_deepface):
            from src.ai_engine import load_face_model

            with pytest.raises(RuntimeError, match="Could not load DeepFace model"):
                load_face_model("Facenet512")

    def test_uses_default_model_name_when_none_given(self) -> None:
        """Calling with no arguments uses DEFAULT_MODEL_NAME."""
        from unittest.mock import MagicMock, patch

        from src.ai_engine import DEFAULT_MODEL_NAME

        mock_deepface = MagicMock()
        mock_deepface.build_model.return_value = MagicMock()

        with patch("src.ai_engine._get_deepface", return_value=mock_deepface):
            from src.ai_engine import load_face_model

            load_face_model()

        call_kwargs = mock_deepface.build_model.call_args
        assert call_kwargs.kwargs.get("model_name") == DEFAULT_MODEL_NAME


# ---------------------------------------------------------------------------
# app.warmup_model — unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWarmupModel:
    """
    Tests for app.warmup_model.

    Patches src.ai_engine.load_face_model at the app module boundary so no
    DeepFace download or real model load occurs.
    """

    def test_success_prints_ready_message(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Happy path: load_face_model succeeds → 'ready' status message."""
        from unittest.mock import MagicMock, patch

        with patch("app.load_face_model", return_value=MagicMock()):
            from app import warmup_model

            warmup_model()

        captured = capsys.readouterr()
        assert "ready" in captured.out.lower() or "ok" in captured.out.lower()

    def test_failure_prints_warning_not_error(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """When load_face_model raises, warmup_model warns and does not re-raise."""
        from unittest.mock import patch

        with patch(
            "app.load_face_model",
            side_effect=RuntimeError("weights unavailable"),
        ):
            from app import warmup_model

            # Must not raise — warm-up failure is non-fatal.
            warmup_model()

        captured = capsys.readouterr()
        assert "warning" in captured.out.lower() or "skipped" in captured.out.lower()

    def test_failure_does_not_raise(self) -> None:
        """Any exception from load_face_model is swallowed by warmup_model."""
        from unittest.mock import patch

        with patch("app.load_face_model", side_effect=Exception("unexpected")):
            from app import warmup_model

            # If this raises, the test fails.
            warmup_model()

    def test_passes_model_name_to_load_face_model(self) -> None:
        """warmup_model forwards the model_name argument."""
        from unittest.mock import MagicMock, patch

        with patch("app.load_face_model", return_value=MagicMock()) as mock_load:
            from app import warmup_model

            warmup_model("SomeOtherModel")

        mock_load.assert_called_once_with("SomeOtherModel")

    def test_uses_default_model_name_when_none_given(self) -> None:
        """Calling warmup_model() with no args uses DEFAULT_MODEL_NAME."""
        from unittest.mock import MagicMock, patch

        from app import DEFAULT_MODEL_NAME

        with patch("app.load_face_model", return_value=MagicMock()) as mock_load:
            from app import warmup_model

            warmup_model()

        mock_load.assert_called_once_with(DEFAULT_MODEL_NAME)

    def test_run_app_calls_warmup_model(self) -> None:
        """run_app must call warmup_model exactly once before showing the menu."""
        from unittest.mock import MagicMock, call, patch

        # Stop the menu loop immediately after the first iteration.
        with (
            patch("app.ensure_project_folders"),
            patch("app.warmup_model") as mock_warmup,
            patch("app.show_banner"),
            patch("app.show_menu"),
            patch("app.get_menu_choice", return_value="3"),  # EXIT on first choice
            patch("app.print_status"),
        ):
            from app import run_app

            run_app()

        mock_warmup.assert_called_once()


# ---------------------------------------------------------------------------
# load_all_user_embeddings resilience tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadAllResiliency:
    """
    load_all_user_embeddings must skip bad files and continue, never abort-all.
    Integrity checks are NOT weakened — tampered files are detected and skipped,
    not silently accepted.
    """

    _SAMPLE_RECORD = {
        "user_name": "alice",
        "safe_name": "alice",
        "model_name": "Facenet512",
        "detector_backend": "opencv",
        "distance_metric": "cosine",
        "threshold": 0.30,
        "embeddings": [[0.1, 0.2, 0.3]],
        "embedding_count": 1,
        "image_paths": [],
        "created_at": "2026-06-01T12:00:00",
    }

    @pytest.fixture()
    def signed_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FACE_LOCK_SIGNING_KEY",
                           "deadbeefdeadbeefdeadbeefdeadbeef0102030405060708")

    @pytest.fixture()
    def unsigned_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FACE_LOCK_SIGNING_KEY", raising=False)

    @pytest.fixture()
    def log_capture(self, caplog: pytest.LogCaptureFixture):
        import logging as _logging
        face_lock = _logging.getLogger("face_lock")
        face_lock.addHandler(caplog.handler)
        face_lock.setLevel(_logging.DEBUG)
        yield caplog
        face_lock.removeHandler(caplog.handler)

    def _write_valid_signed(
        self, path: Path, user_name: str, signed_env: None
    ) -> None:
        import src.ai_engine as ae
        ae.save_user_embeddings(user_name, [[0.1, 0.2, 0.3]], output_path=path)

    def _write_corrupt(self, path: Path) -> None:
        path.write_bytes(b"\x80\x04corrupt-garbage-bytes-not-a-pickle")

    def _write_legacy(self, path: Path) -> None:
        import pickle
        import json
        rec = json.loads(json.dumps(self._SAMPLE_RECORD))
        with path.open("wb") as f:
            pickle.dump(rec, f)

    def _write_tampered(self, path: Path, signed_env: None) -> None:
        from src.integrity import wrap_record_for_storage
        from src.utils import save_pickle
        import json

        rec = json.loads(json.dumps(self._SAMPLE_RECORD))
        payload = wrap_record_for_storage(rec)
        payload["record"]["user_name"] = "mallory"  # tamper after signing
        save_pickle(payload, path)

    def test_corrupt_file_is_skipped(
        self, tmp_path: Path, signed_env: None, log_capture
    ) -> None:
        import src.ai_engine as ae
        self._write_corrupt(tmp_path / "bad.pkl")
        records = ae.load_all_user_embeddings(encodings_dir=tmp_path)
        assert records == []

    def test_corrupt_file_logs_error(
        self, tmp_path: Path, signed_env: None, log_capture
    ) -> None:
        import logging as _logging
        import src.ai_engine as ae
        self._write_corrupt(tmp_path / "bad.pkl")
        with log_capture.at_level(_logging.ERROR, logger="face_lock"):
            ae.load_all_user_embeddings(encodings_dir=tmp_path)
        assert any("embedding_load_skipped" in r.message for r in log_capture.records)

    def test_invalid_signature_is_skipped_not_raised(
        self, tmp_path: Path, signed_env: None, log_capture
    ) -> None:
        import src.ai_engine as ae
        self._write_tampered(tmp_path / "tampered.pkl", signed_env)
        # Must NOT raise — just skip
        records = ae.load_all_user_embeddings(encodings_dir=tmp_path)
        assert records == []

    def test_legacy_file_loads_with_warning(
        self, tmp_path: Path, unsigned_env: None, log_capture
    ) -> None:
        import logging as _logging
        import src.ai_engine as ae
        self._write_legacy(tmp_path / "legacy.pkl")
        with log_capture.at_level(_logging.WARNING, logger="face_lock"):
            records = ae.load_all_user_embeddings(encodings_dir=tmp_path)
        assert len(records) == 1
        assert records[0]["user_name"] == "alice"

    def test_valid_signed_file_loads(
        self, tmp_path: Path, signed_env: None
    ) -> None:
        import src.ai_engine as ae
        self._write_valid_signed(tmp_path / "alice.pkl", "alice", signed_env)
        records = ae.load_all_user_embeddings(encodings_dir=tmp_path)
        assert len(records) == 1
        assert records[0]["user_name"] == "alice"

    def test_one_corrupt_one_valid_returns_valid(
        self, tmp_path: Path, signed_env: None, log_capture
    ) -> None:
        import src.ai_engine as ae
        self._write_corrupt(tmp_path / "aaa_bad.pkl")   # sorts first
        self._write_valid_signed(tmp_path / "zzz_alice.pkl", "alice", signed_env)
        records = ae.load_all_user_embeddings(encodings_dir=tmp_path)
        assert len(records) == 1
        assert records[0]["user_name"] == "alice"

    def test_partial_load_logs_warning(
        self, tmp_path: Path, signed_env: None, log_capture
    ) -> None:
        import logging as _logging
        import src.ai_engine as ae
        self._write_corrupt(tmp_path / "aaa_bad.pkl")
        self._write_valid_signed(tmp_path / "zzz_alice.pkl", "alice", signed_env)
        with log_capture.at_level(_logging.WARNING, logger="face_lock"):
            ae.load_all_user_embeddings(encodings_dir=tmp_path)
        assert any("embedding_load_partial" in r.message for r in log_capture.records)

    def test_all_corrupt_returns_empty_list(
        self, tmp_path: Path, signed_env: None
    ) -> None:
        import src.ai_engine as ae
        for i in range(3):
            (tmp_path / f"bad_{i}.pkl").write_bytes(b"garbage")
        records = ae.load_all_user_embeddings(encodings_dir=tmp_path)
        assert records == []

    def test_integrity_check_not_weakened(
        self, tmp_path: Path, signed_env: None, log_capture
    ) -> None:
        """Tampered file is detected (skipped) — NOT silently accepted."""
        import src.ai_engine as ae
        self._write_tampered(tmp_path / "tampered.pkl", signed_env)
        records = ae.load_all_user_embeddings(encodings_dir=tmp_path)
        # Tampered record must NOT appear in the loaded list
        assert not any(r.get("user_name") == "mallory" for r in records)
