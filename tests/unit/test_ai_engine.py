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
