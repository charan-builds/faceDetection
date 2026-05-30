"""
AI logic for the Face Lock project.

This module is responsible only for face-recognition intelligence:
- generating face embeddings with DeepFace.represent()
- comparing embedding vectors
- calculating cosine similarity and distance
- applying threshold-based match decisions
- saving and loading embedding records

It does not open the webcam, display windows, or show application menus.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    # Preferred import when running the project from app.py.
    from src.config import AI_SETTINGS, STORAGE_SETTINGS
    from src.utils import (
        ENCODINGS_DIR,
        clean_name,
        ensure_directory,
        load_pickle,
        save_pickle,
    )
except ModuleNotFoundError:
    # Fallback import when running this file directly as: python src/ai_engine.py
    from config import AI_SETTINGS, STORAGE_SETTINGS
    from utils import ENCODINGS_DIR, clean_name, ensure_directory, load_pickle, save_pickle


# Default DeepFace model used by this project.
DEFAULT_MODEL_NAME = AI_SETTINGS.model_name

# OpenCV is lightweight and already part of this project.
DEFAULT_DETECTOR_BACKEND = AI_SETTINGS.detector_backend

# We compare embeddings using cosine distance by default.
DEFAULT_DISTANCE_METRIC = AI_SETTINGS.distance_metric

# DeepFace's pre-tuned Facenet512 + cosine distance threshold.
# Lower distance means more similar. Distance <= threshold means match.
DEFAULT_COSINE_THRESHOLD = AI_SETTINGS.cosine_threshold

# Face detection behavior defaults.
DEFAULT_ENFORCE_DETECTION = AI_SETTINGS.enforce_detection
DEFAULT_ALIGN_FACES = AI_SETTINGS.align_faces


def _get_deepface() -> Any:
    """
    Import DeepFace only when an AI function needs it.

    Returns:
        The DeepFace class/module.

    Raises:
        RuntimeError: If DeepFace is not installed.
    """
    try:
        # Lazy import keeps this module easier to inspect before dependencies install.
        from deepface import DeepFace
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "DeepFace is not installed. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from error

    return DeepFace


def _prepare_image_input(image_input: str | Path | Any) -> str | Any:
    """
    Prepare an image input for DeepFace.

    DeepFace accepts image paths and OpenCV frames. Paths should be strings,
    while frames should be passed through unchanged.
    """
    # If the input is a file path, validate it and convert it to a string.
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)

        # A missing image should fail clearly before DeepFace runs.
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return str(image_path)

    # OpenCV frames are NumPy arrays, so they should be returned unchanged.
    return image_input


def _as_float_list(values: Sequence[float]) -> list[float]:
    """
    Convert an embedding-like sequence into a clean list of floats.
    """
    # Turn the sequence into a list so we can validate its length safely.
    numbers = list(values)

    # Empty embeddings are invalid for face recognition.
    if len(numbers) == 0:
        raise ValueError("Embedding cannot be empty.")

    try:
        # Store all values as regular Python floats for stable pickle files.
        return [float(value) for value in numbers]
    except (TypeError, ValueError) as error:
        raise ValueError("Embedding must contain only numeric values.") from error


def _to_numpy_embedding(embedding: Sequence[float]) -> np.ndarray:
    """
    Convert an embedding into a NumPy vector for math operations.
    """
    # NumPy makes vector math fast and concise.
    vector = np.asarray(_as_float_list(embedding), dtype=np.float32)

    # A face embedding should be a one-dimensional vector.
    if vector.ndim != 1:
        raise ValueError("Embedding must be a one-dimensional vector.")

    return vector


def load_face_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """
    Load or warm up a DeepFace recognition model.

    DeepFace caches models internally, so calling this once early can reduce
    delay during the first recognition attempt.
    """
    DeepFace = _get_deepface()

    try:
        # Newer DeepFace versions accept a task argument.
        return DeepFace.build_model(model_name=model_name, task="facial_recognition")
    except TypeError:
        # Older DeepFace versions accept only the model name.
        return DeepFace.build_model(model_name)
    except Exception as error:
        raise RuntimeError(f"Could not load DeepFace model: {model_name}") from error


def represent_face(
    image_input: str | Path | Any,
    model_name: str = DEFAULT_MODEL_NAME,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND,
    enforce_detection: bool = DEFAULT_ENFORCE_DETECTION,
    align: bool = DEFAULT_ALIGN_FACES,
    require_single_face: bool = True,
) -> dict[str, Any]:
    """
    Generate a face representation using DeepFace.represent().

    Args:
        image_input: Image path or OpenCV frame.
        model_name: DeepFace model name.
        detector_backend: Face detector backend.
        enforce_detection: If True, fail when no face is detected.
        align: If True, align face based on facial landmarks.
        require_single_face: If True, reject images with zero or multiple faces.

    Returns:
        A dictionary containing the embedding and face metadata.
    """
    DeepFace = _get_deepface()

    # Prepare either a path string or an OpenCV frame for DeepFace.
    source = _prepare_image_input(image_input)

    try:
        # represent() runs detect -> align -> normalize -> embedding.
        face_representations = DeepFace.represent(
            img_path=source,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend=detector_backend,
            align=align,
        )
    except ValueError as error:
        raise RuntimeError("No usable face was detected in the image.") from error
    except Exception as error:
        raise RuntimeError("Could not generate a face embedding.") from error

    # DeepFace should return a list of face dictionaries.
    if not isinstance(face_representations, list) or len(face_representations) == 0:
        raise RuntimeError("DeepFace did not return any face representations.")

    # Face lock should normally work with exactly one face in the image.
    if require_single_face and len(face_representations) != 1:
        raise RuntimeError(
            f"Expected exactly one face, but found {len(face_representations)}."
        )

    # Use the first detected face when require_single_face is False.
    face_data = face_representations[0]

    # The embedding is the numeric vector used for identity comparison.
    embedding = face_data.get("embedding")

    if embedding is None:
        raise RuntimeError("DeepFace response did not contain an embedding.")

    return {
        "embedding": _as_float_list(embedding),
        "facial_area": face_data.get("facial_area"),
        "face_confidence": face_data.get("face_confidence"),
        "model_name": model_name,
        "detector_backend": detector_backend,
    }


def generate_embedding(
    image_input: str | Path | Any,
    model_name: str = DEFAULT_MODEL_NAME,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND,
    enforce_detection: bool = DEFAULT_ENFORCE_DETECTION,
    align: bool = DEFAULT_ALIGN_FACES,
) -> list[float]:
    """
    Generate only the embedding vector from an image.
    """
    # Reuse represent_face so face validation stays in one place.
    face_record = represent_face(
        image_input=image_input,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return face_record["embedding"]


def generate_embeddings_from_images(
    image_paths: Iterable[str | Path],
    model_name: str = DEFAULT_MODEL_NAME,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND,
) -> list[list[float]]:
    """
    Generate embeddings from multiple saved images.
    """
    # Store one embedding vector per valid image.
    embeddings: list[list[float]] = []

    for image_path in image_paths:
        # Each image is processed independently, making errors easier to trace.
        embedding = generate_embedding(
            image_input=image_path,
            model_name=model_name,
            detector_backend=detector_backend,
        )
        embeddings.append(embedding)

    return embeddings


def cosine_similarity(
    embedding_a: Sequence[float],
    embedding_b: Sequence[float],
) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Cosine similarity compares vector direction.
    Higher value means the two embeddings point in a more similar direction.
    """
    # Convert both embeddings into NumPy vectors.
    vector_a = _to_numpy_embedding(embedding_a)
    vector_b = _to_numpy_embedding(embedding_b)

    # Both embeddings must come from the same model and have the same length.
    if vector_a.shape != vector_b.shape:
        raise ValueError("Embeddings must have the same number of dimensions.")

    # Norm means vector length.
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # A zero vector cannot be compared by cosine similarity.
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Embedding norm cannot be zero.")

    # Dot product divided by vector lengths gives cosine similarity.
    similarity = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))

    # Floating-point math can very slightly exceed [-1, 1], so clamp it.
    return max(min(similarity, 1.0), -1.0)


def cosine_distance(
    embedding_a: Sequence[float],
    embedding_b: Sequence[float],
) -> float:
    """
    Calculate cosine distance between two embeddings.

    DeepFace thresholds are distance-based, so lower is better.
    """
    # Convert similarity into distance.
    return 1.0 - cosine_similarity(embedding_a, embedding_b)


def euclidean_distance(
    embedding_a: Sequence[float],
    embedding_b: Sequence[float],
) -> float:
    """
    Calculate Euclidean distance between two embeddings.

    This is the straight-line distance between two vector points.
    """
    # Convert both embeddings into NumPy vectors.
    vector_a = _to_numpy_embedding(embedding_a)
    vector_b = _to_numpy_embedding(embedding_b)

    # Both embeddings must come from the same model and have the same length.
    if vector_a.shape != vector_b.shape:
        raise ValueError("Embeddings must have the same number of dimensions.")

    return float(np.linalg.norm(vector_a - vector_b))


def compare_embeddings(
    known_embedding: Sequence[float],
    candidate_embedding: Sequence[float],
    threshold: float = DEFAULT_COSINE_THRESHOLD,
) -> dict[str, Any]:
    """
    Compare a stored embedding with a new candidate embedding.

    Args:
        known_embedding: Stored trusted user's embedding.
        candidate_embedding: New embedding from a face trying to unlock.
        threshold: Maximum cosine distance allowed for a match.

    Returns:
        A dictionary with similarity, distance, threshold, and match decision.
    """
    # Cosine similarity is high when faces are similar.
    similarity = cosine_similarity(known_embedding, candidate_embedding)

    # Cosine distance is low when faces are similar.
    distance = 1.0 - similarity

    return {
        "is_match": distance <= threshold,
        "cosine_similarity": similarity,
        "cosine_distance": distance,
        "threshold": threshold,
        "distance_metric": DEFAULT_DISTANCE_METRIC,
    }


def get_user_embedding_path(user_name: str) -> Path:
    """
    Build the pickle file path for one user's embeddings.
    """
    # clean_name prevents spaces or unsafe characters in filenames.
    safe_name = clean_name(user_name)

    return ENCODINGS_DIR / f"{safe_name}.{STORAGE_SETTINGS.embedding_file_extension}"


def create_embedding_record(
    user_name: str,
    embeddings: Sequence[Sequence[float]],
    image_paths: Iterable[str | Path] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND,
    threshold: float = DEFAULT_COSINE_THRESHOLD,
) -> dict[str, Any]:
    """
    Create a structured record for saving user embeddings.
    """
    # Validate and normalize every embedding before saving.
    clean_embeddings = [_as_float_list(embedding) for embedding in embeddings]

    if len(clean_embeddings) == 0:
        raise ValueError("At least one embedding is required.")

    # Convert image paths to strings so pickle records stay simple.
    saved_image_paths = [str(Path(path)) for path in image_paths or []]

    return {
        "user_name": user_name,
        "safe_name": clean_name(user_name),
        "model_name": model_name,
        "detector_backend": detector_backend,
        "distance_metric": DEFAULT_DISTANCE_METRIC,
        "threshold": threshold,
        "embeddings": clean_embeddings,
        "embedding_count": len(clean_embeddings),
        "image_paths": saved_image_paths,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_user_embeddings(
    user_name: str,
    embeddings: Sequence[Sequence[float]],
    image_paths: Iterable[str | Path] | None = None,
    output_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND,
    threshold: float = DEFAULT_COSINE_THRESHOLD,
) -> Path:
    """
    Save one user's embeddings to a pickle file.
    """
    # Make sure encodings/ exists before writing into it.
    ensure_directory(ENCODINGS_DIR)

    # Build a structured record with useful metadata.
    record = create_embedding_record(
        user_name=user_name,
        embeddings=embeddings,
        image_paths=image_paths,
        model_name=model_name,
        detector_backend=detector_backend,
        threshold=threshold,
    )

    # Use the default per-user path unless a custom path is provided.
    target_path = Path(output_path) if output_path else get_user_embedding_path(user_name)

    return save_pickle(record, target_path)


def load_user_embeddings(
    user_name: str,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load one user's embedding record from a pickle file.
    """
    # Use the default per-user path unless a custom path is provided.
    source_path = Path(input_path) if input_path else get_user_embedding_path(user_name)

    record = load_pickle(source_path)

    if not isinstance(record, dict) or "embeddings" not in record:
        raise ValueError(f"Invalid embedding record: {source_path}")

    return record


def load_all_user_embeddings(
    encodings_dir: str | Path = ENCODINGS_DIR,
) -> list[dict[str, Any]]:
    """
    Load all user embedding records from the encodings folder.
    """
    folder = Path(encodings_dir)

    # If there are no saved encodings yet, return an empty list.
    if not folder.is_dir():
        return []

    # Load every configured embedding file as one user record.
    records: list[dict[str, Any]] = []
    embedding_pattern = f"*.{STORAGE_SETTINGS.embedding_file_extension}"

    for file_path in sorted(folder.glob(embedding_pattern)):
        try:
            record = load_pickle(file_path)
        except Exception as error:
            raise RuntimeError(f"Could not load embedding file: {file_path}") from error

        if not isinstance(record, dict) or "embeddings" not in record:
            raise ValueError(f"Invalid embedding record: {file_path}")

        records.append(record)

    return records


def find_best_match(
    candidate_embedding: Sequence[float],
    known_records: Iterable[dict[str, Any]],
    threshold: float = DEFAULT_COSINE_THRESHOLD,
) -> dict[str, Any]:
    """
    Find the closest registered user for a candidate embedding.

    Args:
        candidate_embedding: New embedding from the current face.
        known_records: Loaded embedding records from encodings/.
        threshold: Maximum cosine distance allowed for access.

    Returns:
        Best match result with user name, distance, similarity, and decision.
    """
    # None means we have not compared against any registered embedding yet.
    best_result: dict[str, Any] | None = None

    for record in known_records:
        # Get user metadata from the record.
        user_name = record.get("user_name", "unknown")
        model_name = record.get("model_name", DEFAULT_MODEL_NAME)

        # Each user can have multiple embeddings from multiple registration images.
        for known_embedding in record.get("embeddings", []):
            comparison = compare_embeddings(
                known_embedding=known_embedding,
                candidate_embedding=candidate_embedding,
                threshold=threshold,
            )

            result = {
                "is_match": comparison["is_match"],
                "user_name": user_name,
                "model_name": model_name,
                "cosine_similarity": comparison["cosine_similarity"],
                "cosine_distance": comparison["cosine_distance"],
                "threshold": comparison["threshold"],
                "distance_metric": comparison["distance_metric"],
            }

            # Keep the closest embedding seen so far.
            if (
                best_result is None
                or result["cosine_distance"] < best_result["cosine_distance"]
            ):
                best_result = result

    # If no known embeddings exist, fail closed.
    if best_result is None:
        return {
            "is_match": False,
            "user_name": None,
            "model_name": DEFAULT_MODEL_NAME,
            "cosine_similarity": None,
            "cosine_distance": None,
            "threshold": threshold,
            "distance_metric": DEFAULT_DISTANCE_METRIC,
            "reason": "no_registered_embeddings",
        }

    # The closest face still needs to pass the threshold.
    best_result["is_match"] = best_result["cosine_distance"] <= threshold

    return best_result
