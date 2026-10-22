"""
Integration and regression tests for Clean 20-Image Owner Re-enrollment.

Verifies:
1. Exactly 20 valid, distinct samples required.
2. Invalid frames (blur, lighting, clipping, duplicates) do not count toward 20.
3. Cancellation and incomplete enrollment preserve the previous template.
4. Embedding dimensions (512-D) and finite-value validation.
5. Backup verification and safe atomic template replacement.
6. Compatibility between enrollment template and 1:1 recognition.
7. Non-owner identities remain rejected.
8. Threshold remains 0.30 and HMAC template integrity remains enforced.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.backup_and_prepare_owner import (
    create_verified_backup,
    inventory_files,
    prepare_clean_owner_environment,
    restore_backup,
    sha256_file,
    verify_backup,
)
from scripts.enroll_owner import enroll_owner
from src.ai_engine import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_MODEL_NAME,
    find_best_match,
    load_user_embeddings,
    save_user_embeddings,
)
from src.config import RECOGNITION_SETTINGS, REGISTRATION_SETTINGS
from src.integrity import IntegrityError, wrap_record_for_storage
from src.pad_engine import PadOutcome, PadResult
from src.recognize import load_trusted_records
from src.register import (
    check_frame_quality,
    detect_single_face,
    get_enrollment_guide,
    is_frame_diverse,
    register_user,
    validate_embedding,
)
from src.utils import save_pickle

FAKE_512_EMBEDDING: list[float] = [0.05] * 512
SIGNING_KEY_HEX: str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"


@pytest.fixture()
def setup_enrollment_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate project paths so tests never touch real data, encodings, or backups."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    data_dir = project_root / "data"
    encodings_dir = project_root / "encodings"
    backups_dir = project_root / "backups"
    data_dir.mkdir()
    encodings_dir.mkdir()
    backups_dir.mkdir()

    # Monkeypatch paths in scripts and modules
    monkeypatch.setattr("scripts.backup_and_prepare_owner.PROJECT_ROOT", project_root)
    monkeypatch.setattr("scripts.backup_and_prepare_owner.DATA_DIR", data_dir)
    monkeypatch.setattr("scripts.backup_and_prepare_owner.ENCODINGS_DIR", encodings_dir)
    monkeypatch.setattr("scripts.backup_and_prepare_owner.BACKUPS_DIR", backups_dir)
    monkeypatch.setattr("src.register.DATA_DIR", data_dir)
    monkeypatch.setattr("src.ai_engine.ENCODINGS_DIR", encodings_dir)
    monkeypatch.setattr("src.utils.ENCODINGS_DIR", encodings_dir)
    monkeypatch.setattr("src.utils.DATA_DIR", data_dir)

    monkeypatch.setenv("FACE_LOCK_SIGNING_KEY", SIGNING_KEY_HEX)
    monkeypatch.delenv("FACE_LOCK_OPERATOR_PIN_HASH", raising=False)

    return {
        "root": project_root,
        "data": data_dir,
        "encodings": encodings_dir,
        "backups": backups_dir,
    }


# ===========================================================================
# 1. Exactly 20 valid, distinct samples required
# ===========================================================================

@pytest.mark.integration
def test_twenty_valid_distinct_samples_required(setup_enrollment_env, monkeypatch):
    """Owner registration must require exactly 20 valid embeddings before succeeding."""
    env = setup_enrollment_env
    owner_name = "owner"
    owner_dir = env["data"] / owner_name
    owner_dir.mkdir()

    # 20 distinct mock image paths
    fake_images = [owner_dir / f"owner_{i:03d}.jpg" for i in range(20)]
    for img in fake_images:
        img.write_bytes(b"mock_jpeg_bytes")

    twenty_embeddings = [[0.01 * (i + 1)] * 512 for i in range(20)]

    with (
        patch("src.register.capture_registration_images", return_value=(fake_images, False)),
        patch("src.register.generate_registration_embeddings", return_value=twenty_embeddings),
        patch("src.register.enforce_strict_mode_policy"),
    ):
        result = register_user(
            user_name=owner_name,
            image_count=20,
            minimum_embeddings=20,
            overwrite=True,
        )

    assert result["success"] is True
    assert result["embedding_count"] == 20
    assert result["image_count"] == 20
    assert (env["encodings"] / f"{owner_name}.pkl").is_file()


# ===========================================================================
# 2. Invalid frames do not count toward 20
# ===========================================================================

@pytest.mark.integration
def test_invalid_frames_do_not_count_toward_twenty():
    """Blurry, dark, clipped, duplicate, and multiple-face frames are rejected."""
    # A. Face detection checks
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    box, err = detect_single_face(blank_frame)
    assert box is None
    assert err == "no_face"

    # Clipped face at top-left edge
    mock_cascade = MagicMock()
    mock_cascade.detectMultiScale.return_value = np.array([[2, 2, 100, 100]])
    box, err = detect_single_face(blank_frame, cascade=mock_cascade)
    assert box is None
    assert err == "face_clipped"

    # Multiple faces
    mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100], [250, 50, 100, 100]])
    box, err = detect_single_face(blank_frame, cascade=mock_cascade)
    assert box is None
    assert err == "multiple_faces"

    # B. Lighting checks
    dark_frame = np.full((100, 100, 3), 10, dtype=np.uint8)
    ok, qerr = check_frame_quality(dark_frame, (10, 10, 80, 80))
    assert ok is False
    assert qerr == "too_dark"

    bright_frame = np.full((100, 100, 3), 250, dtype=np.uint8)
    ok, qerr = check_frame_quality(bright_frame, (10, 10, 80, 80))
    assert ok is False
    assert qerr == "too_bright"

    # C. Duplicate frame detection
    face_crop = np.full((100, 100), 128, dtype=np.uint8)
    frame_with_face = np.full((200, 200, 3), 128, dtype=np.uint8)
    # Identical face should NOT be diverse
    assert is_frame_diverse(frame_with_face, (20, 20, 100, 100), face_crop, min_diff=6.0) is False


# ===========================================================================
# 3. Cancellation and incomplete enrollment preserve the previous template
# ===========================================================================

@pytest.mark.integration
def test_cancellation_preserves_previous_template(setup_enrollment_env):
    """Pressing 'q' during capture leaves the existing template completely untouched."""
    env = setup_enrollment_env
    owner_name = "owner"
    template_path = env["encodings"] / f"{owner_name}.pkl"

    # Seed an existing signed template
    original_embeddings = [[0.1] * 512]
    save_user_embeddings(owner_name, original_embeddings, output_path=template_path)
    original_bytes = template_path.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()

    # Simulate cancellation during capture
    with (
        patch("src.register.capture_registration_images", return_value=([], True)),
        patch("src.register.enforce_strict_mode_policy"),
    ):
        result = register_user(
            user_name=owner_name,
            image_count=20,
            minimum_embeddings=20,
            overwrite=True,
        )

    assert result["success"] is False
    assert result["reason"] == "cancelled"
    # Existing template must be 100% byte-for-byte identical
    assert template_path.read_bytes() == original_bytes
    assert hashlib.sha256(template_path.read_bytes()).hexdigest() == original_hash


@pytest.mark.integration
def test_incomplete_embeddings_preserves_previous_template(setup_enrollment_env):
    """If fewer than 20 valid embeddings are obtained, previous template is preserved."""
    env = setup_enrollment_env
    owner_name = "owner"
    template_path = env["encodings"] / f"{owner_name}.pkl"

    # Seed an existing template
    original_embeddings = [[0.1] * 512]
    save_user_embeddings(owner_name, original_embeddings, output_path=template_path)
    original_bytes = template_path.read_bytes()

    owner_dir = env["data"] / owner_name
    owner_dir.mkdir(parents=True, exist_ok=True)
    fake_images = [owner_dir / f"owner_{i:03d}.jpg" for i in range(20)]
    for img in fake_images:
        img.write_bytes(b"data")

    # Only 19 valid embeddings generated (1 rejected)
    partial_embeddings = [[0.05] * 512 for _ in range(19)]

    with (
        patch("src.register.capture_registration_images", return_value=(fake_images, False)),
        patch("src.register.generate_registration_embeddings", return_value=partial_embeddings),
        patch("src.register.enforce_strict_mode_policy"),
    ):
        result = register_user(
            user_name=owner_name,
            image_count=20,
            minimum_embeddings=20,
            overwrite=True,
        )

    assert result["success"] is False
    assert result["reason"] == "not_enough_valid_embeddings"
    assert template_path.read_bytes() == original_bytes


# ===========================================================================
# 4. Embedding dimensions and finite-value validation
# ===========================================================================

@pytest.mark.integration
def test_embedding_dimensions_and_finite_validation():
    """Only finite, non-empty, exactly 512-D vectors pass validation."""
    valid_512 = [0.01] * 512
    assert validate_embedding(valid_512, expected_dim=512) is True

    # Wrong dimensions
    assert validate_embedding([0.01] * 128, expected_dim=512) is False
    assert validate_embedding([0.01] * 511, expected_dim=512) is False
    assert validate_embedding([0.01] * 513, expected_dim=512) is False

    # NaNs and Infinities
    nan_vec = copy.copy(valid_512)
    nan_vec[100] = float("nan")
    assert validate_embedding(nan_vec, expected_dim=512) is False

    inf_vec = copy.copy(valid_512)
    inf_vec[200] = float("inf")
    assert validate_embedding(inf_vec, expected_dim=512) is False

    # Empty and None
    assert validate_embedding([], expected_dim=512) is False
    assert validate_embedding(None, expected_dim=512) is False


# ===========================================================================
# 5. Backup verification and safe atomic template replacement
# ===========================================================================

@pytest.mark.integration
def test_backup_verification_and_restore(setup_enrollment_env):
    """Backup creation, SHA-256 verification, and restore functionality."""
    env = setup_enrollment_env
    owner_name = "owner"
    owner_dir = env["data"] / owner_name
    owner_dir.mkdir(parents=True, exist_ok=True)
    template_path = env["encodings"] / f"{owner_name}.pkl"

    # Create dummy images and template
    test_img = owner_dir / "owner_test.jpg"
    test_img.write_bytes(b"sample_test_image_bytes")
    save_user_embeddings(owner_name, [[0.2] * 512], output_path=template_path)

    # 1. Create verified backup
    backup_dir = create_verified_backup(owner_name=owner_name)
    assert backup_dir.is_dir()
    assert (backup_dir / "manifest.json").is_file()

    # 2. Verify backup
    valid, msg = verify_backup(backup_dir)
    assert valid is True
    assert "Verified" in msg

    # 3. Simulate tamper detection
    backed_up_img = backup_dir / "data" / owner_name / "owner_test.jpg"
    backed_up_img.write_bytes(b"tampered_bytes")
    valid_tampered, err = verify_backup(backup_dir)
    assert valid_tampered is False
    assert "Hash mismatch" in err or "Size mismatch" in err

    # Fix it back
    shutil.copy2(test_img, backed_up_img)
    valid_fixed, _ = verify_backup(backup_dir)
    assert valid_fixed is True

    # 4. Test preparation only cleans owner images after verified backup
    prepare_clean_owner_environment(owner_name=owner_name, backup_dir=backup_dir)
    assert not test_img.exists()
    assert template_path.exists()  # Template preserved

    # 5. Test restore
    assert restore_backup(backup_dir=backup_dir, owner_name=owner_name) is True
    assert test_img.exists()
    assert test_img.read_bytes() == b"sample_test_image_bytes"


@pytest.mark.integration
def test_atomic_template_replacement_failure_safety(setup_enrollment_env, monkeypatch):
    """If verification of temporary template fails, existing template is not replaced."""
    env = setup_enrollment_env
    owner_name = "owner"
    template_path = env["encodings"] / f"{owner_name}.pkl"

    save_user_embeddings(owner_name, [[0.1] * 512], output_path=template_path)
    original_bytes = template_path.read_bytes()

    # Simulate an error during temporary file validation in save_user_embeddings
    def _exploding_verify(*args, **kwargs):
        raise ValueError("Simulated corrupted temporary template")

    monkeypatch.setattr("src.ai_engine._load_and_verify_record", _exploding_verify)

    with pytest.raises(ValueError, match="Simulated corrupted temporary template"):
        save_user_embeddings(owner_name, [[0.9] * 512], output_path=template_path)

    # Target template must remain intact
    assert template_path.read_bytes() == original_bytes


# ===========================================================================
# 6. Compatibility between enrollment and 1:1 recognition
# ===========================================================================

@pytest.mark.integration
def test_enrollment_recognition_compatibility(setup_enrollment_env):
    """Embeddings enrolled via 20-image flow are compatible with live 1:1 recognition."""
    env = setup_enrollment_env
    owner_name = "owner"
    template_path = env["encodings"] / f"{owner_name}.pkl"

    # Generate 20 distinct simulated embeddings centered around vector V
    base_vector = np.full(512, 0.1, dtype=np.float64)
    base_vector /= np.linalg.norm(base_vector)

    twenty_embeddings: list[list[float]] = []
    for i in range(20):
        v = base_vector + np.random.RandomState(i).normal(0, 0.01, 512)
        v /= np.linalg.norm(v)
        twenty_embeddings.append(v.tolist())

    save_user_embeddings(
        user_name=owner_name,
        embeddings=twenty_embeddings,
        output_path=template_path,
        threshold=0.30,
    )

    # Load trusted records strictly filtering for owner
    trusted_records = load_trusted_records()
    assert len(trusted_records) == 1
    assert trusted_records[0]["user_name"] == owner_name
    assert len(trusted_records[0]["embeddings"]) == 20

    # Test candidate matching: identical to sample #3 should match with distance ~0.0
    candidate = twenty_embeddings[3]
    match = find_best_match(candidate, trusted_records, threshold=0.30)
    assert match["is_match"] is True
    assert match["user_name"] == owner_name
    assert match["cosine_distance"] < 0.05
    assert match["threshold"] == 0.30


# ===========================================================================
# 7. Non-owner identities remain rejected
# ===========================================================================

@pytest.mark.integration
def test_non_owner_identities_remain_rejected(setup_enrollment_env):
    """An impostor face vector (orthogonal or distant) is rejected with is_match == False."""
    env = setup_enrollment_env
    owner_name = "owner"
    template_path = env["encodings"] / f"{owner_name}.pkl"

    owner_vector = np.zeros(512, dtype=np.float64)
    owner_vector[:256] = 1.0
    owner_vector /= np.linalg.norm(owner_vector)

    save_user_embeddings(owner_name, [owner_vector.tolist()], output_path=template_path, threshold=0.30)
    trusted_records = load_trusted_records()

    # Impostor vector in orthogonal subspace
    impostor_vector = np.zeros(512, dtype=np.float64)
    impostor_vector[256:] = 1.0
    impostor_vector /= np.linalg.norm(impostor_vector)

    match = find_best_match(impostor_vector.tolist(), trusted_records, threshold=0.30)
    assert match["is_match"] is False
    assert match["cosine_distance"] > 0.90


# ===========================================================================
# 8. Threshold remains 0.30 and HMAC integrity validation enforced
# ===========================================================================

@pytest.mark.integration
def test_threshold_remains_0_30_and_integrity_enforced(setup_enrollment_env):
    """Cosine threshold must be 0.30 and tampered templates must raise IntegrityError."""
    assert DEFAULT_COSINE_THRESHOLD == 0.30
    assert RECOGNITION_SETTINGS.owner_name == "owner"

    env = setup_enrollment_env
    owner_name = "owner"
    template_path = env["encodings"] / f"{owner_name}.pkl"

    save_user_embeddings(owner_name, [[0.1] * 512], output_path=template_path, threshold=0.30)

    # Legitimate load succeeds
    rec = load_user_embeddings(owner_name, input_path=template_path)
    assert rec["threshold"] == 0.30

    # Tamper payload
    raw_payload = save_pickle.__globals__["pickle"].loads(template_path.read_bytes())
    raw_payload["record"]["embeddings"] = [[0.9] * 512]  # altered without updating HMAC
    save_pickle(raw_payload, template_path)

    # Must raise IntegrityError
    with pytest.raises(IntegrityError):
        load_user_embeddings(owner_name, input_path=template_path)


# ===========================================================================
# 9. CLI enroll_owner flow test
# ===========================================================================

@pytest.mark.integration
def test_enroll_owner_cli_full_flow(setup_enrollment_env):
    """enroll_owner CLI command executes backup, verification, and 20-image registration."""
    env = setup_enrollment_env
    owner_name = "owner"
    owner_dir = env["data"] / owner_name
    owner_dir.mkdir(parents=True, exist_ok=True)
    template_path = env["encodings"] / f"{owner_name}.pkl"

    # Seed existing data to be backed up
    old_img = owner_dir / "old_face.jpg"
    old_img.write_bytes(b"old_data")
    save_user_embeddings(owner_name, [[0.2] * 512], output_path=template_path)

    # Mock successful 20-image registration
    mock_reg_result = {
        "success": True,
        "user_name": owner_name,
        "image_count": 20,
        "embedding_count": 20,
        "encoding_path": template_path,
        "model_name": DEFAULT_MODEL_NAME,
    }

    with patch("scripts.enroll_owner.register_user", return_value=mock_reg_result) as mock_reg:
        code = enroll_owner(overwrite=True, skip_auth=True)

    assert code == 0
    mock_reg.assert_called_once_with(
        user_name=owner_name,
        image_count=20,
        minimum_embeddings=20,
        overwrite=True,
    )
    # Check that a backup was created and verified
    backups = list(env["backups"].glob("backup_*"))
    assert len(backups) == 1
    assert (backups[0] / "manifest.json").is_file()
