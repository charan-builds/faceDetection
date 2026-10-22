import pytest
import numpy as np
from pathlib import Path

from tools.collect_recognition_dataset import (
    validate_subject_id,
    get_next_subject_id,
    get_next_session,
    validate_faces,
    save_frames,
    CollectionValidationError,
    StorageFailureError,
    CollectionConfig
)


def test_validate_subject_id():
    assert validate_subject_id("subj_001") == "subj_001"
    assert validate_subject_id("SUBJ_042") == "subj_042"
    with pytest.raises(CollectionValidationError):
        validate_subject_id("person_001")


def test_get_next_subject_id(tmp_path: Path):
    assert get_next_subject_id(tmp_path) == "subj_001"
    
    (tmp_path / "bona_fide" / "subj_001").mkdir(parents=True)
    assert get_next_subject_id(tmp_path) == "subj_002"
    
    (tmp_path / "bona_fide" / "subj_099").mkdir(parents=True)
    assert get_next_subject_id(tmp_path) == "subj_100"


def test_get_next_session(tmp_path: Path):
    subject_dir = tmp_path / "subj_001"
    subject_dir.mkdir()
    
    # Empty dir -> session_01, 0 frames
    sess, count = get_next_session(subject_dir)
    assert sess == "session_01"
    assert count == 0
    
    # session_01 with 3 frames (incomplete -> should resume session_01)
    s1 = subject_dir / "session_01"
    s1.mkdir()
    for i in range(1, 4):
        (s1 / f"frame_{i:04d}.jpg").write_text("data")
        
    sess, count = get_next_session(subject_dir)
    assert sess == "session_01"
    assert count == 3
    
    # session_01 with 5 frames (minimum met -> move to session_02)
    for i in range(4, 6):
        (s1 / f"frame_{i:04d}.jpg").write_text("data")
        
    sess, count = get_next_session(subject_dir)
    assert sess == "session_02"
    assert count == 0
    
    # Create session_02 with 10 files -> fully captured
    s2 = subject_dir / "session_02"
    s2.mkdir()
    for i in range(1, 11):
        (s2 / f"frame_{i:04d}.jpg").write_text("data")
        
    sess, count = get_next_session(subject_dir)
    assert sess is None
    assert count == 0


def test_validate_faces_mock(monkeypatch: pytest.MonkeyPatch):
    try:
        from deepface import DeepFace
        
        def mock_extract_faces(img_path, **kwargs):
            val = img_path.tobytes().decode()
            if val == "zero":
                return []
            elif val.startswith("one"):
                return [{"confidence": 0.99}]
            elif val == "multiple":
                return [{"confidence": 0.99}, {"confidence": 0.98}]
            elif val == "error":
                raise ValueError("No face detected")
            return []
            
        monkeypatch.setattr(DeepFace, "extract_faces", mock_extract_faces)
        
        # Test 1: duplicate identical frames "one_A"
        f1 = np.frombuffer(b"zero", dtype=np.uint8)
        f2 = np.frombuffer(b"one_A", dtype=np.uint8)
        f3 = np.frombuffer(b"one_A", dtype=np.uint8)  # duplicate, should drop
        f4 = np.frombuffer(b"one_B", dtype=np.uint8)  # valid
        f5 = np.frombuffer(b"multiple", dtype=np.uint8)
        f6 = np.frombuffer(b"error", dtype=np.uint8)
        
        frames = [f1, f2, f3, f4, f5, f6]
        valid, zero, multiple, dups = validate_faces(frames)
        
        assert len(valid) == 2  # one_A, one_B
        assert zero == 2  # zero, error
        assert multiple == 1
        assert dups == 1  # f3
        
    except ImportError:
        pass


def test_save_frames_dry_run(tmp_path: Path):
    config = CollectionConfig(
        collection_root=tmp_path,
        camera_index=0,
        capture_fps=5.0,
        jpeg_quality=90,
        dry_run=True
    )
    
    target_dir = tmp_path / "test_dir"
    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    
    saved = save_frames(frames, target_dir, config)
    assert saved == 1
    assert not target_dir.exists()  # Nothing should be written


def test_save_frames_storage_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config = CollectionConfig(
        collection_root=tmp_path,
        camera_index=0,
        capture_fps=5.0,
        jpeg_quality=90,
        dry_run=False
    )
    
    target_dir = tmp_path / "test_dir"
    
    def mock_imwrite(filename, img, params=None):
        return False
        
    import cv2
    monkeypatch.setattr(cv2, "imwrite", mock_imwrite)
    
    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    
    with pytest.raises(StorageFailureError):
        save_frames(frames, target_dir, config)
