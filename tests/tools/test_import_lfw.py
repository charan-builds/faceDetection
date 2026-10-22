import json
from pathlib import Path

import pytest

from tools.import_lfw import import_lfw


def create_mock_lfw(tmp_path: Path, config: dict[str, int]) -> Path:
    """Create a mock LFW directory structure."""
    source_dir = tmp_path / "lfw_raw"
    source_dir.mkdir()
    
    for identity, img_count in config.items():
        identity_dir = source_dir / identity
        identity_dir.mkdir()
        for i in range(1, img_count + 1):
            img_path = identity_dir / f"{identity}_{i:04d}.jpg"
            img_path.write_text("dummy_image_data")
            
    return source_dir


def test_import_lfw_valid_dataset(tmp_path: Path) -> None:
    source_dir = create_mock_lfw(tmp_path, {
        "Person_A": 2,
        "Person_B": 3,  # Should truncate to 2
    })
    output_dir = tmp_path / "face_lock_lfw"
    
    import_lfw(source_dir, output_dir)
    
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "bona_fide" / "subj_0001" / "session_01" / "frame_0001.jpg").exists()
    assert (output_dir / "bona_fide" / "subj_0001" / "session_02" / "frame_0001.jpg").exists()
    assert (output_dir / "bona_fide" / "subj_0002" / "session_01" / "frame_0001.jpg").exists()
    assert (output_dir / "bona_fide" / "subj_0002" / "session_02" / "frame_0001.jpg").exists()
    
    # Should ignore 3rd image of Person_B
    assert not (output_dir / "bona_fide" / "subj_0002" / "session_03").exists()
    
    with open(output_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
        
    assert len(manifest["samples"]) == 4
    assert manifest["samples"][0]["subject_id"] == "subj_0001"
    assert manifest["samples"][1]["subject_id"] == "subj_0001"
    assert manifest["samples"][2]["subject_id"] == "subj_0002"
    assert manifest["samples"][3]["subject_id"] == "subj_0002"


def test_import_lfw_insufficient_samples(tmp_path: Path) -> None:
    source_dir = create_mock_lfw(tmp_path, {
        "Person_A": 1,  # Should skip because < 2
        "Person_B": 2,  # Should include
    })
    output_dir = tmp_path / "face_lock_lfw"
    
    import_lfw(source_dir, output_dir)
    
    with open(output_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
        
    # Only Person_B should be imported (mapped to subj_0001)
    assert len(manifest["samples"]) == 2
    assert manifest["samples"][0]["subject_id"] == "subj_0001"
    assert manifest["samples"][1]["subject_id"] == "subj_0001"


def test_import_lfw_empty_dataset(tmp_path: Path) -> None:
    source_dir = create_mock_lfw(tmp_path, {})
    output_dir = tmp_path / "face_lock_lfw"
    
    import_lfw(source_dir, output_dir)
    
    with open(output_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
        
    assert len(manifest["samples"]) == 0


def test_import_lfw_missing_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "does_not_exist"
    output_dir = tmp_path / "face_lock_lfw"
    
    with pytest.raises(SystemExit):
        import_lfw(source_dir, output_dir)
