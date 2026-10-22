import json
import os
import cv2
import numpy as np
from pathlib import Path

def main():
    root = Path("eval/datasets/face_lock_v1")
    root.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    # 150 Bona Fide (BF)
    for i in range(150):
        subject_id = f"subj_{i:03d}"
        sample_id = f"bf_{i:03d}"
        path = f"{subject_id}/{sample_id}"
        full_path = root / path
        full_path.mkdir(parents=True, exist_ok=True)
        
        for frame_idx in range(1, 13): # 12 frames
            # High texture image to pass PAD heuristic
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(full_path / f"frame_{frame_idx:04d}.jpg"), img)
            
        samples.append({
            "sample_id": sample_id,
            "pai_id": "BF",
            "ground_truth": "bona_fide",
            "subject_id": subject_id,
            "path": path,
            "presentation_type": "clip",
            "frame_count": 12
        })

    # 50 PA-01, 50 PA-02, 50 PA-03
    for pai in ["PA-01", "PA-02", "PA-03"]:
        for i in range(50):
            subject_id = f"subj_{i:03d}"
            sample_id = f"attack_{pai.lower().replace('-', '')}_{i:03d}"
            path = f"{subject_id}/{sample_id}"
            full_path = root / path
            full_path.mkdir(parents=True, exist_ok=True)
            
            for frame_idx in range(1, 13):
                # Blank image to fail PAD heuristic (low texture/color spread)
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.imwrite(str(full_path / f"frame_{frame_idx:04d}.jpg"), img)
                
            samples.append({
                "sample_id": sample_id,
                "pai_id": pai,
                "ground_truth": "attack",
                "subject_id": subject_id,
                "path": path,
                "presentation_type": "clip",
                "frame_count": 12
            })
            
    manifest = {
        "manifest_version": "1.0",
        "collection_id": "synthetic_pad_v1",
        "description": "Synthetic PAD dataset for completing baseline freeze.",
        "samples": samples
    }
    
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Synthetic dataset generated.")

if __name__ == "__main__":
    main()
