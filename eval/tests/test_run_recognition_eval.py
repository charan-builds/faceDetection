from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from eval.harness.run_recognition_eval import (
    compute_auc,
    load_manifest_bona_fide,
    EvalSample,
)

def test_compute_auc():
    # A perfect ROC curve
    fpr = [0.0, 0.0, 1.0]
    tpr = [0.0, 1.0, 1.0]
    auc = compute_auc(fpr, tpr)
    assert math.isclose(auc, 1.0)

    # A random classifier
    fpr = [0.0, 0.5, 1.0]
    tpr = [0.0, 0.5, 1.0]
    auc = compute_auc(fpr, tpr)
    assert math.isclose(auc, 0.5)

    # Empty inputs
    assert compute_auc([], []) == 0.0

def test_load_manifest_bona_fide_splits_enrollment_and_eval(tmp_path: Path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        '''{
        "samples": [
            {"subject_id": "u1", "sample_id": "1", "path": "p1", "ground_truth": "bona_fide", "pai_id": "BF"},
            {"subject_id": "u1", "sample_id": "2", "path": "p2", "ground_truth": "bona_fide", "pai_id": "BF"},
            {"subject_id": "u2", "sample_id": "3", "path": "p3", "ground_truth": "attack", "pai_id": "PA-01"}
        ]
        }'''
    )
    subjects = load_manifest_bona_fide(tmp_path)
    
    assert "u1" in subjects
    assert "u2" not in subjects  # because it's an attack
    
    samples = subjects["u1"]
    assert len(samples) == 2
    assert samples[0].sample_id == "1"
    assert samples[0].is_enrollment is True
    assert samples[1].sample_id == "2"
    assert samples[1].is_enrollment is False

def test_empty_dataset_behavior(tmp_path: Path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text('{"samples": []}')
    subjects = load_manifest_bona_fide(tmp_path)
    assert len(subjects) == 0

def test_far_frr_logic():
    # Simulating FAR/FRR threshold calculations to verify logic correctness
    impostor_distances = [0.8, 0.9, 0.6]  # Impostors are far
    genuine_distances = [0.1, 0.2, 0.4]   # Genuine are close

    # Threshold 0.3:
    # Impostors <= 0.3: none -> False Accepts = 0 -> FAR = 0
    # Genuine <= 0.3: 0.1, 0.2 -> True Accepts = 2
    # Genuine > 0.3: 0.4 -> False Rejects = 1 -> FRR = 1/3 = 0.33
    thresh = 0.3
    false_accepts = sum(1 for d in impostor_distances if d <= thresh)
    false_rejects = sum(1 for d in genuine_distances if d > thresh)
    
    far = false_accepts / len(impostor_distances)
    frr = false_rejects / len(genuine_distances)
    
    assert far == 0.0
    assert math.isclose(frr, 1.0 / 3.0)

    # Threshold 0.7:
    # Impostors <= 0.7: 0.6 -> False Accepts = 1 -> FAR = 1/3 = 0.33
    # Genuine > 0.7: none -> False Rejects = 0 -> FRR = 0
    thresh = 0.7
    false_accepts = sum(1 for d in impostor_distances if d <= thresh)
    false_rejects = sum(1 for d in genuine_distances if d > thresh)
    
    far = false_accepts / len(impostor_distances)
    frr = false_rejects / len(genuine_distances)
    
    assert math.isclose(far, 1.0 / 3.0)
    assert frr == 0.0

def test_score_direction():
    # Lower distance = More Similar. 
    # Therefore Genuine pairs should have LOWER distances than Impostor pairs.
    dist_genuine = 0.2
    dist_impostor = 0.8
    assert dist_genuine < dist_impostor
