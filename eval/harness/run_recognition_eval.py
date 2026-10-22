#!/usr/bin/env python3
"""
Face Lock AI — Recognition Validation Harness

Evaluates FAR, FRR, ROC, and AUC for the face recognition pipeline.
Reads bona_fide samples from a face_lock_v1 dataset manifest, splits
them into enrollment and evaluation sets, and computes genuine and
impostor distance distributions using the production ai_engine logic.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from eval.harness.clip_loader import load_presentation
from src.ai_engine import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_DETECTOR_BACKEND,
    DEFAULT_MODEL_NAME,
    cosine_distance,
    generate_embedding,
)

LOGGER = logging.getLogger("face_lock.eval.recognition")

@dataclass
class EvalSample:
    subject_id: str
    sample_id: str
    path: str
    is_enrollment: bool

def load_manifest_bona_fide(collection_root: Path) -> dict[str, list[EvalSample]]:
    """Loads bona_fide samples and splits them into enrollment vs eval."""
    manifest_path = collection_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json in {collection_root}")

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by subject
    subjects: dict[str, list[EvalSample]] = defaultdict(list)
    for sample in data.get("samples", []):
        if sample.get("ground_truth") == "bona_fide" and sample.get("pai_id") == "BF":
            subjects[sample["subject_id"]].append(
                EvalSample(
                    subject_id=sample["subject_id"],
                    sample_id=sample["sample_id"],
                    path=sample["path"],
                    is_enrollment=False,
                )
            )

    # Sort and designate the first sample as enrollment
    for subject_id, samples in subjects.items():
        samples.sort(key=lambda s: s.sample_id)
        if samples:
            samples[0].is_enrollment = True

    return subjects

def get_face_embedding(frame: np.ndarray) -> list[float] | None:
    """Generates an embedding from a single frame. Returns None if no face detected."""
    try:
        # Use production logic
        return generate_embedding(
            image_input=frame,
            model_name=DEFAULT_MODEL_NAME,
            detector_backend=DEFAULT_DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
    except RuntimeError:
        return None

def compute_auc(fpr: list[float], tpr: list[float]) -> float:
    """Computes Area Under the Curve (AUC) using the trapezoidal rule."""
    if not fpr or not tpr:
        return 0.0
    # Ensure ordered by fpr ascending
    points = sorted(zip(fpr, tpr), key=lambda x: x[0])
    sorted_fpr = [p[0] for p in points]
    sorted_tpr = [p[1] for p in points]
    return float(np.trapz(sorted_tpr, sorted_fpr))

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Face Recognition Evaluation")
    parser.add_argument("--collection-root", type=Path, required=True, help="Dataset root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for reports")
    parser.add_argument("--max-frames", type=int, default=5, help="Max frames to process per sample")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Loading manifest from {args.collection_root}")
    try:
        subjects = load_manifest_bona_fide(args.collection_root)
    except Exception as e:
        LOGGER.error(f"Failed to load manifest: {e}")
        return 1

    enrollment_db: dict[str, list[list[float]]] = defaultdict(list)
    evaluation_embeddings: list[tuple[str, list[float]]] = []

    # Process samples
    for subject_id, samples in subjects.items():
        for sample in samples:
            LOGGER.info(f"Processing {'Enrollment' if sample.is_enrollment else 'Evaluation'} sample: {sample.sample_id}")
            try:
                presentation = load_presentation(
                    collection_root=args.collection_root,
                    relative_path=sample.path,
                    presentation_type="clip",
                    max_frames=args.max_frames,
                )
            except Exception as e:
                LOGGER.warning(f"Failed to load {sample.sample_id}: {e}")
                continue

            valid_embeddings = []
            for frame in presentation.frames:
                emb = get_face_embedding(frame)
                if emb is not None:
                    valid_embeddings.append(emb)

            if not valid_embeddings:
                LOGGER.warning(f"No faces detected in {sample.sample_id}")
                continue

            if sample.is_enrollment:
                enrollment_db[subject_id].extend(valid_embeddings)
            else:
                for emb in valid_embeddings:
                    evaluation_embeddings.append((subject_id, emb))

    LOGGER.info(f"Enrolled {len(enrollment_db)} subjects.")
    LOGGER.info(f"Extracted {len(evaluation_embeddings)} evaluation embeddings.")

    if not enrollment_db or not evaluation_embeddings:
        LOGGER.error("Insufficient data for evaluation. Blocked.")
        return 1

    genuine_distances: list[float] = []
    impostor_distances: list[float] = []

    # Perform pairwise comparisons
    for eval_subject, eval_emb in evaluation_embeddings:
        for enrolled_subject, enrolled_embs in enrollment_db.items():
            if not enrolled_embs:
                continue
            
            # The system uses 1:N best match, so we find the minimum distance to the enrolled template
            distances = [cosine_distance(eval_emb, e) for e in enrolled_embs]
            min_distance = min(distances)

            if eval_subject == enrolled_subject:
                genuine_distances.append(min_distance)
            else:
                impostor_distances.append(min_distance)

    LOGGER.info(f"Generated {len(genuine_distances)} Genuine pairs and {len(impostor_distances)} Impostor pairs.")

    # Sweep threshold to compute FAR/FRR/ROC
    thresholds = np.linspace(0.0, 1.0, 101)
    results = []
    fpr_list = []
    tpr_list = []

    prod_far = 0.0
    prod_frr = 0.0

    for thresh in thresholds:
        # A distance <= threshold is an Accept
        false_accepts = sum(1 for d in impostor_distances if d <= thresh)
        false_rejects = sum(1 for d in genuine_distances if d > thresh)
        true_accepts = sum(1 for d in genuine_distances if d <= thresh)

        far = false_accepts / max(1, len(impostor_distances)) if impostor_distances else 0.0
        frr = false_rejects / max(1, len(genuine_distances)) if genuine_distances else 0.0
        tpr = true_accepts / max(1, len(genuine_distances)) if genuine_distances else 0.0

        fpr_list.append(far)
        tpr_list.append(tpr)

        results.append({
            "threshold": round(float(thresh), 2),
            "far": far,
            "frr": frr,
            "false_accepts": false_accepts,
            "false_rejects": false_rejects
        })

        if math.isclose(thresh, DEFAULT_COSINE_THRESHOLD, abs_tol=1e-5):
            prod_far = far
            prod_frr = frr

    auc = compute_auc(fpr_list, tpr_list)

    # Determine threshold evidence strength
    # Since dataset is very small, we must flag it.
    evidence_strength = "NOT SUPPORTED"
    if len(genuine_distances) == 0 or len(impostor_distances) == 0:
        evidence_strength = "CANNOT DETERMINE"
    elif len(genuine_distances) < 50 or len(impostor_distances) < 50:
        evidence_strength = "PLAUSIBLE BUT INSUFFICIENT EVIDENCE (Tiny Dataset)"
    elif prod_far <= 0.01 and prod_frr <= 0.05:
        evidence_strength = "SUPPORTED"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": DEFAULT_MODEL_NAME,
        "detector_backend": DEFAULT_DETECTOR_BACKEND,
        "dataset_stats": {
            "enrolled_subjects": len(enrollment_db),
            "evaluation_embeddings": len(evaluation_embeddings),
            "genuine_comparisons": len(genuine_distances),
            "impostor_comparisons": len(impostor_distances),
        },
        "distance_distributions": {
            "genuine": {
                "count": len(genuine_distances),
                "min": float(np.min(genuine_distances)) if genuine_distances else None,
                "max": float(np.max(genuine_distances)) if genuine_distances else None,
                "mean": float(np.mean(genuine_distances)) if genuine_distances else None,
            },
            "impostor": {
                "count": len(impostor_distances),
                "min": float(np.min(impostor_distances)) if impostor_distances else None,
                "max": float(np.max(impostor_distances)) if impostor_distances else None,
                "mean": float(np.mean(impostor_distances)) if impostor_distances else None,
            }
        },
        "roc": {
            "auc": auc
        },
        "production_threshold_analysis": {
            "threshold": DEFAULT_COSINE_THRESHOLD,
            "far": prod_far,
            "frr": prod_frr,
            "evidence_strength": evidence_strength
        },
        "sweep": results
    }

    report_path = args.output_dir / "recognition_eval_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    LOGGER.info(f"Evaluation report written to {report_path}")
    LOGGER.info(f"Production Threshold 0.30 -> FAR: {prod_far:.4f}, FRR: {prod_frr:.4f} ({evidence_strength})")
    LOGGER.info(f"AUC: {auc:.4f}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
