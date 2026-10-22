#!/usr/bin/env python3
"""
Importer for Labeled Faces in the Wild (LFW) dataset.

Maps the LFW structure:
  lfw/
    Subject_Name/
      Subject_Name_0001.jpg
      Subject_Name_0002.jpg

To Face Lock AI canonical format:
  face_lock_lfw/
    bona_fide/
      subj_0001/
        session_01/
          frame_0001.jpg  (Mapped from image 1)
        session_02/
          frame_0001.jpg  (Mapped from image 2)

Subjects with fewer than 2 images are discarded.
Subjects with > 2 images only use the first 2 images to guarantee balanced Genuine pairs.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

LOGGER = logging.getLogger("face_lock.import_lfw")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def import_lfw(lfw_source_dir: Path, output_collection_dir: Path) -> None:
    if not lfw_source_dir.is_dir():
        LOGGER.error(f"LFW source directory not found: {lfw_source_dir}")
        sys.exit(1)

    output_collection_dir.mkdir(parents=True, exist_ok=True)
    bona_fide_dir = output_collection_dir / "bona_fide"
    bona_fide_dir.mkdir(exist_ok=True)

    manifest = {
        "manifest_version": "1.0.0",
        "collection_id": output_collection_dir.name,
        "samples": []
    }

    subject_count = 0
    skipped_count = 0

    # LFW has one folder per identity
    for identity_dir in sorted(lfw_source_dir.iterdir()):
        if not identity_dir.is_dir():
            continue

        images = sorted(identity_dir.glob("*.jpg"))
        if len(images) < 2:
            skipped_count += 1
            continue

        subject_count += 1
        # Use a canonical subj_XXX ID to preserve anonymity standard
        canonical_subject_id = f"subj_{subject_count:04d}"
        target_subject_dir = bona_fide_dir / canonical_subject_id
        target_subject_dir.mkdir(exist_ok=True)

        for session_idx, img_path in enumerate(images[:2], start=1):
            session_name = f"session_{session_idx:02d}"
            target_session_dir = target_subject_dir / session_name
            target_session_dir.mkdir(exist_ok=True)

            target_img_path = target_session_dir / "frame_0001.jpg"
            shutil.copy2(img_path, target_img_path)

            sample_id = f"BF_{canonical_subject_id}_s{session_idx:02d}"
            manifest["samples"].append({
                "sample_id": sample_id,
                "pai_id": "BF",
                "ground_truth": "bona_fide",
                "subject_id": canonical_subject_id,
                "path": f"bona_fide/{canonical_subject_id}/{session_name}",
                "presentation_type": "clip",
                "frame_count": 1
            })

    manifest_path = output_collection_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    LOGGER.info(f"Import complete.")
    LOGGER.info(f"Imported subjects: {subject_count}")
    LOGGER.info(f"Skipped subjects (< 2 images): {skipped_count}")
    LOGGER.info(f"Total samples written: {subject_count * 2}")
    LOGGER.info(f"Manifest written to: {manifest_path}")

    # Pair capacity estimate
    impostor_pairs = subject_count * (subject_count - 1)
    LOGGER.info(f"Pair capacity -> Genuine: {subject_count}, Impostor: {impostor_pairs}")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Import LFW dataset to Face Lock canonical format")
    parser.add_argument("--source", type=Path, required=True, help="Path to raw LFW extracted dataset")
    parser.add_argument("--output", type=Path, required=True, help="Path to output canonical dataset (e.g. eval/datasets/face_lock_lfw)")
    args = parser.parse_args()

    import_lfw(args.source, args.output)


if __name__ == "__main__":
    main()
