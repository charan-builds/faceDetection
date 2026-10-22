# Face Lock AI — Dataset Protocol

## Purpose
This protocol defines the exact methodology for capturing, validating, and testing Face Recognition populations for Face Lock AI. The purpose is to produce reproducible, statistically significant measurements of False Accept Rate (FAR) and False Reject Rate (FRR) without data leakage or threshold bias.

## Dataset Structure
We use a structured multi-session layout (`face_lock_v1`).
- `bona_fide/` — Genuine human presentations.
  - `subject_001/` — Identity directory.
    - `session_01/` — **Enrollment** images.
    - `session_02/` — **Evaluation** images.

This explicit separation is enforced system-wide.

## Enrollment Protocol
1. An identity is captured in `session_01`.
2. The user should face the webcam in standard lighting.
3. A minimum of 12 frames are captured.
4. Only `session_01` is ever used as the gallery/enrollment template.

## Evaluation Protocol
1. The same identity is captured later in `session_02`.
2. Capture conditions should vary (lighting, slight pose angle, distance).
3. The frames in `session_02` act as probe samples against the enrolled templates.

## Session Separation & Leakage Prevention
To ensure a true zero-shot evaluation, the following leakage checks are enforced by `tools/qc_dataset.py`:
- **Identity overlap**: A single subject must have `session_01` (enrollment) and `session_02` (evaluation) exclusively assigned.
- **Hash checking**: A SHA-256 hash is computed for all images. The QC pipeline guarantees `data_leakage_duplicate_hash` is emitted if any frame is duplicated across sessions or identities.
- **Visual content overlap**: (Future implementation if needed) Perceptual hashing.

## Dataset Quality Control (QC)
The QC pipeline evaluates every dataset prior to the evaluation loop:
1. **File validity**: Detects corrupted JPEGs.
2. **Face validity**: Uses `DeepFace.extract_faces` to ensure exactly 1 face is present in all frames. Emits `no_face_detected` or `multiple_faces_detected`.
3. **Capture limits**: Warns if frames are below the recommended threshold (12).
4. **Brightness/Blur**: Warns if images are too dark or too blurry.

## Pair Generation
- **Genuine Pairs**: The evaluation harness pairs `session_02` frames against the `session_01` template of the *same* subject.
- **Impostor Pairs**: The evaluation harness pairs `session_02` frames against the `session_01` templates of *all other* subjects in the dataset.

## Staged Population Plan
We adhere to a staged dataset population approach:
- **Stage A (Development)**: 2–5 identities. Verifies pipeline mechanics and test suite correctness. Cannot prove threshold safety.
- **Stage B (Medium Validation)**: 50 identities. Provides ~50 genuine and ~2,450 impostor comparisons. Sufficient for initial threshold estimates and FAR tuning.
- **Stage C (Strong Evaluation)**: 500+ identities. Provides ~500 genuine and ~250,000 impostor comparisons. Necessary for statistically significant production release.

## Reproducibility
The system stores immutable JSON reports for each evaluation sweep containing:
- Dataset ID and sample counts.
- Detected Genuine / Impostor sizes.
- Calculated ROC and AUC.
- Resulting threshold evidence.

## Limitations
- Internal webcams capture highly constrained environment setups; this may cause domain-gap failures when deployed to new hardware.
- The pipeline does not currently evaluate demographic variance unless explicitly modeled in the manifest metadata.
- Impostor comparisons scale as `O(N^2)` — at Stage C, deterministic sampling of impostor pairs may be necessary to manage computational limits.
