# Face Lock AI — Physical PAD Data Collection Protocol
# Version: 1.0.0
# Status: DRAFT — approved for P1 collection planning
# Source of truth for sample targets: eval/protocols/pad_iso_lab_v1.yaml

---

## 0. Verified Target Numbers

The following counts are taken directly from
`eval/protocols/pad_iso_lab_v1.yaml → sample_sizes`:

| Category       | PAI ID | Min presentations | Min subjects | Gate role               |
|----------------|--------|-------------------|--------------|-------------------------|
| Bona fide      | BF     | **150**           | **30**       | BPCER denominator       |
| Print attack   | PA-01  | **50**            | **10**       | Pooled APCER (PA-01–03) |
| Static screen  | PA-02  | **50**            | **10**       | Pooled APCER (PA-01–03) |
| Video replay   | PA-03  | **50**            | **10**       | Pooled APCER (PA-01–03) |
| Virtual camera | PA-04  | 30 (optional)     | 6 (optional) | Separate APCER; deferred to P2 |

**Minimum for first approved baseline:** 300 presentations (150 BF + 150 attacks across PA-01/02/03).

Tier gate thresholds (standard tier from YAML):
- APCER pooled (PA-01 + PA-02 + PA-03): ≤ 5.00 %
- BPCER: ≤ 5.00 %
- P95 latency: ≤ 1200 ms

---

## 1. Roles

| Role                | Responsibility                                                          |
|---------------------|-------------------------------------------------------------------------|
| **Collector**       | Operates camera and collection tool. Does not assign labels.           |
| **Labeler**         | Assigns pai_id, lux_bucket, device notes; writes manifest entries.     |
| **QC reviewer**     | Spot-checks frames, runs qc_dataset.py, signs off counts.              |
| **Eval engineer**   | Runs E3/E4 harness, authors REPORT.md.                                 |
| **Security reviewer** | Signs REPORT.md before baseline freeze. Must be different from eval engineer. |

The collector and labeler may be the same person. The eval engineer and
security reviewer must be different people for any production baseline claim.

---

## 2. Consent and Privacy

1. Every participant must give explicit informed consent before any capture.
   Describe: what is being captured, how it will be used (internal R&D
   evaluation only), how long it will be retained, and how to request deletion.
2. Never store real names, emails, employee IDs, or any PII in dataset files,
   manifests, or report outputs.
3. Assign a pseudonymous identifier immediately: `subj_NNN` (three digits,
   zero-padded) for bona fide; `subj_NNN` with the same pool for attack
   subjects. A single real person maps to exactly one `subj_NNN` across
   the entire collection.
4. Record a consent log (separate from the dataset) that maps real identities
   to `subj_NNN`. The consent log must be stored encrypted and separately
   from all frame files, manifests, and evaluation artifacts.
5. Participants may withdraw at any time. On withdrawal, delete all frames,
   manifest entries, and derived artifacts for that subject.
6. Do not capture: minors, subjects who appear distressed, subjects who
   cannot give informed consent.

---

## 3. Hardware and Environment Requirements

### 3.1 Camera

| Requirement | Value |
|-------------|-------|
| Device | The same physical webcam used in the production Face Lock deployment |
| Index | Must match `WEBCAM_SETTINGS.camera_index` (default: 0) |
| Resolution | Record actual resolution in `collection_meta.json` |
| Mount | Fixed — tripod or laptop lid. No hand-held camera. |
| Cleaning | Lens clean before each subject |

This is a hard requirement. APCER/BPCER measured on a different camera
does not predict performance on the production device.

### 3.2 Physical environment

| Element | Requirement |
|---------|-------------|
| Background | Neutral wall or partition. No mirrors, screens, or strong patterns behind subject. |
| Subject distance | 40–55 cm eye-to-lens. Mark the floor with tape before P1. |
| Ambient lighting | Measure with a lux meter or phone app. Record in `collection_meta.json`. |
| Lux bucket rubric | low < 100 lux · normal 100–500 lux · high > 500 lux |
| Backlight | None. Light source should be in front of or above the subject, not behind. |
| Secondary persons | No other face must be visible in the camera frame during capture. |

### 3.3 Collection workstation

- Python 3.11, all `requirements.txt` dependencies installed.
- `collect_pad_dataset.py` verified to run without error before P1 begins.
- Sufficient local disk space: estimate 300 clips × 16 frames × ~100 KB ≈ 480 MB
  (reserve 2 GB for retakes and intermediate files).

---

## 4. Bona Fide Protocol (PAI-ID: BF)

**Goal:** Capture 150 genuine live-person presentations from 30 distinct subjects,
covering lighting variation.

### 4.1 Per-subject session plan

Each subject completes **5 sessions**:

| Session | Lighting | Notes |
|---------|----------|-------|
| session_01 | normal | Enrollment-like: neutral expression, direct gaze |
| session_02 | normal | Slight natural head variation (±15° nod or turn) |
| session_03 | normal | Smile or changed expression |
| session_04 | low     | Office with one ceiling light off or lamp moved |
| session_05 | high    | Near a window in daylight or extra lamp added |

5 sessions × 30 subjects = 150 presentations. This exactly meets the
minimum. Collect additional subjects or sessions if any session must be
discarded at QC.

### 4.2 Per-session capture procedure

1. Subject is seated, face centered on screen, eyes open.
2. Collector verifies: single face visible, background clear, correct lux bucket.
3. Run tool:
   ```powershell
   python tools/collect_pad_dataset.py `
     --collection-root eval/datasets/face_lock_v1 `
     --collector-id <collector_id>
   ```
4. Enter `subj_NNN` at the subject ID prompt.
5. Enter `BF` at the presentation type prompt.
6. Press **s** to start recording.
7. Subject holds natural pose; slight slow head motion is encouraged but not required.
   No exaggerated movement. Duration: 3–5 seconds.
8. Press **e** to end and save when 12–16 frames have been captured.
9. Verify the tool reports at least 12 frames saved. If fewer than 8, discard
   and re-capture.
10. Record lux bucket and any notes in `meta.json` (written by tool with `--write-meta`).

### 4.3 Exclusions

Discard and do not count toward the 150 minimum:
- Session where a second face entered the frame.
- Session where the subject was wearing a mask, balaclava, or opaque face covering.
- Session where the frame is visibly motion-blurred (QC will flag this).
- Session where brightness is so low that the face cannot be seen.

### 4.4 Per-session metadata fields

| Field | Value |
|-------|-------|
| `pai_id` | `BF` |
| `ground_truth` | `bona_fide` |
| `subject_id` | `subj_NNN` |
| `presentation_type` | `clip` |
| `capture_device` | camera model string from `cv2.VideoCapture.getBackendName()` |
| `lux_bucket` | `low` / `normal` / `high` |
| `collector_id` | e.g., `collector_a` |
| `session_variant` | e.g., `normal_neutral`, `normal_variation`, `low_light` |
| `frame_count` | actual saved frame count |
| `notes` | any deviation from SOP |

---

## 5. Print Attack Protocol (PAI-ID: PA-01)

**Goal:** Capture 50 presentations where the camera sees a printed photograph
of a face, not a live person.

### 5.1 Source image preparation

- Use a photograph of a real person who is **not** enrolled as a subject
  in the bona fide set and will never be enrolled.
- Preferred: photograph of the same camera framing and lighting as the
  bona fide sessions (face centered, ~40–55 cm equivalent distance,
  neutral expression).
- Print at: **A4 or letter size**, face occupying ≥ 30% of the page.
- Try both: **glossy photo paper** and **matte paper** (at least 25
  presentations each type if possible).
- Print quality: at least 300 DPI. Inkjet and laser both acceptable.
  Record printer type in `notes`.

### 5.2 Subject pool

Use 10 distinct printed photographs (different source faces → 10 different
`subj_NNN`). Each printed face is a separate `subj_NNN`. Capture 5
presentations per printed subject.

### 5.3 Per-presentation procedure

1. Hold the print (or prop it) at **40–55 cm** from the camera lens —
   the same distance as the bona fide subjects.
2. Collector verifies: print face is fully visible, no live faces in frame,
   single face detected.
3. Run tool with `--presentation-type PA-01`.
4. Press **s** to start.
5. Keep the print **static** for 3–4 seconds. Minor unintentional hand
   tremor is acceptable and realistic.
6. Also capture: **slight tilt** (5–15° off vertical) — record angle in `notes`.
7. Press **e** to save. Verify ≥ 12 frames.

### 5.4 Variations to capture across presentations (spread across 50)

| Variation | Target count |
|-----------|-------------|
| Glossy paper, head-on | ~15 |
| Matte paper, head-on | ~15 |
| Glossy paper, 5–15° tilt | ~10 |
| Matte paper, 5–15° tilt | ~10 |

### 5.5 Exclusions

- Do not use photographs of enrolled bona fide subjects.
- Do not use cropped ID photos; use full-frame webcam-like photographs.
- Discard if the print face is significantly smaller than a live face would appear.

---

## 6. Static Screen Attack Protocol (PAI-ID: PA-02)

**Goal:** Capture 50 presentations where the camera sees a static face
photograph displayed on a phone, tablet, or monitor screen.

### 6.1 Source content

- A full-face photograph of a real person not in the bona fide subject pool.
- Display the image full-screen (no browser chrome, no UI decoration).
- Face must fill ≥ 50% of the display screen.
- Use **different source faces** from PA-01 prints if possible (helps
  attribute APCER to PAI category rather than source content).

### 6.2 Devices (cover at least two different device types)

| Device class | Example | Notes |
|---|---|---|
| Phone (5–7 in) | Standard smartphone | Hold at 40–55 cm from camera |
| Tablet (10 in) | iPad / Android tablet | Prop at 40–55 cm from camera |
| Monitor (24 in) | Desktop LCD | Camera faces the monitor at 40–55 cm |

Record exact device model and display diagonal in `notes`.

### 6.3 Per-presentation procedure

1. Display the source image full-screen. Set screen brightness to
   **70–90%** of maximum.
2. Record display brightness setting and device model in `notes`.
3. Position device at 40–55 cm from camera, face centered.
4. Collector verifies: single face visible (the displayed face), correct framing.
5. Run tool with `--presentation-type PA-02`.
6. Press **s**. Hold the device **static** for 3–4 seconds.
7. Press **e**. Verify ≥ 12 frames.

### 6.4 Variations to capture across 50 presentations

| Variation | Target count |
|-----------|-------------|
| Phone, head-on, normal brightness | ~10 |
| Phone, 5–15° angle (moiré test) | ~10 |
| Tablet, head-on | ~10 |
| Tablet, slight angle | ~5  |
| Monitor, head-on | ~10 |
| Monitor, slight angle | ~5  |

### 6.5 Exclusions

- Do not use photographs of enrolled bona fide subjects.
- Do not use video (even if paused) — that is PA-03.
- Discard if there is severe glare on the screen that obscures the face.

---

## 7. Video Replay Attack Protocol (PAI-ID: PA-03)

**Goal:** Capture 50 presentations where the camera sees a video of a
face playing on a device screen.

### 7.1 Source video

- A short (5–15 second) video clip of a real person not in the bona fide
  subject pool.
- The person should be speaking or making natural head movements.
- The face should be centered and fill ≥ 50% of the frame in the source video.
- Video must be **playing** (not paused) during every presentation capture.
- Use different source identities from PA-01 and PA-02 where possible.

### 7.2 Playback devices (cover at least two types)

| Device class | Notes |
|---|---|
| Phone | Hold at 40–55 cm, playing video in full-screen |
| Tablet | Prop at 40–55 cm, full-screen |
| Monitor | Camera faces monitor, video full-screen |

Record playback device, screen diagonal, and video file hash (SHA-256 of
the source file) in `notes`.

### 7.3 Per-presentation procedure

1. Start the video playing in a loop on the device.
2. Position device at 40–55 cm from camera.
3. Collector verifies: video is clearly playing (motion visible), face is
   centered, no live faces in frame.
4. Run tool with `--presentation-type PA-03`.
5. Press **s**. Let the video play for 3–4 seconds of capture.
6. Press **e**. Verify ≥ 12 frames.

### 7.4 Variations across 50 presentations

| Variation | Target count |
|-----------|-------------|
| Phone, natural video, head-on | ~15 |
| Phone, with screen glare | ~5  |
| Tablet, head-on | ~15 |
| Monitor, head-on | ~10 |
| Any device, 5–15° tilt | ~5  |

### 7.5 Exclusions

- Do not use deepfake or AI-generated video.
- Do not use photographs of enrolled bona fide subjects as the video source.
- Discard if the playback device shows visible buffering or a black frame
  during the capture window.

---

## 8. Subject-Disjoint Evaluation Design

This section prevents identity leakage between the bona fide and attack
subject pools.

### 8.1 Pools

| Pool | Who is in it | Who is not |
|------|-------------|------------|
| BF subjects (`subj_001`–`subj_030`) | 30 consenting real people | No one from the attack source pool |
| Attack source faces (printed/displayed) | Different real people, not consenting participants | Not enrolled; not in BF pool |

**Critical rule:** A person who participates as a bona fide subject must never
appear as a source face for any attack presentation (PA-01, PA-02, PA-03).
Verify this manually before finalizing the manifest:
```
for each attack pai_source in manifest.notes:
    assert pai_source not in enrolled_bona_fide_real_identities
```

### 8.2 Why this matters

If a bona fide subject's face also appears as a PA-01 print, the scorer may
produce anomalously high similarity scores for that attack presentation.
This would deflate APCER for that PAI and produce a misleading baseline.

### 8.3 PA-04 (virtual camera, deferred)

PA-04 uses digital injection rather than physical props. When collected in P2,
the injected face identity must also be disjoint from the BF pool.

---

## 9. Manifest Schema

Each presentation row in `manifest.json` must contain all required fields.
The full schema is at `eval/datasets/_template/manifest.schema.json`.

### 9.1 Required fields per sample

| Field | Type | Example | Notes |
|-------|------|---------|-------|
| `sample_id` | string | `BF_subj001_s01` | Unique across collection. Convention: `{PAI}_{subject}_{seq}` |
| `subject_id` | string | `subj_001` | Pattern: `^subj_[0-9]{3}$`. Never a real name. |
| `pai_id` | string | `BF`, `PA-01`, `PA-02`, `PA-03` | Must match physical folder |
| `ground_truth` | string | `bona_fide` or `attack` | `BF` → `bona_fide`; `PA-*` → `attack`. Enforced by harness. |
| `presentation_type` | string | `clip` | Always `clip` for this protocol |
| `path` | string | `bona_fide/subject_001/session_01` | Relative to collection root |
| `frame_count` | int | 14 | Actual saved frame count |
| `capture_device` | string | `MSHD3000 USB Camera` | From `cv2.VideoCapture.getBackendName()` or device label |
| `lux_bucket` | string | `normal` | `low` / `normal` / `high` |
| `collector_id` | string | `collector_a` | Anonymous collector identifier |
| `timestamp` | string | `2026-09-01T14:23:00+01:00` | ISO 8601, local time with offset |

### 9.2 Recommended additional fields

| Field | Type | Example | Notes |
|-------|------|---------|-------|
| `camera_distance_cm` | int | 47 | Estimated eye-to-lens distance |
| `attack_device` | string | `iPhone 14, 6.1 in LCD` | Attack only; device used to display/hold attack |
| `attack_medium` | string | `glossy_photo_paper` | PA-01 only |
| `attack_source_hash` | string | SHA-256 hex | Hash of source image/video file. NOT the person's identity. |
| `attack_angle_deg` | int | 0 | 0 = head-on; positive = tilted |
| `notes` | string | `subject wore glasses` | Free text, no PII |

### 9.3 Sample ID convention

```
{PAI_ID_COMPACT}_{subject_id_compact}_{session_or_attempt_compact}

BF_subj001_s01    ← bona fide, subject 001, session 01
PA01_subj001_a01  ← print attack, subject 001 (of the attack pool), attempt 01
PA02_subj003_a02  ← static screen, subject 003, attempt 02
PA03_subj007_a01  ← video replay, subject 007, attempt 01
```

---

## 10. Directory Structure and Naming Conventions

This follows the layout enforced by `collect_pad_dataset.py`.

```
eval/datasets/face_lock_v1/
├── collection_meta.json          ← per-collection fixed metadata (camera, git SHA, floor tape)
├── manifest.json                 ← generated by tools/generate_manifest.py
├── dataset_qc_report.json        ← generated by tools/qc_dataset.py
│
├── bona_fide/
│   └── subject_001/              ← folder name: subject_NNN (3-digit)
│       ├── session_01/           ← folder name: session_NN (2-digit)
│       │   ├── frame_0001.jpg    ← 4-digit 1-based index, JPEG
│       │   ├── frame_0002.jpg
│       │   └── meta.json         ← optional; written by tool with --write-meta
│       ├── session_02/
│       └── ...
│
└── attacks/
    ├── PA-01_print/
    │   └── subject_001/
    │       └── attempt_01/       ← folder name: attempt_NN (2-digit)
    │           ├── frame_0001.jpg
    │           └── meta.json
    ├── PA-02_screen_static/
    │   └── subject_001/
    │       └── attempt_01/
    └── PA-03_screen_video/
        └── subject_001/
            └── attempt_01/
```

**Naming rules:**
- Subject folder: `subject_NNN` (3 digits, zero-padded). Never use real names.
- Session folder: `session_NN` (2 digits) for BF; `attempt_NN` (2 digits) for attacks.
- Frame file: `frame_NNNN.jpg` (4 digits, 1-based). The tool generates these automatically.
- No other files except `meta.json` and frame JPEGs. Any other file will trigger a QC warning.
- Folder and file names are case-sensitive and must exactly match the patterns above.

---

## 11. `collection_meta.json`

Write this file once before P1 begins. Update if hardware changes.
Store at: `eval/datasets/face_lock_v1/collection_meta.json`.

```json
{
  "collection_id": "face_lock_v1",
  "collection_start_date": "YYYY-MM-DD",
  "target_protocol": "PAD-ISO-LAB-v1",
  "camera_device": "manufacturer and model",
  "camera_index": 0,
  "camera_resolution": "1280x720",
  "camera_backend": "value from cv2.VideoCapture.getBackendName()",
  "git_sha_at_collection_start": "value from git rev-parse HEAD",
  "floor_tape_distance_cm": 47,
  "room_description": "e.g., office, no window, fluorescent overhead",
  "collector_ids": ["collector_a"],
  "lux_rubric": {
    "low": "< 100 lux",
    "normal": "100-500 lux",
    "high": "> 500 lux"
  },
  "qc_notes": []
}
```

---

## 12. QC Process

### 12.1 After every capture session (daily)

```powershell
python tools/qc_dataset.py `
  --collection-root eval/datasets/face_lock_v1
```

Interpret exit codes:
- `0` = PASS. No warnings, no errors.
- `1` = WARNING. Count shortfalls or quality warnings. **Expected during P1.** Review
  the JSON report. Continue collecting unless structural errors are present.
- `2` = FAIL (error-level finding). **Stop and fix before continuing.**

### 12.2 QC checks performed automatically

| Check | Fail threshold |
|-------|---------------|
| Frame count per presentation | < 8 = ERROR (hard minimum from PADSettings.min_frames_passive) |
| Frame count per presentation | < 12 = WARNING |
| Face count per frame | 0 faces = ERROR; > 1 face = ERROR |
| Mean brightness per frame | < 45 = WARNING (aligns with recognition brightness threshold) |
| Laplacian blur variance | < 50 = WARNING |
| Consistent frame dimensions | Mismatch within a presentation = WARNING |
| Sequential frame numbering | Gap in frame sequence = ERROR |
| Unexpected files in folder | Any non-frame, non-meta file = WARNING |
| Manifest path existence | Path in manifest.json does not exist = ERROR |
| Duplicate sample_id | ERROR |
| Duplicate frame hashes across sessions | ERROR (leakage detection) |
| PAI/ground_truth consistency | `BF` with `attack` = ERROR; `PA-*` with `bona_fide` = ERROR |

### 12.3 Manual spot-check (QC reviewer, weekly)

Review a random 10% sample from each PAI category:
1. Open 3 frames from each checked presentation.
2. Confirm BF frames show a live person, not a screen or print.
3. Confirm attack frames show the correct PAI type.
4. Flag any presentation where the face is not the primary subject.
5. Record findings in `collection_meta.json` under `qc_notes`.

### 12.4 Pre-evaluation QC gate

Before running the E3 harness for a baseline candidate:
- `qc_dataset.py` must exit `0` (PASS, zero errors and zero warnings).
- Re-run QC after the final collection session before triggering evaluation.

---

## 13. Validation Process

### 13.1 Manifest generation

After every QC-passing collection session:

```powershell
python tools/generate_manifest.py `
  --collection-root eval/datasets/face_lock_v1 `
  --dry-run
```

Review the dry-run output for unexpected warnings. Then write the manifest:

```powershell
python tools/generate_manifest.py `
  --collection-root eval/datasets/face_lock_v1 `
  --strict
```

`--strict` exits `1` on any error-level finding. Only use `--strict` when
counts are expected to be complete. During partial collection, omit `--strict`.

### 13.2 Manifest validation checklist (manual, before freezing)

- [ ] All `subject_id` values match `^subj_[0-9]{3}$`
- [ ] No `subject_id` appears in both BF and attack sections as a source face identity
- [ ] All `path` values resolve under the collection root
- [ ] All `frame_count` values match actual files on disk
- [ ] `manifest_version` and `collection_id` are correct
- [ ] `created_at` field is set to collection finalization date

---

## 14. Evaluation Commands

These are the exact commands from `docs/PAD_BASELINE_RUNBOOK.md`. Do not
run them until the pre-approval checklist in the Runbook §10 is complete.

```powershell
# 1. Generate manifest (final, strict)
python tools/generate_manifest.py `
  --collection-root eval/datasets/face_lock_v1 `
  --strict

# 2. QC gate (must exit 0)
python tools/qc_dataset.py `
  --collection-root eval/datasets/face_lock_v1

# 3. Run E3 harness
python -m eval.harness.run_pad_eval `
  --protocol eval/protocols/pad_iso_lab_v1.yaml `
  --manifest eval/datasets/face_lock_v1/manifest.json `
  --collection-root eval/datasets/face_lock_v1 `
  --output-dir eval/reports/runs `
  --run-id face_lock_v1_heuristic_standard_001

# 4. Generate REPORT.md
python -m eval.reporting.report_builder `
  --run-dir eval/reports/runs/face_lock_v1_heuristic_standard_001

# 5. (Optional) Live-sim validation
python -m eval.harness.run_pad_eval `
  --protocol eval/protocols/pad_live_sim_v1.yaml `
  --manifest eval/datasets/face_lock_v1/manifest.json `
  --collection-root eval/datasets/face_lock_v1 `
  --output-dir eval/reports/runs `
  --run-id face_lock_v1_live_sim_001
```

**Never use `--skip-missing-media` for a baseline candidate run.**

---

## 15. Versioning

| Artifact | Versioning strategy |
|----------|---------------------|
| `manifest.json` | `manifest_version` field; increment on structural schema changes |
| Protocol YAML | `version` field in `pad_iso_lab_v1.yaml`; change requires new baseline run |
| Collection | `collection_id: face_lock_v1` frozen. New collection = new ID (e.g., `face_lock_v2`) |
| Baseline | `eval/baselines/heuristic_rgb_v1_standard.json`; overwrite only with dual sign-off |
| This document | `Version:` header above; increment on any SOP change before next collection session |

---

## 16. Checksums

### 16.1 Source media (attack artifacts)

For each source image or video used to produce attack presentations:

```powershell
Get-FileHash -Algorithm SHA256 <source_image_or_video_path>
```

Record the SHA-256 hex in the `attack_source_hash` field of the manifest.
This allows the reviewer to verify the exact artifact used without storing
the raw file in the repository.

### 16.2 Manifest integrity

After generating the final manifest, compute and record its hash:

```powershell
Get-FileHash -Algorithm SHA256 eval\datasets\face_lock_v1\manifest.json
```

Record this in `collection_meta.json` under `manifest_sha256` before running
the evaluation harness. The harness `settings_snapshot.json` captures the
git SHA at run time; the manifest hash closes the provenance loop.

### 16.3 Frame files

`qc_dataset.py` computes SHA-256 for every frame and checks for duplicates.
The QC report records any hash collisions. The per-frame hashes are not stored
individually in the manifest (too verbose); they are regenerated at QC time.

---

## 17. Security and Privacy

### 17.1 Storage

| Data type | Storage location | Access control |
|-----------|-----------------|----------------|
| Frame files | `eval/datasets/face_lock_v1/` — **local disk only** | Owner-only file permissions (`icacls` on Windows: `F` for current user, deny all others) |
| Manifest and QC reports | Same directory | Same: local, owner-only |
| Evaluation reports (`eval/reports/runs/`) | Local disk | Owner-only; contain only aggregate metrics, no biometrics |
| Baseline JSON (`eval/baselines/`) | Git repository | Public within the project — contains metrics only, no face data |
| Consent log (real name → subj_NNN mapping) | Separate encrypted file, outside the project tree | Collector only |
| Source attack images/videos | Local disk, outside the project tree | Collector only |

Raw face images must never be committed to git. Confirm `.gitignore` contains:
```
eval/datasets/
```
(already in `.gitignore` as of the repository audit.)

### 17.2 Retention

| Data type | Retention period | Deletion trigger |
|-----------|-----------------|-----------------|
| BF frame files | Retain until baseline is approved and alternative evidence exists | Participant withdrawal, or 12 months after last use |
| Attack frame files | Retain as long as the baseline they produced is in force | 12 months after baseline replacement |
| Consent log | Same as BF frame files for the corresponding subject | Participant withdrawal |
| Evaluation reports | Indefinite (contain no biometric data) | Never (archive only) |
| Baseline JSON | Indefinite | Only replaced by a new approved run |

### 17.3 On participant withdrawal

1. Identify all `subj_NNN` values linked to the withdrawing participant
   via the consent log.
2. Delete all frame files in `bona_fide/subject_NNN/`.
3. Remove all `sample_id` entries for that subject from `manifest.json`.
4. Regenerate `manifest.json` (without the subject).
5. Regenerate and re-run `qc_dataset.py`.
6. If the subject was enrolled (their embedding exists in `encodings/`),
   delete the corresponding `.pkl` file.
7. Record the withdrawal in `collection_meta.json` under `qc_notes`
   using only the anonymous `subj_NNN` identifier.
8. If the baseline was already produced with this subject's data, note it
   in the baseline file as a caveat. A full re-run is required for any
   new production deployment claim.

### 17.4 What is never stored

- Real names, biographic data, or any PII in dataset files, manifests, or reports.
- Raw facial embeddings in evaluation reports (only aggregate distances and metrics).
- Source attack images or videos in the git repository.

---

## 18. Known Limitations of This Protocol

| Limitation | Impact |
|------------|--------|
| Single camera device | APCER/BPCER are camera-specific; metrics may not generalize to other webcams |
| 30 BF subjects | BPCER CI width at n=150, BPCER=5%: ±3.4 pp (Wilson). Acceptable for standard tier; not sufficient for high-tier claim. |
| Heuristic scorer (Laplacian) | Known weakness on PA-01 (print texture). Expect elevated PA-01 APCER. |
| Controlled lab environment | Outdoor or highly variable lighting conditions not covered |
| No 3D mask (PA-05) | Sophisticated spoofs not tested in P1 |
| Attack source identity pool | Diversity of attack source faces not specified; may affect APCER variability |
| No demographic stratification requirement | BPCER may vary by subgroup; document demographics in `collection_meta.json` but do not use them as a gate |

---

## 19. Required Hardware Summary

Before beginning P1 collection, confirm all of the following are available:

### Must-have

| Item | Purpose |
|------|---------|
| Production webcam (same as deployment device) | All BF and attack presentations must use this camera |
| Laptop or desktop with Python 3.11 environment | Collection tool execution |
| Tape measure or ruler | Floor tape at 40–55 cm |
| Floor/desk tape | Mark subject distance |
| Neutral wall or partition | Background for BF and attack capture |
| Lux meter or phone app (lux measurement) | Confirm lux bucket per session |

### For attack capture

| Item | Purpose |
|------|---------|
| Printer + glossy photo paper | PA-01 glossy variant |
| Printer + matte paper | PA-01 matte variant |
| Smartphone (any model) | PA-02 and PA-03 phone variant |
| Tablet (≥ 9 in screen) | PA-02 and PA-03 tablet variant |
| External monitor (optional) | PA-02 and PA-03 monitor variant |
| Source face photographs (printed) | PA-01 source material (10 distinct, non-enrolled faces) |
| Source face video clips (5–15 sec each) | PA-03 source material (10 distinct, non-enrolled faces) |

### Nice-to-have

| Item | Purpose |
|------|---------|
| Adjustable desk lamp | Controlled `high` and `low` lux bucket sessions |
| Tripod or camera stand | Eliminates camera wobble during capture |
| Second collector | Enables simultaneous capture and labeling |

---

## 20. Pre-Collection Checklist

Complete every item before the first P1 session.

```
Environment
  [ ] Python 3.11 environment with requirements.txt installed
  [ ] python tools/collect_pad_dataset.py --help prints usage (no error)
  [ ] python tools/generate_manifest.py --help prints usage
  [ ] python tools/qc_dataset.py --help prints usage
  [ ] python -m eval.harness.run_pad_eval --help prints usage

Camera
  [ ] Camera index 0 opens: python -c "import cv2; c=cv2.VideoCapture(0); print(c.isOpened())"
  [ ] Camera resolution and backend name recorded in collection_meta.json
  [ ] Floor tape at 40-55 cm from camera

Consent and privacy
  [ ] Consent form prepared (paper or digital)
  [ ] Consent log file created (separate from dataset, outside project tree)
  [ ] eval/datasets/ confirmed in .gitignore

Security
  [ ] eval/datasets/face_lock_v1/ has owner-only file permissions
  [ ] No attack source images inside the project tree

Dataset
  [ ] eval/datasets/face_lock_v1/ directory exists
  [ ] collection_meta.json written with camera, git SHA, room details

Attack materials
  [ ] 10 printed face photographs prepared (glossy and matte)
  [ ] 10 source face video clips verified on playback device
  [ ] Source identities confirmed to NOT be enrolled BF subjects
```

---

## 21. Related Documents

| Document | Path |
|----------|------|
| PAD Baseline Runbook (collection → baseline freeze) | `docs/PAD_BASELINE_RUNBOOK.md` |
| PAD evaluation protocol YAML | `eval/protocols/pad_iso_lab_v1.yaml` |
| Live-sim protocol YAML | `eval/protocols/pad_live_sim_v1.yaml` |
| Manifest schema | `eval/datasets/_template/manifest.schema.json` |
| Report template | `eval/reports/template/report_template.md` |
| QC tool | `tools/qc_dataset.py` |
| Manifest generator | `tools/generate_manifest.py` |
| Collection tool | `tools/collect_pad_dataset.py` |
| Evaluation harness | `eval/harness/run_pad_eval.py` |
| Pre-baseline verification (previous audit) | `docs/PRE_BASELINE_VERIFICATION.md` |
