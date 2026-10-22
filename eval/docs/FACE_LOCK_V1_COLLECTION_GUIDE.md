# Face Lock AI — First Production PAD Benchmark Plan

This plan aligns with **PAD-ISO-LAB-v1**, the E1 layout, E3 harness, and E4 reporting. It is written for a first **real** dataset and **approved baseline**, not fixture smoke tests.

**Related artifacts:**

| Artifact | Path |
|----------|------|
| Protocol | `eval/protocols/pad_iso_lab_v1.yaml` |
| Dataset template | `eval/datasets/_template/` |
| Manifest schema | `eval/datasets/_template/manifest.schema.json` |
| Harness | `eval/harness/run_pad_eval.py` |
| Reporting | `eval/reporting/report_builder.py` |
| Baselines | `eval/baselines/` |

---

## 1. Practical dataset collection plan

### Goals

| Goal | Why |
|------|-----|
| Measure **APCER** on PA-01–03 under standard tier (≤ 5%) | Spoof acceptance risk |
| Measure **BPCER** on live users (≤ 5%) | Usability / false lockouts |
| Freeze a **baseline** for regression | Future model/threshold changes |
| Match **production unlock** conditions | Same camera, distance, clip length as `recognize_live()` |

### Phased rollout (recommended)

| Phase | Scope | Duration (typical) | Outcome |
|-------|--------|-------------------|---------|
| **P0 — Pilot** | 10 BF subjects × 3 sessions; 5 attacks × 5 attempts per PAI (PA-01–03) | 1–2 days | Sanity check harness, lighting, manifest QC |
| **P1 — Standard minimum** | Full **PAD-ISO-LAB-v1** counts (below) | 1–2 weeks | First baseline candidate |
| **P2 — Extended** | Add PA-04, lux spread, second device | Optional | Stronger claim before ONNX / high tier |

Do **not** promote a baseline from P0. Use P0 only to fix capture SOP and tooling.

### Collection environment (fix per collection)

Document once and reuse for every session:

- **Device:** e.g. laptop integrated webcam (match primary user hardware).
- **OS / app build:** git SHA at collection start.
- **Distance:** mark floor tape — e.g. 40–55 cm eye-to-camera.
- **Background:** neutral wall; avoid strong backlight.
- **Resolution:** native camera default; do not crop in capture tool.
- **Unlock simulation:** capture **clips**, not single photos (except optional PAD-REG later).

### Roles

| Role | Responsibility |
|------|----------------|
| **Collector** | Captures clips per SOP; no labeling |
| **Labeler** | Assigns PAI, `ground_truth`, `lux_bucket`; builds manifest |
| **QC reviewer** | Spot-checks frames, manifest, failed presentations |
| **Evaluation engineer** | Runs E3/E4, signs baseline |

---

## 2. Minimum sample counts

From `eval/protocols/pad_iso_lab_v1.yaml` (**standard tier**):

| Category | PAI | Min presentations | Min subjects | Notes |
|----------|-----|-------------------|--------------|--------|
| Bona fide | BF | **150** | **30** | ~5 presentations per subject |
| Print | PA-01 | **50** | **10** | ~5 per subject |
| Static screen | PA-02 | **50** | **10** | |
| Video replay | PA-03 | **50** | **10** | |
| Virtual camera | PA-04 | 30 (optional) | 6 | Not in pooled APCER gate; collect after PA-01–03 stable |

**Pooled APCER gate** uses **PA-01 + PA-02 + PA-03 only** (150 attack presentations minimum for sign-off).

### Per-presentation frame minimum

| Requirement | Value |
|-------------|--------|
| Protocol `min_frames_passive` | 8 |
| **Recommended captured frames** | **12–16** per presentation |
| Frame naming | `frame_0001.jpg` … sequential |
| One presentation | One simulated unlock attempt |

### Pilot (P0) — optional but strongly advised

| Category | Presentations | Subjects |
|----------|---------------|----------|
| BF | 30 | 10 |
| Each PA-01, PA-02, PA-03 | 15 each | 5 |

---

## 3. Capture instructions

### 3.1 Bona fide (BF)

**Intent:** Live user at the camera, as in normal unlock.

1. Subject sits at marked distance, face centered, eyes open.
2. Start clip capture; over **3–5 seconds**:
   - slight natural head motion (small nod or turn, within ±15°);
   - normal blink allowed;
   - no exaggerated expression.
3. Capture **12–16 frames** (~15 fps equivalent if extracting from video).
4. Repeat for **5 sessions** per subject under varied but realistic conditions:
   - 3× `lux_bucket: normal`
   - 1× `low` (dim room, no direct lamp on face)
   - 1× `high` (bright ambient, not blown out)
5. **Exclude:** masks, sunglasses, hands over face, another person in frame.
6. **Do not** use the same session for both BF and attack.

### 3.2 PA-01 — Printed photo attack

**Intent:** Flat printed face held to camera (most common presentation attack).

1. Source: high-quality color print or photo paper of subject’s face (or public face photo for attack-subject set — do not mix with BF identity unless study design requires it).
2. Hold print **flat**, fill ~40–60% of frame, at same distance as BF.
3. Optional variation across attempts (document in `notes`):
   - slight tilt (5–15°)
   - glossy vs matte
   - A4 vs phone-sized print
4. Capture 12–16 frames; **minimal motion** (simulate someone holding print steady).
5. **Ground truth:** always `attack`, `pai_id: PA-01`.

### 3.3 PA-02 — Static phone screen attack

**Intent:** Still image of face displayed on a phone/tablet/monitor.

1. Display a **static** face image full-screen on phone (brightness 70–90%).
2. Hold phone at same distance as BF; camera sees screen, not live person.
3. Variations across attempts:
   - phone model / LCD vs OLED
   - slight moiré angle (5–10° off-axis) on some attempts
4. 12–16 frames per presentation.
5. `pai_id: PA-02`.

### 3.4 PA-03 — Video replay attack

**Intent:** Moving replay on screen (higher fidelity than PA-02).

1. Play a **video clip** of the face (loop or 5–10 s segment) on phone/laptop screen.
2. Include natural motion in source video (blink, small turn).
3. Variations:
   - different replay devices
   - with/without screen glare
4. 12–16 frames while video plays (not paused on one frame — that is PA-02).
5. `pai_id: PA-03`.

### 3.5 PA-04 — Virtual camera attack

**Intent:** Software-injected stream (OBS Virtual Camera, ManyCam, etc.) — **optional for first baseline**.

1. Feed pre-recorded face video into virtual camera; select it as system camera (if your test setup allows).
2. Face Lock eval should use the **same camera index** as production.
3. Variations: different virtual camera tools, compressed vs raw source.
4. `pai_id: PA-04` — report separately; not in pooled standard APCER gate.

---

## 4. Folder structure

Use collection ID **`face_lock_v1`** (or `face_lock_lab_2026_05` with date).

```text
eval/datasets/face_lock_v1/          ← LOCAL ONLY; never commit media
├── manifest.json
├── collection_meta.json           ← optional: device, room, collectors, dates
├── bona_fide/
│   ├── subject_001/
│   │   ├── session_01/
│   │   │   ├── frame_0001.jpg
│   │   │   ├── …
│   │   │   ├── frame_0012.jpg
│   │   │   └── meta.json          ← optional
│   │   ├── session_02/
│   │   └── …
│   └── subject_030/
└── attacks/
    ├── PA-01_print/
    │   └── subject_001/
    │       ├── attempt_01/
    │       └── attempt_05/
    ├── PA-02_screen_static/
    ├── PA-03_screen_video/
    └── PA-04_virtual_camera/        ← optional phase P2
```

**Rules:**

- `subject_XXX` = pseudonym (`subj_001` in manifest).
- BF: `session_NN`; attacks: `attempt_NN`.
- Only **one presentation** per folder.
- Paths in manifest are **relative** to collection root.

---

## 5. Manifest examples

### 5.1 Top-level `manifest.json` (excerpt)

```json
{
  "manifest_version": "1.0.0",
  "collection_id": "face_lock_v1",
  "created_at": "2026-05-30",
  "description": "First lab PAD collection for heuristic_rgb_v1 standard tier",
  "protocol_ids": ["PAD-ISO-LAB-v1"],
  "samples": []
}
```

Populate `samples` with one entry per presentation folder.

### 5.2 Bona fide sample

```json
{
  "sample_id": "BF_subj001_s03",
  "pai_id": "BF",
  "ground_truth": "bona_fide",
  "subject_id": "subj_001",
  "path": "bona_fide/subject_001/session_03",
  "presentation_type": "clip",
  "capture_device": "laptop_integrated",
  "lux_bucket": "normal",
  "frame_count": 12,
  "notes": "Windows 11; built-in cam; collector A"
}
```

### 5.3 Attack samples

```json
{
  "sample_id": "PA01_subj007_a02",
  "pai_id": "PA-01",
  "ground_truth": "attack",
  "subject_id": "subj_007",
  "path": "attacks/PA-01_print/subject_007/attempt_02",
  "presentation_type": "clip",
  "capture_device": "laptop_integrated",
  "lux_bucket": "normal",
  "frame_count": 14,
  "notes": "matte A4 print; 10deg tilt"
}
```

```json
{
  "sample_id": "PA03_subj004_a01",
  "pai_id": "PA-03",
  "ground_truth": "attack",
  "subject_id": "subj_004",
  "path": "attacks/PA-03_screen_video/subject_004/attempt_01",
  "presentation_type": "clip",
  "capture_device": "laptop_integrated",
  "lux_bucket": "low",
  "frame_count": 16,
  "notes": "iPhone 14 replay; screen glare mild"
}
```

### 5.4 Optional `meta.json` (per session folder)

```json
{
  "collector_id": "collector_a",
  "captured_at_utc": "2026-05-30T14:22:00Z",
  "fps_source": 15,
  "room_lux_estimate": 320,
  "face_fill_ratio": 0.45
}
```

---

## 6. Labeling rules

| Field | Rule |
|-------|------|
| `sample_id` | Unique; pattern `{PAI or BF}_{subject}_{session/attempt}` |
| `pai_id` | `BF` or `PA-01` … `PA-04` — must match folder |
| `ground_truth` | `bona_fide` iff `pai_id == BF`; else `attack` |
| `subject_id` | `subj_NNN` three digits; **never** real names in manifest |
| `path` | Relative path; must exist on disk before eval |
| `presentation_type` | Always `clip` for PAD-ISO-LAB |
| `lux_bucket` | `low` \| `normal` \| `high` — collector judgment, consistent rubric |
| `frame_count` | Actual frame count; must be ≥ 8 |

**Consistency checks:**

- BF folder → `pai_id: BF` + `ground_truth: bona_fide`.
- `attacks/PA-02_*` → never label `PA-01`.
- Paused video frame → **PA-02**, not PA-03.
- Live person at camera → **BF**, never attack folder.

---

## 7. Quality-control checks

### 7.1 Per-presentation (before manifest entry)

| Check | Pass criteria |
|-------|----------------|
| Frame count | ≥ 12 frames (≥ 8 hard minimum) |
| Face visible | Face in ≥ 80% of frames |
| Exposure | Not majority blown out or < 20 mean brightness |
| Wrong modality | Print in PA-01 folder only, etc. |
| Duplicate | Same frames as another `sample_id` |

### 7.2 Manifest-level (automated + manual)

| Check | How |
|-------|-----|
| Schema | All required fields; `subject_id` matches `^subj_[0-9]{3}$` |
| Counts | ≥ 150 BF, ≥ 50 per PA-01–03, ≥ 30 subjects BF |
| Path safety | Paths stay under collection root |
| PAI / GT alignment | BF/attack rules above |
| Balance | Each `lux_bucket` ≥ 15% of BF set (target) |

### 7.3 Post-harness QC

| Check | Action if fail |
|-------|----------------|
| High `ERROR` rate | Re-capture; check corrupt JPEGs |
| All attacks `LIVE` | Stop — capture or model broken |
| All BF `ATTACK` | Check thresholds / lighting SOP |
| Latency P95 > 1200 ms | Note in report; optimize later |

---

## 7.4 Dataset capture tool

Use the OpenCV collector (does not run PAD or write manifests):

```powershell
python tools/collect_pad_dataset.py `
  --collection-root eval/datasets/face_lock_v1 `
  --collector-id collector_a
```

Prompts for `subject_id` (`subj_NNN`) and presentation type (`BF`, `PA-01`, …).  
Keyboard: **s** start, **e** end/save, **q** quit. Default **12–16** frames (`--min-frames` / `--max-frames`).

### Generate manifest.json

After capture and QC, build the manifest from disk:

```powershell
python tools/generate_manifest.py `
  --collection-root eval/datasets/face_lock_v1 `
  --strict
```

Use `--dry-run` to preview counts and warnings without writing. The tool prints BF / PA-01–04 statistics and warns on missing frames, invalid folders, and bad subject directory names.

### Dataset QC (before evaluation)

Validate structure, frames, manifest consistency, and image quality:

```powershell
python tools/qc_dataset.py `
  --collection-root eval/datasets/face_lock_v1
```

Writes `dataset_qc_report.json` and prints **PASS**, **WARNING**, or **FAIL**. Exit codes: `0` = PASS, `1` = WARNING, `2` = FAIL. Run after manifest generation and before E3 harness.

---

## 8. Evaluation procedure (E3 + E4 only)

### Prerequisites

- Dataset on disk: `eval/datasets/face_lock_v1/`
- `manifest.json` complete
- `pip install -r requirements.txt -r requirements-dev.txt`
- Git commit SHA recorded (harness snapshot does this)

### Step 1 — Run harness (PAD-ISO-LAB)

```powershell
cd c:\face_detection

python -m eval.harness.run_pad_eval `
  --protocol eval/protocols/pad_iso_lab_v1.yaml `
  --manifest eval/datasets/face_lock_v1/manifest.json `
  --collection-root eval/datasets/face_lock_v1 `
  --output-dir eval/reports/runs `
  --run-id face_lock_v1_heuristic_standard_001
```

**Before first real baseline:** edit protocol overrides only if intentional — e.g. confirm `passive_model_id: heuristic_rgb_v1` and `assurance_tier: standard`.

Outputs:

```text
eval/reports/runs/face_lock_v1_heuristic_standard_001/
├── summary.json
├── results.jsonl
├── settings_snapshot.json
```

### Step 2 — Generate report (E4)

```powershell
python -m eval.reporting.report_builder `
  --run-dir eval/reports/runs/face_lock_v1_heuristic_standard_001
```

Produces `REPORT.md` plus console **PASS/FAIL** (tier gates and baseline regression when a baseline exists).

### Step 3 — Optional PAD-LIVE-SIM

Same manifest; different protocol — validates session feeding:

```powershell
python -m eval.harness.run_pad_eval `
  --protocol eval/protocols/pad_live_sim_v1.yaml `
  --manifest eval/datasets/face_lock_v1/manifest.json `
  --collection-root eval/datasets/face_lock_v1 `
  --output-dir eval/reports/runs `
  --run-id face_lock_v1_live_sim_001
```

---

## 9. First baseline report

### When to freeze a baseline

All of the following:

| Criterion | Target |
|-----------|--------|
| P1 sample counts | Met |
| Tier gate (standard) | `overall: pass` in E3/E4 |
| QC | No open capture SOP violations |
| Review | Evaluation + security sign-off on `REPORT.md` |

### How to create the baseline file

1. Copy the approved run summary:

```text
eval/reports/runs/face_lock_v1_heuristic_standard_001/summary.json
  → eval/baselines/heuristic_rgb_v1_standard.json
```

2. Naming convention: `{passive_model_id}_{assurance_tier}.json`

3. Keep `REPORT.md` and full run folder archived (local secure storage, not public git).

4. Re-run E4 on future runs — baseline comparison appears automatically when `eval/baselines/heuristic_rgb_v1_standard.json` exists.

### First baseline report contents (what to read)

| Section | Use |
|---------|-----|
| Executive summary | APCER/BPCER vs 5% targets |
| APCER by PAI | Weakest attack type |
| BPCER breakdown | INCONCLUSIVE vs ATTACK on live users |
| Confusion matrix | FN = spoof accepted; FP = user rejected |
| Baseline comparison | Empty first time; populated on run 2+ |
| Settings snapshot | Reproduce exact thresholds |

---

## 10. Interpreting APCER and BPCER

### Definitions (fail_closed_v1)

| Metric | Meaning | “Good” direction |
|--------|---------|------------------|
| **APCER** | Fraction of **attacks** classified as **LIVE** | **Lower** (0% ideal) |
| **BPCER** | Fraction of **bona fide** **not** LIVE (ATTACK / INCONCLUSIVE / ERROR) | **Lower** |

Example: APCER = 0.04 → **4%** of attacks incorrectly accepted as live.  
BPCER = 0.06 → **6%** of real users rejected or challenged — **fails** standard 5% gate.

### Standard tier gates (your protocol)

| Metric | Max allowed |
|--------|-------------|
| Pooled APCER (PA-01–03) | **5%** |
| BPCER | **5%** |
| Latency P95 | **1200 ms** |

### How to read per-PAI APCER

| Pattern | Likely cause |
|---------|----------------|
| PA-01 high, others low | Print texture fools heuristic |
| PA-03 high | Motion replay; screen refresh |
| PA-02 high | Moiré / bezel; static image too clean |
| All high | Thresholds too loose or capture not attack-like |

### BPCER interpretation

| Dominant outcome on BF | Likely cause |
|------------------------|--------------|
| `INCONCLUSIVE` | Borderline scores; active challenge not completed in clip eval |
| `ATTACK` | Too strict thresholds or poor lighting |
| `ERROR` | Too few frames, corrupt images, or scorer exceptions |

### Business framing (for stakeholders)

| Metric | Security / product angle |
|--------|---------------------------|
| APCER | **Spoof bypass rate** — “1 in 25 attacks might unlock” at 4% |
| BPCER | **User friction** — “1 in 20 legitimate unlocks fails liveness” at 5% |

### Regression vs baseline (after first freeze)

E4 flags regression if APCER or BPCER **increases by more than 0.01** (1 percentage point absolute) vs baseline. A small increase within tolerance is not a regression; above tolerance requires investigation before release.

---

## Recommended timeline (summary)

| Week | Activity |
|------|----------|
| 1 | Finalize SOP; P0 pilot; fix capture tool |
| 2–3 | P1 BF + PA-01–03 collection |
| 4 | QC manifest; E3 run; E4 report |
| 5 | Review; if pass → copy `summary.json` to `eval/baselines/` |

---

## Security and compliance reminders

- Store `eval/datasets/face_lock_v1/` on encrypted disk; exclude from git (already in `.gitignore`).
- Consent for biometric collection where required.
- Attack media may use **different** subjects than BF if that matches your threat model; document in `collection_meta.json`.
