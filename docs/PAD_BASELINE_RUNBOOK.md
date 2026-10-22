# PAD Baseline Runbook — `heuristic_rgb_v1` / Standard Tier

**Document version:** 1.0.0  
**Protocol:** PAD-ISO-LAB-v1  
**Model under test:** `heuristic_rgb_v1`  
**Assurance tier:** `standard`  
**Target baseline file:** `eval/baselines/heuristic_rgb_v1_standard.json`  
**Policy:** `fail_closed_v1` (INCONCLUSIVE and ERROR collapse to ATTACK)

---

## 1. Purpose

This runbook is the single authoritative procedure for producing, approving, and
freezing the first PAD baseline for Face Lock AI. Follow it in order. Do not
skip phases. Do not freeze a baseline from a pilot (P0) run.

Baseline approval certifies that `heuristic_rgb_v1` running under
`PADSettings` / standard tier meets the security and usability gates defined
in `eval/protocols/pad_iso_lab_v1.yaml` before any production deployment.

---

## 2. Prerequisites

### 2.1 Software

```powershell
# Confirm Python version
python --version        # must be 3.11.x

# Install all dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Confirm eval harness is importable
python -c "from eval.harness.run_pad_eval import main; print('OK')"
```

### 2.2 Tools required on disk

| Script | Location | Purpose |
|--------|----------|---------|
| `collect_pad_dataset.py` | `tools/` | Frame capture with labeling |
| `generate_manifest.py` | `tools/` | Build `manifest.json` from disk |
| `qc_dataset.py` | `tools/` | Pre-eval QC gate |
| `run_pad_eval.py` | `eval/harness/` | E3 evaluation harness |
| `report_builder.py` | `eval/reporting/` | E4 REPORT.md + tier gate |

### 2.3 Roles

| Role | Responsibility |
|------|----------------|
| **Collector** | Captures clips per SOP; does not label |
| **Labeler** | Assigns `pai_id`, `lux_bucket`, writes `notes`; builds manifest |
| **QC reviewer** | Spot-checks frames, runs `qc_dataset.py`, signs off counts |
| **Evaluation engineer** | Runs E3/E4, interprets metrics, authors `REPORT.md` |
| **Security reviewer** | Signs `REPORT.md` sign-off table before baseline freeze |

Both the evaluation engineer and security reviewer must sign off. One person
cannot fill both roles for a production baseline.

### 2.4 Current dataset status

`eval/datasets/face_lock_v1/dataset_qc_report.json` (last run 2026-05-30):

| Category | Have | Need (standard) | Gap |
|----------|------|-----------------|-----|
| BF | 4 presentations, 2 subjects | 150 / 30 | **146 presentations, 28 subjects** |
| PA-01 | 0 | 50 / 10 | **50 presentations, 10 subjects** |
| PA-02 | 1 presentation, 1 subject | 50 / 10 | **49 presentations, 9 subjects** |
| PA-03 | 0 | 50 / 10 | **50 presentations, 10 subjects** |
| PA-04 | 0 | 30 / 6 (optional) | deferred to Phase P2 |

**The current dataset is at P0 (pilot) scale. No baseline can be approved
until P1 counts are met.**

---

## 3. Dataset requirements

### 3.1 Minimum sample counts (standard tier)

From `eval/protocols/pad_iso_lab_v1.yaml`:

| Category | PAI ID | Min presentations | Min subjects | Pooled APCER gate? |
|----------|--------|-------------------|--------------|--------------------|
| Bona fide | BF | **150** | **30** | No (BPCER) |
| Print | PA-01 | **50** | **10** | Yes |
| Static screen | PA-02 | **50** | **10** | Yes |
| Video replay | PA-03 | **50** | **10** | Yes |
| Virtual camera | PA-04 | 30 (optional) | 6 | No (report separately) |

**Total minimum for first baseline:** 150 BF + 150 attacks (PA-01–03) = **300 presentations**.

### 3.2 Per-presentation frame requirements

| Requirement | Value | Source |
|-------------|-------|--------|
| Minimum frames (hard) | **8** | `PADSettings.min_frames_passive` |
| Recommended frames | **12–16** | Collection guide |
| Frame naming | `frame_0001.jpg` … | `clip_loader.py` pattern |
| Presentation type | `clip` | `pad_iso_lab_v1.yaml` |

### 3.3 Capture environment — fix and document per collection

Record the following in `eval/datasets/face_lock_v1/collection_meta.json`
before starting capture:

- Camera device and index (must match `WEBCAM_SETTINGS.camera_index`)
- Git SHA at collection start (`git rev-parse HEAD`)
- Subject distance: mark floor tape at 40–55 cm eye-to-camera
- Background: neutral wall, no strong backlight
- `lux_bucket` rubric: `low` ≤ 100 lux, `normal` 100–500 lux, `high` > 500 lux

---

## 4. Collection workflow

### Step C1 — Start a capture session

```powershell
python tools/collect_pad_dataset.py `
  --collection-root eval/datasets/face_lock_v1 `
  --collector-id collector_a
```

The tool prompts for `subject_id` (`subj_NNN`, three digits) and presentation
type (`BF`, `PA-01`, `PA-02`, `PA-03`, `PA-04`). Keyboard: **s** start,
**e** end/save clip, **q** quit. Default 12–16 frames per clip.

**Folder layout produced (example):**

```
eval/datasets/face_lock_v1/
├── bona_fide/subject_001/session_01/frame_0001.jpg … frame_0016.jpg
└── attacks/PA-02_screen_static/subject_001/attempt_01/frame_0001.jpg …
```

### Step C2 — Bona fide capture SOP

1. Subject at marked distance (40–55 cm), face centered, eyes open.
2. Capture over 3–5 seconds with slight natural head motion (≤ 15° nod or turn).
3. 12–16 frames per session.
4. **5 sessions per subject:** 3× `normal`, 1× `low`, 1× `high` lighting.
5. Exclude: masks, sunglasses, occlusions, secondary persons in frame.

### Step C3 — Attack capture SOP (per PAI)

| PAI | Key requirement | Variation to document in `notes` |
|-----|----------------|----------------------------------|
| PA-01 | Flat print, same distance as BF, minimal motion | glossy vs matte; tilt 5–15° |
| PA-02 | Static face image on phone/tablet, brightness 70–90% | LCD vs OLED; slight moiré angle |
| PA-03 | Playing video of face on screen (not paused) | different devices; with/without glare |
| PA-04 | Virtual camera injection (OBS/ManyCam) | defer to P2; document tool version |

Each attack attempt = one independent presentation folder (`attempt_NN`).

### Step C4 — Labeling rules

| Field | Rule |
|-------|------|
| `sample_id` | Unique per collection: `{PAI}_{subject}_{session/attempt}` |
| `pai_id` | `BF` or `PA-01`…`PA-04` — must match physical folder |
| `ground_truth` | `bona_fide` iff `pai_id == BF`; else `attack` |
| `subject_id` | `subj_NNN` — never real names |
| `path` | Relative to collection root; folder must exist on disk |
| `presentation_type` | Always `clip` for PAD-ISO-LAB |
| `lux_bucket` | `low` / `normal` / `high` — per rubric in §3.3 |
| `frame_count` | Actual frame count; must be ≥ 8 |

Consistency rule enforced by harness: `pai_id BF` → `ground_truth: bona_fide`;
`pai_id PA-*` → `ground_truth: attack`. Violations abort the run.

---

## 5. Manifest generation

### Step M1 — Generate manifest from disk

Run after capture is complete for each phase. Use `--strict` to hard-fail on
count shortfalls (required before any eval run):

```powershell
python tools/generate_manifest.py `
  --collection-root eval/datasets/face_lock_v1 `
  --strict
```

Use `--dry-run` first to preview warnings without writing:

```powershell
python tools/generate_manifest.py `
  --collection-root eval/datasets/face_lock_v1 `
  --dry-run
```

The tool prints BF / PA-01–PA-04 statistics and warns on:
- Fewer frames than `min_frames_passive` (8)
- Folder names that don't match expected PAI pattern
- Subject directory names not matching `subj_NNN`
- Missing required PAI categories

Output: `eval/datasets/face_lock_v1/manifest.json`

### Step M2 — Validate manifest schema

The manifest must conform to `eval/datasets/_template/manifest.schema.json`.
Required top-level fields: `manifest_version`, `collection_id`, `samples`.
Required per-sample: `sample_id`, `pai_id`, `ground_truth`, `subject_id`,
`path`, `presentation_type`.

Verify manually that:
- `subject_id` matches `^subj_[0-9]{3}$` for all samples
- All `path` values resolve under the collection root
- `created_at` is set to the collection finalization date

---

## 6. QC workflow

### Step Q1 — Run automated QC

```powershell
python tools/qc_dataset.py `
  --collection-root eval/datasets/face_lock_v1
```

Writes `eval/datasets/face_lock_v1/dataset_qc_report.json`.

Exit codes: `0` = PASS, `1` = WARNING, `2` = FAIL.

**Do not proceed to evaluation if exit code is `2`.**

### Step Q2 — Interpret QC report

| QC check | Pass criteria |
|----------|--------------|
| Frame count per presentation | ≥ 12 (soft); ≥ 8 (hard minimum) |
| Face visible | Face in ≥ 80% of frames (manual spot-check) |
| `mean_brightness` | > 45 per frame (aligns with `RecognitionSettings.brightness_threshold`) |
| `mean_blur_variance` | > 50 (lower = likely motion-blurred or under-exposed) |
| `inconsistent_dimensions` | False for all presentations |
| Category counts | Meet §3.1 minimums |
| Subject counts | Meet §3.1 minimums |
| Missing category folders | None for PA-01, PA-02, PA-03 |

Current dataset has 13 warnings and 0 errors. All warnings are count
shortfalls — expected at P0. **Re-run QC after P1 collection completes.**

### Step Q3 — Manual spot-check (QC reviewer)

Review a random 10% sample from each PAI category:
- Open 3 frames from each checked presentation
- Confirm BF frames show a live face, not a screen or print
- Confirm attack frames show the correct PAI type
- Flag any presentations where the face is not the primary subject

Document findings in `eval/datasets/face_lock_v1/collection_meta.json`
under a `qc_notes` field.

---

## 7. Evaluation workflow (E3)

### Step E1 — Confirm effective PAD settings

The harness merges protocol YAML overrides onto `PAD_SETTINGS`. Verify the
effective settings before running:

| Setting | Required value for this baseline |
|---------|----------------------------------|
| `passive_model_id` | `heuristic_rgb_v1` |
| `assurance_tier` | `standard` |
| `mode` | `passive_then_active_on_borderline` |
| `min_frames_passive` | `8` |
| `passive_pass_threshold` | `0.62` |
| `passive_reject_threshold` | `0.38` |
| `borderline_low` | `0.45` |
| `borderline_high` | `0.58` |
| `evaluation_policy` | `fail_closed_v1` |

These values are set in `eval/protocols/pad_iso_lab_v1.yaml`. Do not change
`src/config.py` thresholds between collection and evaluation.

### Step E2 — Run primary evaluation (PAD-ISO-LAB-v1)

Choose a descriptive `--run-id` that includes the model, tier, and sequence
number. Replace `001` with the actual run number if re-running:

```powershell
cd c:\face_detection

python -m eval.harness.run_pad_eval `
  --protocol eval/protocols/pad_iso_lab_v1.yaml `
  --manifest eval/datasets/face_lock_v1/manifest.json `
  --collection-root eval/datasets/face_lock_v1 `
  --output-dir eval/reports/runs `
  --run-id face_lock_v1_heuristic_standard_001
```

To skip presentations whose media cannot be loaded (use only in exploratory
runs, never for a baseline candidate):

```powershell
  --skip-missing-media
```

To cap frames for a quick smoke test (not for baseline):

```powershell
  --max-frames-per-clip 12
```

**Artifacts written to:**

```
eval/reports/runs/face_lock_v1_heuristic_standard_001/
├── summary.json          ← aggregated metrics (validates against summary_schema.json)
├── results.jsonl         ← one JSON object per presentation
└── settings_snapshot.json ← full PADSettings + Python/OpenCV/git environment
```

### Step E3 — Run live-sim validation (PAD-LIVE-SIM-v1)

Optional but strongly recommended for the first baseline. Uses the same
manifest; validates the session-buffered unlock path:

```powershell
python -m eval.harness.run_pad_eval `
  --protocol eval/protocols/pad_live_sim_v1.yaml `
  --manifest eval/datasets/face_lock_v1/manifest.json `
  --collection-root eval/datasets/face_lock_v1 `
  --output-dir eval/reports/runs `
  --run-id face_lock_v1_live_sim_001
```

### Step E4 — Inspect raw summary

```powershell
type eval\reports\runs\face_lock_v1_heuristic_standard_001\summary.json
```

Check that `apcer_overall`, `bpcer_overall`, and `tier_gate` fields are
present and numeric before proceeding to report generation.

---

## 8. Reporting workflow (E4)

### Step R1 — Generate REPORT.md and tier gate result

```powershell
python -m eval.reporting.report_builder `
  --run-dir eval/reports/runs/face_lock_v1_heuristic_standard_001
```

Optional: point at a specific baseline for regression comparison (not
applicable for the first run — no baseline exists yet):

```powershell
python -m eval.reporting.report_builder `
  --run-dir eval/reports/runs/face_lock_v1_heuristic_standard_002 `
  --baseline eval/baselines/heuristic_rgb_v1_standard.json
```

Console output:

```
Report written: .../REPORT.md | tier_gate=pass | regression=False | overall=PASS
```

Exit code `0` = PASS, `1` = FAIL. A CI gate can use this exit code directly.

### Step R2 — Review REPORT.md

Open `eval/reports/runs/face_lock_v1_heuristic_standard_001/REPORT.md` and
verify every placeholder was filled (no literal `{{ ... }}` remaining).

Check all sections:
- Executive summary: APCER/BPCER/latency vs gates
- APCER by PAI: identify which attack type has the highest acceptance rate
- BPCER breakdown: distinguish INCONCLUSIVE vs ATTACK outcomes on live users
- Confusion matrix: FN count = spoofs accepted; FP count = users rejected
- Settings snapshot: thresholds match §7 Step E1 table
- Sign-off table: leave blank until §9 approval

### Step R3 — Baseline regression (run 2+)

After the first baseline is frozen, every subsequent run automatically
includes a regression section when `eval/baselines/heuristic_rgb_v1_standard.json`
exists. The comparison uses `BaselineCompareConfig` defaults:

| Tolerance | Value |
|-----------|-------|
| Max APCER increase | 0.01 (1 pp absolute) |
| Max BPCER increase | 0.01 (1 pp absolute) |
| Model must match | Yes |
| Tier must match | Yes |

A delta above tolerance is reported as **REGRESSION** and causes exit code 1.

---

## 9. PASS criteria — `heuristic_rgb_v1_standard`

All four conditions must hold simultaneously. A single failure blocks baseline
approval.

| Gate | Metric | Maximum allowed | Source |
|------|--------|-----------------|--------|
| **APCER gate** | Pooled APCER (PA-01 + PA-02 + PA-03) | **≤ 5.00%** | `pad_iso_lab_v1.yaml` → `apcer_max_pooled: 0.05` |
| **BPCER gate** | BPCER (all bona fide) | **≤ 5.00%** | `bpcer_max: 0.05` |
| **Latency gate** | P95 latency (ms) | **≤ 1200 ms** | `latency_ms_p95_max: 1200` |
| **Regression gate** | APCER/BPCER vs baseline | **≤ +1.0 pp** each | `BaselineCompareConfig` (first run: N/A) |

### Metric definitions (fail_closed_v1)

**APCER** = (attack presentations classified as `LIVE`) / (total attack presentations)

**BPCER** = (bona fide presentations NOT classified as `LIVE`) / (total bona fide presentations)

`INCONCLUSIVE` and `ERROR` outcomes count as not-LIVE for both metrics.

### Per-PAI APCER interpretation

| Pattern | Likely cause | Action |
|---------|-------------|--------|
| PA-01 high | Print texture fools Laplacian heuristic | Adjust `passive_reject_threshold` or await ONNX model |
| PA-02 high | Clean static image scores high on texture | Lower `passive_pass_threshold` cautiously |
| PA-03 high | Replay motion mimics live motion signal | Increase `min_frames_passive`; capture more PA-03 |
| All high | Thresholds too loose or SOP deviation | Audit capture SOP before adjusting thresholds |

### BPCER interpretation

| Dominant BF outcome | Likely cause | Action |
|--------------------|-------------|--------|
| `INCONCLUSIVE` | Borderline scores; active challenge not resolved in clip | Normal for `passive_then_active_on_borderline` mode |
| `ATTACK` | Thresholds too strict or poor lighting in BF set | Check `lux_bucket: low` subset; review `passive_reject_threshold` |
| `ERROR` | Too few frames or corrupt JPEGs | Re-capture; ensure ≥ 12 frames per presentation |

---

## 10. Baseline approval workflow

### Step A1 — Pre-approval checklist

Complete every item before requesting sign-off:

- [ ] P1 sample counts met (§3.1 table)
- [ ] `qc_dataset.py` exit code `0` (PASS) on final dataset
- [ ] Harness run completed with `--run-id face_lock_v1_heuristic_standard_<N>`
- [ ] No `--skip-missing-media` flag used in baseline candidate run
- [ ] `summary.json` `tier_gate.overall == "pass"`
- [ ] `REPORT.md` generated; all placeholders filled
- [ ] No literal `{{ ... }}` tokens in `REPORT.md`
- [ ] Settings snapshot confirms `passive_model_id: heuristic_rgb_v1`
- [ ] Settings snapshot confirms `assurance_tier: standard`
- [ ] Git SHA at collection start matches `settings_snapshot.json` `environment.git_sha`
- [ ] `collection_meta.json` present with device, distance, and collector IDs
- [ ] QC reviewer has spot-checked ≥ 10% of presentations per PAI
- [ ] No open capture SOP violations

### Step A2 — Sign-off

Fill in the sign-off table in `REPORT.md`:

| Role | Name | Date |
|------|------|------|
| Evaluation engineer | | |
| Security reviewer | | |

Both signatures are required. The security reviewer must be a different person
from the evaluation engineer for any production deployment claim.

### Step A3 — Freeze the baseline

Copy the approved run summary to the baselines directory.
The filename convention is `{passive_model_id}_{assurance_tier}.json`:

```powershell
copy `
  eval\reports\runs\face_lock_v1_heuristic_standard_001\summary.json `
  eval\baselines\heuristic_rgb_v1_standard.json
```

Verify the file was created:

```powershell
type eval\baselines\heuristic_rgb_v1_standard.json | python -m json.tool
```

### Step A4 — Archive full run artifacts

The run folder (`results.jsonl`, `settings_snapshot.json`, `REPORT.md`) must
be archived on encrypted local or shared secure storage. These files contain
aggregate metrics only — no biometric images — but document the evaluation
conditions. Do not delete them.

### Step A5 — Commit baseline file to source control

```powershell
git add eval/baselines/heuristic_rgb_v1_standard.json
git commit -m "freeze: PAD baseline heuristic_rgb_v1_standard <run_id>"
```

Do **not** commit `eval/datasets/face_lock_v1/` (already in `.gitignore`).
Do **not** commit `eval/reports/runs/` run artifacts (see `.gitignore`).

---

## 11. Updating or replacing a baseline

| Scenario | Action |
|----------|--------|
| Intentional threshold change with passing metrics | New run → repeat §10; overwrite baseline file |
| Failed experiment (metrics regressed) | Do not update baseline; investigate |
| New passive model (e.g. ONNX) | New file: `onnx_pad_v1_standard.json`; keep old baseline |
| High-tier promotion | New file: `heuristic_rgb_v1_high.json` |

Never overwrite an approved baseline without a full new evaluation run and
dual sign-off.

---

## 12. `summary.json` → `eval/baselines/heuristic_rgb_v1_standard.json`

The baseline file is the approved run's `summary.json` verbatim. It must
validate against `eval/reports/template/summary_schema.json`.

Required fields in the frozen file:

```json
{
  "run_id": "face_lock_v1_heuristic_standard_001",
  "date_utc": "2026-...",
  "protocol_id": "PAD-ISO-LAB-v1",
  "collection_id": "face_lock_v1",
  "model_id": "heuristic_rgb_v1",
  "evaluation_policy": "fail_closed_v1",
  "assurance_tier": "standard",
  "n_attack": 150,
  "n_bona_fide": 150,
  "apcer_overall": 0.0333,
  "bpcer_overall": 0.0267,
  "tier_gate": {
    "apcer": "pass",
    "bpcer": "pass",
    "latency": "pass",
    "overall": "pass"
  }
}
```

Numeric values above are illustrative targets only. Use actual run values.

Future E4 runs compare against this file using `baseline_compare.py`:
- Regression flagged when `apcer_overall` or `bpcer_overall` increase by > 0.01.
- The comparison requires `model_id` and `assurance_tier` to match.

---

## 13. Missing prerequisites

The following must be resolved before a baseline run can be attempted:

### BLOCKER — Dataset counts (all categories)

The `face_lock_v1` dataset is at P0 pilot scale. P1 collection is required.

| Gap | Required action |
|-----|----------------|
| 146 BF presentations (4 → 150) | Continue BF capture; 28 more subjects |
| 50 PA-01 presentations | Full PA-01 capture; 10 subjects |
| 49 PA-02 presentations (1 → 50) | Continue PA-02; 9 more subjects |
| 50 PA-03 presentations | Full PA-03 capture; 10 subjects |

Estimated capture time (single collector): **1–2 weeks** if capture sessions
are run daily.

### BLOCKER — `tools/` scripts not verified in current environment

`collect_pad_dataset.py`, `generate_manifest.py`, and `qc_dataset.py` are
referenced in collection guides but were not verified as part of this audit.
Confirm they run successfully before P1 collection begins:

```powershell
python tools/generate_manifest.py --help
python tools/qc_dataset.py --help
python tools/collect_pad_dataset.py --help
```

If any script is missing or fails, implement or fix it before collection.

### BLOCKER — `tools/project_paths.py` must export `resolve_project_root`

`report_builder.py` imports `from tools.project_paths import resolve_project_root`.
Verify this module exists and the function is accessible:

```powershell
python -c "from tools.project_paths import resolve_project_root; print(resolve_project_root())"
```

### WARNING — No CI fixture regression for `heuristic_rgb_v1`

`eval/fixtures/` contains a placeholder manifest but image files are excluded
from git. Until a CI-runnable fixture is wired up, regressions in
`pad_engine.py` will not be caught automatically between baseline runs.

### WARNING — Active challenge latency unquantified

Clips submitted to `PAD-ISO-LAB-v1` use `PadSession.evaluate_final` in batch
mode. The P95 latency gate (1200 ms) covers this path. The live-sim path
(`PAD-LIVE-SIM-v1`) adds session overhead. Measure both before claiming a
latency-safe baseline.

### WARNING — Subject pool diversity not specified

The collection guide does not define diversity requirements (age range, skin
tone, glasses, facial hair). The heuristic Laplacian scorer is known to be
sensitive to skin tone via channel correlation. Document demographics in
`collection_meta.json` and note any observed BPCER skew by subset.

---

## 14. Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Heuristic scorer fails PA-01 gate (print texture) | High | Medium | Expected weakness of Laplacian heuristic; document APCER per PAI; plan ONNX model for high tier |
| BPCER exceeds 5% due to borderline → active challenge on BF clips | Medium | Medium | `INCONCLUSIVE` collapses to ATTACK under `fail_closed_v1`; if borderline rate is high, reduce `borderline_high` or increase clip length |
| Capture SOP deviation between P0 and P1 | Medium | Medium | Fix camera, distance, and background before P1; record in `collection_meta.json` |
| Corrupt JPEGs (ERROR outcomes) inflate BPCER | Low | Low | `qc_dataset.py` catches decode failures; re-capture affected presentations |
| Baseline frozen from dirty git tree | Low | Low | `settings_snapshot.json` records `git_dirty: true`; reject baseline if dirty |
| Attack subjects overlap with BF subjects | Medium | Low | Collector SOP; document in `collection_meta.json`; treat as known limitation |
| Latency P95 exceeds 1200 ms on slow hardware | Low | Low | Measure on target hardware before collection; `latency_ms_p95_max` is advisory for demo tier |

---

## 15. Recommended dataset size (production claim)

For a robust first baseline that supports a production deployment claim:

| Category | Recommended | Minimum (standard tier gate) |
|----------|-------------|------------------------------|
| Bona fide | **200–250 presentations / 40+ subjects** | 150 / 30 |
| PA-01 (print) | **75 presentations / 15 subjects** | 50 / 10 |
| PA-02 (static screen) | **75 presentations / 15 subjects** | 50 / 10 |
| PA-03 (video replay) | **75 presentations / 15 subjects** | 50 / 10 |
| PA-04 (virtual camera) | **50 presentations / 10 subjects** | 30 / 6 |

The larger BF set improves BPCER confidence interval width. With n=150 BF,
a BPCER of 5% has a 95% CI of approximately ±3.4 pp (Wilson interval).
With n=250, the CI narrows to ±2.7 pp — more defensible for production sign-off.

Confidence interval rule of thumb: collect at least **4× the minimum** if the
metric is expected to be close to the gate threshold.

---

## 16. Quick reference — command sequence

Complete command sequence for a first approved baseline (after P1 collection):

```powershell
# 1. Generate manifest
python tools/generate_manifest.py `
  --collection-root eval/datasets/face_lock_v1 `
  --strict

# 2. QC gate — must exit 0
python tools/qc_dataset.py `
  --collection-root eval/datasets/face_lock_v1
# exit code 0 = proceed; 1 = fix warnings first; 2 = stop

# 3. Run E3 evaluation harness
python -m eval.harness.run_pad_eval `
  --protocol eval/protocols/pad_iso_lab_v1.yaml `
  --manifest eval/datasets/face_lock_v1/manifest.json `
  --collection-root eval/datasets/face_lock_v1 `
  --output-dir eval/reports/runs `
  --run-id face_lock_v1_heuristic_standard_001

# 4. (Optional) Live-sim validation
python -m eval.harness.run_pad_eval `
  --protocol eval/protocols/pad_live_sim_v1.yaml `
  --manifest eval/datasets/face_lock_v1/manifest.json `
  --collection-root eval/datasets/face_lock_v1 `
  --output-dir eval/reports/runs `
  --run-id face_lock_v1_live_sim_001

# 5. Generate REPORT.md and tier gate result
python -m eval.reporting.report_builder `
  --run-dir eval/reports/runs/face_lock_v1_heuristic_standard_001
# exit code 0 = PASS; 1 = FAIL

# 6. Review REPORT.md — dual sign-off required

# 7. Freeze baseline
copy `
  eval\reports\runs\face_lock_v1_heuristic_standard_001\summary.json `
  eval\baselines\heuristic_rgb_v1_standard.json

# 8. Commit
git add eval/baselines/heuristic_rgb_v1_standard.json
git commit -m "freeze: PAD baseline heuristic_rgb_v1_standard face_lock_v1_heuristic_standard_001"
```

---

## 17. Related documents

| Document | Path |
|----------|------|
| Collection guide (SOP) | `eval/docs/FACE_LOCK_V1_COLLECTION_GUIDE.md` |
| Protocol definition | `eval/protocols/pad_iso_lab_v1.yaml` |
| Live-sim protocol | `eval/protocols/pad_live_sim_v1.yaml` |
| Registration protocol | `eval/protocols/pad_registration_v1.yaml` |
| Manifest schema | `eval/datasets/_template/manifest.schema.json` |
| Summary schema | `eval/reports/template/summary_schema.json` |
| Report template | `eval/reports/template/report_template.md` |
| Baselines directory | `eval/baselines/` |
| Security audit | `docs/security_audit_v1.md` |
| V1 release checklist | `docs/V1_RELEASE_CHECKLIST.md` |
