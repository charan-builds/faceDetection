# PAD Evaluation Report

> Auto-generated report template (Phase E2+). Copy to `eval/reports/runs/<run_id>/REPORT.md` and fill placeholders.

---

## Run metadata

| Field | Value |
|-------|-------|
| **Run ID** | `{{ run_id }}` |
| **Date (UTC)** | `{{ date_utc }}` |
| **Protocol** | `{{ protocol_id }}` |
| **Git SHA** | `{{ git_sha }}` |
| **Python** | `{{ python_version }}` |
| **OpenCV** | `{{ opencv_version }}` |

---

## System under test

| Field | Value |
|-------|-------|
| **Model ID** | `{{ model_id }}` |
| **Model version** | `{{ model_version }}` |
| **PAD mode** | `{{ pad_mode }}` |
| **Assurance tier** | `{{ assurance_tier }}` |
| **Evaluation policy** | `{{ evaluation_policy }}` |
| **Dataset** | `{{ collection_id }}` (manifest `{{ manifest_version }}`) |

---

## Executive summary

| Metric | Result | Target | Gate |
|--------|--------|--------|------|
| APCER (pooled PA-01–03) | `{{ apcer_pooled_pct }}` | `≤ {{ apcer_target_pct }}` | `{{ apcer_gate }}` |
| BPCER | `{{ bpcer_pct }}` | `≤ {{ bpcer_target_pct }}` | `{{ bpcer_gate }}` |
| Latency P95 (ms) | `{{ latency_p95_ms }}` | `≤ {{ latency_target_ms }}` | `{{ latency_gate }}` |

**Overall gate:** `{{ overall_gate }}`

---

## APCER by attack type

| PAI | Label | N | APCER |
|-----|-------|---|-------|
| PA-01 | Print | `{{ n_pa01 }}` | `{{ apcer_pa01_pct }}` |
| PA-02 | Static screen | `{{ n_pa02 }}` | `{{ apcer_pa02_pct }}` |
| PA-03 | Video replay | `{{ n_pa03 }}` | `{{ apcer_pa03_pct }}` |
| PA-04 | Virtual camera | `{{ n_pa04 }}` | `{{ apcer_pa04_pct }}` |
| **Pooled** | PA-01–03 | `{{ n_attack_pooled }}` | `{{ apcer_pooled_pct }}` |

---

## BPCER breakdown (bona fide)

| Final outcome | Count | % of BF |
|---------------|-------|---------|
| LIVE | `{{ bf_live_count }}` | `{{ bf_live_pct }}` |
| INCONCLUSIVE | `{{ bf_inconclusive_count }}` | `{{ bf_inconclusive_pct }}` |
| ATTACK | `{{ bf_attack_count }}` | `{{ bf_attack_pct }}` |
| ERROR | `{{ bf_error_count }}` | `{{ bf_error_pct }}` |

---

## Confusion matrix (`fail_closed_v1`)

|  | Predicted LIVE | Predicted not LIVE |
|--|----------------|---------------------|
| **Ground truth: attack** | `{{ cm_attack_as_live }}` (APCER errors) | `{{ cm_attack_correct }}` |
| **Ground truth: bona fide** | `{{ cm_bf_live }}` | `{{ cm_bf_rejected }}` (BPCER errors) |

---

## PAD settings snapshot

See `settings_snapshot.json` in this run folder.

Key thresholds:

| Setting | Value |
|---------|-------|
| `passive_pass_threshold` | `{{ passive_pass_threshold }}` |
| `passive_reject_threshold` | `{{ passive_reject_threshold }}` |
| `borderline_low` / `borderline_high` | `{{ borderline_low }}` / `{{ borderline_high }}` |
| `min_frames_passive` | `{{ min_frames_passive }}` |

---

## Recommendations

- `{{ recommendation_1 }}`
- `{{ recommendation_2 }}`

---

## Artifacts

| File | Description |
|------|-------------|
| `results.jsonl` | Per-presentation outcomes |
| `results.csv` | Flat export |
| `summary.json` | Aggregated metrics (validates against `summary_schema.json`) |
| `settings_snapshot.json` | Full `PADSettings` + environment |

---

## Sign-off

| Role | Name | Date |
|------|------|------|
| Evaluation engineer | | |
| Security reviewer | | |
