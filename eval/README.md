# Face Lock AI — PAD Evaluation

Presentation Attack Detection (PAD) evaluation infrastructure for Face Lock AI.

This folder supports **measurement and validation only**. It does not contain PAD models, ONNX artifacts, or recognition workflow code.

**Source of truth:** FL-PAD-EVAL-001 (PAD Evaluation Framework)

**System under test (future phases):** `src/pad_engine.py` with `PADSettings` from `src/config.py`

---

## Purpose

Before adopting ONNX or other passive scorers, the project needs:

| Need | How this folder helps |
|------|------------------------|
| Repeatable APCER / BPCER | Protocols + dataset manifests |
| Baseline comparison | `baselines/` frozen summaries |
| CI regression (later) | `fixtures/` synthetic clips |
| Audit trail | `reports/` templates and run outputs |

**Phase E1:** folders, protocols, schemas, templates.

**Phase E2 (metrics):** `eval/metrics/`, `eval/harness/outcome_mapper.py`.

**Phase E3 (harness):** `run_pad_eval.py`, `clip_loader.py`, `settings_snapshot.py` — **no markdown reports or baselines yet**.

---

## Layout

```text
eval/
├── README.md                 ← you are here
├── protocols/                ← YAML suite definitions
├── datasets/                 ← real lab data (local only, not in git)
│   └── _template/            ← committed structure + schema
├── fixtures/                 ← tiny clips for CI (manifest committed)
├── harness/                  ← outcome_mapper (E2); run_pad_eval (E3+)
├── metrics/                  ← APCER/BPCER, confusion, latency (E2)
├── reports/                  ← generated run outputs + templates
└── baselines/                ← approved baseline summary JSON files
```

---

## Evaluation suites (planned)

| Suite | Protocol file | Description |
|-------|---------------|-------------|
| PAD-ISO-LAB | `protocols/pad_iso_lab_v1.yaml` | Primary APCER/BPCER on clip presentations |
| PAD-LIVE-SIM | `protocols/pad_live_sim_v1.yaml` | Unlock path via `PadSession` |
| PAD-REG | `protocols/pad_registration_v1.yaml` | Registration still / burst path |

---

## Dataset policy

- **Do not commit** real face images, videos, or biometrics.
- Collect datasets under `eval/datasets/<collection_id>/` on local or secure storage.
- Use `datasets/_template/` and `manifest.schema.json` as the layout reference.
- Only pseudonymous `subject_id` values in manifests (e.g. `subj_001`).

### Synthetic dataset generator

`datasets/generate_synthetic_dataset.py` produces a seeded, procedurally
generated collection (150 BF / 50 PA-01 / 50 PA-02 / 50 PA-03 clips) for
exercising the harness and heuristic scorer:

```bash
python eval/datasets/generate_synthetic_dataset.py \
    --output-dir eval/datasets/synthetic_v1 --seed 20260826
```

Generated collections carry `"data_provenance": "SYNTHETIC"` in their
manifest. Synthetic results validate the harness pipeline only — they are
**not** real-world PAD evidence and cannot satisfy tier sign-off, which
requires real bona fide and physical attack presentations per the protocol
sample-size requirements.

---

## Metrics (definitions)

Aligned with ISO/IEC 30107 and Face Lock **fail-closed** policy (`fail_closed_v1`):

| Metric | Definition |
|--------|------------|
| **APCER** | Attack presentations classified as bona fide (`PadOutcome.LIVE`) / total attacks |
| **BPCER** | Bona fide presentations classified as not live (`ATTACK`, `INCONCLUSIVE`, `ERROR`) / total bona fide |

Tier targets (Standard): APCER ≤ 5% (PA-01–03 pooled), BPCER ≤ 5%.

---

## Workflow (future phases)

1. Collect data using `_template` layout and manifest schema.
2. Point protocol YAML at dataset path and `PADSettings` tier.
3. Run harness (Phase E3):

   ```powershell
   python -m eval.harness.run_pad_eval `
     --protocol eval/protocols/pad_iso_lab_v1.yaml `
     --manifest eval/fixtures/manifest.json `
     --collection-root eval/fixtures `
     --output-dir eval/reports/runs
   ```
4. Review report from `reports/template/report_template.md`.
5. On approval, copy `summary.json` to `baselines/`.

---

## What is explicitly out of scope here

- ONNX models or new PAD scorers
- Changes to `recognize.py` or `pad_engine.py`
- Production logging aggregation for APCER

---

## Related project modules

| Module | Role |
|--------|------|
| `src/pad_engine.py` | PAD decisions (`PadResult`, `PadSession`) |
| `src/config.py` | `PADSettings` / `PAD_SETTINGS` |
| `tests/unit/test_pad_engine.py` | Unit regression (not lab APCER) |

---

## Phase roadmap

| Phase | Deliverable |
|-------|-------------|
| **E1** | Scaffold (protocols, datasets template, reports) |
| **E2** | Metrics: outcome mapper, APCER/BPCER, confusion, latency |
| **E3** | Harness: protocol/manifest load, clip loader, `run_pad_eval`, snapshots |
| E4 | Markdown reports, baseline comparison, first lab sign-off |
| E4 | CI fixture regression + ONNX comparison protocol |
