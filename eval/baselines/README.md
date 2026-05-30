# PAD evaluation baselines

Frozen **approved** run summaries for regression comparison.

---

## Purpose

After a successful lab evaluation (Phase E3), copy the run's `summary.json` here:

```text
eval/baselines/<model_id>_<tier>.json
```

Example:

```text
heuristic_rgb_v1_standard.json
```

Future runs compare against this file; regression fails if APCER/BPCER degrade beyond configured deltas (see FL-PAD-EVAL-001).

---

## Naming convention

```text
{passive_model_id}_{assurance_tier}.json
```

Optional suffix for protocol:

```text
heuristic_rgb_v1_standard_iso_lab.json
```

---

## Baseline file requirements

Must validate against:

`eval/reports/template/summary_schema.json`

Required fields include `run_id`, `apcer_overall`, `bpcer_overall`, `model_id`, `tier_gate`.

---

## When to update

| Action | Update baseline? |
|--------|------------------|
| Intentional model/threshold release | Yes, with review sign-off |
| Failed experiment | No |
| ONNX model promotion | New file, e.g. `onnx_pad_v1_standard.json` |

---

## Phase E1 status

No baseline JSON files yet. Add after first approved lab run.
