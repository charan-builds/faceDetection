# Evaluation fixtures

Tiny synthetic clips for **CI regression** (Phase E2+).

These are **not** a substitute for lab APCER measurement.

---

## Purpose

| Use | Lab dataset |
|-----|-------------|
| CI: outcome regression with `mock_*` or local synthetic JPEGs | Full APCER/BPCER |
| Fast PR checks | Tier sign-off |

---

## Layout

```text
fixtures/
├── manifest.json
├── bona_fide_synthetic/
│   └── session_01/          ← frame_0001.jpg ... (local only)
└── attack_flat_synthetic/
    └── session_01/
```

Image files are excluded from git by `.gitignore` (`*.jpg`). Operators or CI may generate them with a documented script in Phase E2.

---

## manifest.json

Lists fixture samples with `subject_id: subj_000` reserved for synthetic data.

---

## Relation to unit tests

`tests/unit/test_pad_engine.py` uses in-memory NumPy frames and does not depend on these files.

Future `tests/eval/` may load this manifest when harness is implemented.
