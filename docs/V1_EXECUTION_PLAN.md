# Face Lock AI — V1 Execution Plan

**Date:** 2026-06-01  
**Status inputs:** Unit ✓ | Integration ✓ | Eval ✓ | DeepFace/TF ✓ | PAD E1–E4 ✓  
**Goal:** Shortest safe path from current state to v1 release.

---

## Scoring

| Dimension | Now | Target |
|-----------|-----|--------|
| Security | 42 / 100 | 70 / 100 |
| Test coverage | 62 / 100 | 80 / 100 |
| Reliability | 64 / 100 | 82 / 100 |
| Performance | 50 / 100 | 65 / 100 |
| Usability | 70 / 100 | 80 / 100 |

Reaching **P0** items alone moves production readiness from **52 → ~75**.

---

## A — Pickle Security Hardening Plan

### Current pickle call sites

| File | Function | Line | Operation | Concern |
|------|----------|------|-----------|---------|
| `src/utils.py` | `save_pickle` | ~143 | `pickle.dump(data, file)` | Unrestricted serialization |
| `src/utils.py` | `load_pickle` | ~173 | `pickle.load(file)` | **RCE** if `encodings/` is writable |
| `src/ai_engine.py` | `save_user_embeddings` | ~456 | calls `save_pickle` | Writes embedding record dicts |
| `src/ai_engine.py` | `load_user_embeddings` | ~469 | calls `load_pickle` | Reads per-user record |
| `src/ai_engine.py` | `load_all_user_embeddings` | ~495 | calls `load_pickle` in glob loop | Loads every `*.pkl` in `encodings/` |

**Root risk:** any process that drops a crafted `.pkl` into `encodings/` before the next recognition run gains RCE via `pickle.load`.

### Migration design: JSON + NumPy `.npy`

Replace the opaque pickle blob with two files per user:

```
encodings/
  alice.json       # metadata + schema version
  alice.npy        # float32 embedding matrix, shape (n_images, 512)
```

**`alice.json` schema (v2):**

```json
{
  "schema_version": 2,
  "user_name": "alice",
  "safe_name": "alice",
  "model_name": "Facenet512",
  "detector_backend": "opencv",
  "distance_metric": "cosine",
  "threshold": 0.3,
  "embedding_count": 4,
  "image_paths": ["data/alice/image_001.jpg", "..."],
  "created_at": "2026-06-01T08:00:00",
  "npy_file": "alice.npy"
}
```

**Backward compatibility layer** in `load_user_embeddings`:

```python
def load_user_embeddings(user_name, input_path=None):
    base = input_path or get_user_embedding_path(user_name)
    if Path(str(base)).suffix == ".pkl":
        # Legacy path: load with restricted unpickler
        return _load_pickle_legacy(base)
    return _load_json_npy(base)  # new path: base == .json file
```

**Restricted unpickler** for legacy `.pkl` (no upgrade path):

```python
import pickle, io

class _SafeUnpickler(pickle.Unpickler):
    ALLOWED_GLOBALS = {
        "builtins": {"dict", "list", "float", "int", "str"},
    }
    def find_class(self, module, name):
        if module in self.ALLOWED_GLOBALS and name in self.ALLOWED_GLOBALS[module]:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked: {module}.{name}")
```

**Migration order:**

1. Add `_SafeUnpickler` to `utils.py` — replaces bare `pickle.load` in `load_pickle`.
2. Implement `save_user_embeddings_v2` (JSON + `.npy`) alongside existing.
3. Implement `load_user_embeddings_v2` with schema validation.
4. Add `migrate_encodings_to_v2` script (`scripts/migrate_encodings.py`).
5. On first load of a `.pkl`, auto-migrate and write v2 files.
6. Remove v1 pickle files after confirmed migration.

**Effort:** ~1 day  
**Risk if skipped:** CRITICAL — first ranked blocker.

---

## B — Operator Authentication Plan

### Privileged actions requiring a gate

| Action | Location | Current control |
|--------|----------|-----------------|
| Register new user | `app.py` menu choice 1 | **None** |
| Overwrite existing user | `ai_engine.save_user_embeddings` | **None** (silent overwrite) |
| Start live recognition | `app.py` menu choice 2 | **None** |
| Delete user (future) | `encodings/*.json` + `data/` | **None** |

### Minimal PIN gate design

One new module: `src/operator_auth.py`

```python
import getpass, hashlib, os

ENV_VAR = "FACE_LOCK_OPERATOR_PIN_HASH"   # sha256 hex of PIN, set at deploy time

def require_operator_pin() -> bool:
    """Return True if PIN is verified or auth is disabled."""
    pin_hash = os.environ.get(ENV_VAR)
    if not pin_hash:
        return True   # no PIN configured → local-demo mode
    attempt = getpass.getpass("Operator PIN: ")
    return hashlib.sha256(attempt.encode()).hexdigest() == pin_hash
```

**Integration points (two lines each):**

1. `app.py` → `handle_registration()`: call `require_operator_pin()` before `run_safely("Registration", ...)`.
2. `app.py` → `main()`: document operator-mode assumption in startup banner.
3. `src/ai_engine.py` → `save_user_embeddings()`: optionally accept a `verified: bool` param; raise `PermissionError` if `False`.

**Deployment:**

```powershell
# One-time setup (PowerShell)
$hash = [System.BitConverter]::ToString(
    [System.Security.Cryptography.SHA256]::Create().ComputeHash(
        [System.Text.Encoding]::UTF8.GetBytes("your-pin")
    )
).Replace("-","").ToLower()
[System.Environment]::SetEnvironmentVariable("FACE_LOCK_OPERATOR_PIN_HASH", $hash, "User")
```

**Effort:** ~1 hour for shell; ~2 hours with integration tests  
**Risk if skipped:** HIGH — anyone at the console can enroll themselves.

---

## C — Coverage Gap Report

### Priority by security impact

| Priority | Function | File | Lines | Reason |
|----------|----------|------|-------|--------|
| **P0** | `register_user` | `src/register.py` | ~256 | Entire enrollment path untested — highest-risk write operation |
| **P0** | `load_all_user_embeddings` — corrupt file | `src/ai_engine.py` | ~493 | No test for single corrupt file; currently aborts all users |
| **P0** | `save_user_embeddings` — output_path bypass | `src/ai_engine.py` | ~454 | No test that arbitrary `output_path` is blocked or contained |
| **P1** | `capture_registration_images` | `src/register.py` | ~119 | No test for PAD enforcement on registration, webcam error handling |
| **P1** | `get_user_data_folder` — overwrite guard | `src/register.py` | ~99 | No test that re-registering the same user prompts or warns |
| **P1** | `classify_recognition_error` | `src/recognize.py` | ~185 | Fragile substring matching; no tests for new error message variants |
| **P1** | `create_embedding_record` | `src/ai_engine.py` | ~395 | Schema shape not asserted in tests; threshold field stored but unused |
| **P2** | `load_trusted_records` — fail-open | `src/recognize.py` | ~104 | Returns `[]` on error; should be tested to confirm denial behavior |
| **P2** | `ensure_project_folders` | `src/utils.py` | ~71 | Untested |
| **P2** | `ask_user_name` | `src/register.py` | ~69 | Input loop; `"unknown"` collision untested via direct call |
| **P2** | `app.py` menu/dispatch | `app.py` | all | Zero coverage; smoke test with mocked workflows |

**Quickest coverage wins (< 1 hour each):**

- `test_create_embedding_record.py` — pure unit, no mocks, validates schema keys
- `test_classify_recognition_error.py` — pure unit, no mocks
- `test_load_trusted_records_on_corrupt_file.py` — tmp_path + bad pickle bytes

---

## D — Release Checklist

### P0 — Must fix before any non-demo deployment

- [ ] **SEC-002** Replace or restrict-unpickle embedding storage (`save_user_embeddings` → JSON+npy)
  - File: `src/utils.py`, `src/ai_engine.py`
  - Effort: 1 day
  - Blocks: any deployment where `encodings/` is accessible by non-root processes

- [ ] **SEC-003** Restrict `encodings/` OS permissions to owner-only (0600 files, 0700 dir)
  - File: `src/ai_engine.py` `save_user_embeddings`, deploy docs
  - Effort: 1 hour
  - Note: Windows ACL equivalent via `icacls`

- [ ] **SEC-001** Operator PIN gate on registration
  - File: new `src/operator_auth.py` + 2-line hook in `app.py`
  - Effort: 2 hours
  - Blocks: any shared-access machine deployment

- [ ] **TEST** `tests/unit/test_register.py` — registration workflow with mocked cv2 + DeepFace
  - Effort: 3–4 hours
  - Blocks: CI gate

- [ ] **PERF-001** Warm up DeepFace model at app startup (`load_face_model()` in `app.py`)
  - File: `app.py`
  - Effort: 1 hour (single line + test)

- [ ] **PAD** Run full `run_pad_eval` harness on `face_lock_v1` dataset → confirm APCER/BPCER within gates
  - Effort: 1 hour (data collection already done)
  - Blocks: production PAD claim

- [ ] **PAD** Freeze baseline JSON: copy approved `summary.json` → `eval/baselines/heuristic_rgb_v1_standard.json`
  - Effort: 10 minutes

- [ ] **ENV** Verify `pip install -e ".[dev]" && python scripts/patch_deepface_tf215.py && pytest` clean on Python 3.11 fresh venv
  - Effort: 30 minutes

---

### P1 — Should fix (high value, will not block demo release)

- [ ] **SEC-008** Apply low-light check in PAD session recognition path
  - File: `src/recognize.py` ~L421
  - Effort: 30 minutes

- [ ] **SEC-004** Re-registration guard — warn/confirm before overwriting existing user
  - File: `src/ai_engine.py` `save_user_embeddings`, `src/register.py` `register_user`
  - Effort: 2 hours

- [ ] **SEC-010** Username `"unknown"` collision guard in `register_user` (already guarded in `ask_user_name`, but not in direct API call)
  - File: `src/ai_engine.py` `save_user_embeddings`
  - Effort: 30 minutes

- [ ] **RELIABILITY** Corrupt `.pkl` skip-and-continue: replace abort-all with per-file warn+skip in `load_all_user_embeddings`
  - File: `src/ai_engine.py` ~L496
  - Effort: 30 minutes

- [ ] **RELIABILITY** `save_user_embeddings` / `load_user_embeddings` — confine `output_path` / `input_path` to `ENCODINGS_DIR`
  - File: `src/ai_engine.py`
  - Effort: 30 minutes

- [ ] **TEST** `tests/unit/test_ai_engine_register.py` — `create_embedding_record`, `save_user_embeddings`, `load_user_embeddings`
  - Effort: 2 hours

- [ ] **TEST** `tests/unit/test_classify_recognition_error.py`
  - Effort: 1 hour

- [ ] **CODE** `classify_recognition_error` — replace fragile `str(error).lower()` substring match with exception type check or explicit error codes
  - File: `src/recognize.py`
  - Effort: 2 hours

- [ ] **CODE** Remove `src/capture.py` missing-module ambiguity — the file exists at runtime, verify it is shipped; add explicit import test
  - Effort: 30 minutes

- [ ] **DOC** Add security model to README: "This is a local demo — threat model is trusted operator, untrusted physical visitors"
  - Effort: 30 minutes

---

### P2 — Nice to have

- [ ] **PERF-003** Batch embedding matrix comparison (NumPy vectorised cosine) for >5 users
  - File: `src/ai_engine.py` `find_best_match`
  - Effort: 2 hours

- [ ] **PERF-002** Cheap face presence pre-check before DeepFace (OpenCV Haar)
  - File: `src/recognize.py` `recognize_identity`
  - Effort: 3 hours

- [ ] **TEST** `tests/unit/test_load_trusted_records.py` — fail-open behavior on corrupt encodings
  - Effort: 1 hour

- [ ] **TEST** `tests/system/` — end-to-end smoke tests with `@pytest.mark.system`
  - Effort: 1 day

- [ ] **LINT** Add `ruff` + `mypy` to dev deps; wire to CI
  - Effort: 2 hours

- [ ] **CI** GitHub Actions matrix: Python 3.10 + 3.11, `pytest -m "not system"`
  - Effort: 2 hours

- [ ] **DOC** Consolidate `CONFIGURATION.md` + `LOGGING.md` → README appendix
  - Effort: 1 hour

- [ ] **PACKAGING** Gitignore `face_lock_ai.egg-info/`; add `encodings/`, `data/` to `.gitignore`
  - Effort: 10 minutes

---

## Recommended implementation order

The table below shows the shortest path to a **stable, shippable v1 demo** (target ~8 hours total engineering time).

```
Hour 0–1    ENV      Fresh venv Python 3.11 + patch_deepface + pytest clean run
Hour 1–2    PERF-001 load_face_model() at startup  (single line)
Hour 2–3    SEC-001  Operator PIN gate              (new file + 2-line hook)
Hour 3–5    SEC-002  Pickle → restricted unpickler  (utils.py + migration note)
Hour 5–8    TEST     tests/unit/test_register.py    (mock cv2 + DeepFace)
Hour 8+     PAD      run_pad_eval on face_lock_v1 → freeze baseline JSON
```

After those 8 hours, address **P1** items in any order during stabilisation. **P2** items are post-v1.

---

## Risk ranking (all open items)

| Rank | ID | Category | Severity | Effort |
|------|-----|----------|----------|--------|
| 1 | SEC-002 | Pickle RCE | Critical | 1 day |
| 2 | SEC-001 | No operator auth | Critical | 2 hrs |
| 3 | SEC-003 | Encodings file permissions | High | 1 hr |
| 4 | TEST register.py | Zero coverage on enrollment | High | 4 hrs |
| 5 | PAD baseline frozen | No released eval gate | High | 1 hr |
| 6 | SEC-004 | Silent overwrite on re-register | Medium | 2 hrs |
| 7 | RELIABILITY corrupt pkl | Abort-all on bad file | Medium | 30 min |
| 8 | SEC-010 | "unknown" collision | Medium | 30 min |
| 9 | PERF-001 | Cold-start latency | Medium | 1 hr |
| 10 | SEC-008 | Low-light skip in PAD path | Medium | 30 min |
| 11 | code classify_error | Fragile string matching | Low | 2 hrs |
| 12 | PERF-003 | Batch cosine comparison | Low | 2 hrs |

---

## What is intentionally NOT planned

- Upgrade TensorFlow to 2.16+ / Keras 3 → **v1.1 epic**
- Web / HTTP API → **future scope**
- Multi-machine / remote deployment → outside threat model for v1 demo
- Change PAD thresholds or eval metrics → require explicit approval + re-baseline
