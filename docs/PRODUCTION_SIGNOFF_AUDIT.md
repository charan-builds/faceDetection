# Face Lock AI — Production Sign-Off Audit

**Audit date:** 2026-06-03  
**Auditor:** Automated repository review  
**Scope:** `src/`, `tests/`, `eval/`, `tools/`, `docs/`, `app.py`, config, `.gitignore`  
**Test suite state at audit time:** 218/218 unit + integration tests passing  
**Python version:** 3.11.9

---

## Executive Summary

Face Lock AI has made substantial security progress since the v1 security audit
(2026-05-30). Three of the four P0 security findings are now resolved in code.
The test suite is healthy and growing. The application architecture is clean and
well-separated. The remaining blocker before any production deployment is the
PAD evaluation dataset — the project cannot make a defensible liveness claim
without a frozen baseline backed by real collected data.

**Current production readiness: NO-GO for deployment. GO for continued development.**

The system is a safe, well-tested local demo. It is not ready for deployment on
a shared or untrusted machine without completing the PAD baseline.

---

## A. Security Status

### Summary table

| Finding | Severity (original) | Status | Notes |
|---------|---------------------|--------|-------|
| SEC-001 Unauthenticated operator access | Critical | ✅ **FIXED** | `src/operator_auth.py`; SHA-256 PIN gate on registration |
| SEC-002 Pickle RCE | Critical | ✅ **FIXED** | `_SafeUnpickler` in `src/utils.py`; allowlist enforced |
| SEC-003 Embedding forgery / swap | High | ✅ **FIXED** | HMAC-SHA256 in `src/integrity.py`; verified on every load |
| SEC-004 Open registration (no approval) | High | ⚠️ **PARTIAL** | PIN gate blocks unauth access; no max-user cap or re-registration confirmation |
| SEC-005 Virtual camera allowed | High | ❌ **OPEN** | `allow_virtual_camera=True` in `src/config.py`; no device fingerprinting |
| SEC-006 Replay / spoof attack (heuristic PAD) | High | ⚠️ **PARTIAL** | Heuristic model present; no lab-validated APCER/BPCER baseline yet |
| SEC-007 Threshold bypass via config change | Medium | ❌ **OPEN** | `cosine_threshold` in config.py is unprotected; stored per-user value unused |
| SEC-008 Low-light check skipped in PAD path | Medium | ❌ **OPEN** | `is_bad_lighting` only runs outside `pad_session`; PAD path unguarded |
| SEC-009 Registration PAD weaker than live | Medium | ❌ **OPEN** | Burst/single-frame PAD only; no full PadSession on registration |
| SEC-010 Username normalisation collision | Medium | ❌ **OPEN** | `John`/`john`/`john!` map to same safe_name; silent overwrite |
| SEC-011 Path traversal via username | Low (mitigated) | ✅ **MITIGATED** | `clean_name` strips traversal characters |
| SEC-012 Information disclosure on deny | Low | ❌ **OPEN** | Closest user name + distance shown in UI and logs on deny |
| SEC-013 Dataset poisoning (eval pipeline) | Medium | ⚠️ **PARTIAL** | `qc_dataset.py` gate exists; no manifest checksums yet |
| SEC-014 YAML unsafe load | Low (mitigated) | ✅ **MITIGATED** | `yaml.safe_load` confirmed throughout |
| SEC-016 Git subprocess in snapshot | Low | ❌ **OPEN** (acceptable) | Dev tooling only; acceptable for v1 |

### Critical finding verification

**SEC-001 (operator auth) — VERIFIED CLOSED:**
- `src/operator_auth.py`: `sha256_hash_pin()`, `verify_operator_pin()`, `is_demo_mode()`
- `app.py`: `handle_registration()` calls `verify_operator_pin()` before `register_user()`
- `FACE_LOCK_OPERATOR_PIN_HASH` env var; demo mode logs WARNING; `hmac.compare_digest` prevents timing attacks
- 31 unit tests in `tests/unit/test_operator_auth.py` — all passing

**SEC-002 (pickle RCE) — VERIFIED CLOSED:**
- `_SafeUnpickler` in `src/utils.py` overrides `find_class`
- Allowlist: `builtins.{dict,list,tuple,str,int,float,bool}`, `datetime.datetime`
- `os.system`, `subprocess.Popen`, `builtins.eval`, `importlib.import_module` — all blocked
- 21 unit tests in `tests/unit/test_pickle_security.py` including live gadget tests — all passing

**SEC-003 (embedding forgery) — VERIFIED CLOSED:**
- `src/integrity.py`: `sign_record()`, `verify_record()`, `wrap_record_for_storage()`
- HMAC-SHA256 over canonical key-sorted JSON; `hmac.compare_digest` constant-time
- `FACE_LOCK_SIGNING_KEY` env var; unsigned mode logs WARNING
- `_load_and_verify_record()` in `ai_engine.py` raises `IntegrityError` on tamper
- Legacy flat-format files load with WARNING (backward compatibility)
- 34 unit tests in `tests/unit/test_integrity.py` including tamper detection at 1e-9 float delta — all passing

### Residual security concerns

**Unsigned / unkeyed operation is the default** — if neither
`FACE_LOCK_OPERATOR_PIN_HASH` nor `FACE_LOCK_SIGNING_KEY` are set, both controls
fail open (with auditable WARNING logs). For any deployment beyond single-operator
development, both env vars must be configured before first run.

**`allow_virtual_camera: True`** is the default in `PADSettings`. A virtual
camera (OBS, ManyCam) can bypass the heuristic PAD scorer with pre-recorded video.
Flip this to `False` in the config for any security-sensitive deployment.

**`load_all_user_embeddings` abort-all** — if any `.pkl` file in `encodings/` fails
to load (corrupt, unreadable, or restricted-unpickle blocked), the entire recognition
session is blocked. This is fail-closed but may cause denial-of-service if an
attacker drops a corrupt file. The fix (skip-and-continue with WARNING) is noted in
`docs/V1_EXECUTION_PLAN.md` as P1.

---

## B. Testing Status

### Test suite overview

| Suite | Tests | Status | Coverage area |
|-------|-------|--------|---------------|
| `tests/unit/` | 218 | ✅ ALL PASSING | src, app |
| `tests/integration/` | ~15 | ✅ ALL PASSING | recognize.py end-to-end |
| `eval/tests/` | ~60 | ✅ ALL PASSING | eval harness, metrics, reporting |
| `tests/system/` | 0 | ❌ EMPTY | Manual/camera tests |

**Total passing: 218 (unit + integration). Eval suite: passing separately.**

### Unit test coverage by module

| Module | Tests | Estimated coverage | Assessment |
|--------|-------|-------------------|------------|
| `src/utils.py` | `test_utils.py` + `test_pickle_security.py` | ~90% | Strong |
| `src/ai_engine.py` | `test_ai_engine.py` + `test_integrity.py` | ~80% | Good |
| `src/integrity.py` | `test_integrity.py` | ~95% | Strong |
| `src/operator_auth.py` | `test_operator_auth.py` | ~95% | Strong |
| `src/pad_engine.py` | `test_pad_engine.py` | ~75% | Good |
| `src/register.py` | `test_register.py` | ~89% | Good |
| `src/config.py` | `test_config.py` | ~70% | Adequate |
| `src/display.py` | `test_display.py` | ~70% | Adequate |
| `src/logging_config.py` | `test_logging_config.py` | ~70% | Adequate |
| `src/recognize.py` | `tests/integration/test_recognize.py` | ~70% | Adequate |
| `src/capture.py` | Mocked only | ~10% | **Gap** |
| `app.py` | Partial (warmup + registration gate) | ~40% | **Gap** |
| `tests/system/` | — | 0% | **Gap** |

### Testing gaps (priority order)

**P1 — should have before baseline freeze:**
- `tests/unit/test_capture.py` — `open_webcam`, `read_frame`, `save_frame` failure paths; currently tested only via mocks in `test_register.py`
- `tests/system/` — at least one smoke test with `@pytest.mark.system` covering a full `app.py` menu flow with stdin mocking

**P2 — post-baseline:**
- `app.py` full menu flow (handle_recognition, handle_menu_choice, run_app exit paths)
- `load_all_user_embeddings` corrupt/skip-and-continue behaviour
- `classify_recognition_error` substring matching (fragile; needs direct unit tests)
- Harness integration test on `eval/fixtures/` with real `src.pad_engine`

### Test quality strengths

- Security controls (pickle, integrity, operator auth) are tested with adversarial payloads, not just happy paths
- PAD integration is tested through both unit (mock scorer) and integration (real `recognize_frame` with PAD enabled/disabled)
- Registration workflow has 20 tests covering all documented failure modes
- No test file uses `time.sleep`, real webcam, or real DeepFace downloads
- `conftest.py` `reset_face_lock_logging` autouse fixture prevents handler pollution across tests

---

## C. Performance Status

### Summary

| Metric | Status | Source |
|--------|--------|--------|
| DeepFace cold start | ✅ **FIXED** — warm-up at startup | `app.warmup_model()` called in `run_app()` |
| First recognition latency | Improved (~2–5 s saved on first frame) | PERF-001 resolved |
| Steady-state recognition | Acceptable for single-user demo | frame_skip=10, cooldown=1.5s |
| Embedding comparison | O(users × embeddings) linear scan | PERF-003 open; acceptable < 20 users |
| PAD latency P95 | Unquantified on target hardware | Required for baseline freeze |
| TF thread contention | Undocumented | PERF-010 open |

### Performance findings

**PERF-001 (DeepFace cold start) — RESOLVED:**
`app.warmup_model()` calls `ai_engine.load_face_model()` at startup. If the load
fails for any reason, a WARNING is printed and startup continues — the model will
load on first recognition attempt instead. This is non-fatal and tested.

**PERF-002 (full inference without face pre-check) — OPEN:**
Every eligible frame runs full DeepFace inference (`detect + align + embed`) even
when no face is in the frame. A cheap Haar cascade pre-check would reduce wasted
inference calls significantly. Deferred to P2.

**PERF-003 (linear embedding scan) — OPEN:**
`find_best_match` iterates all users × all embeddings per unlock attempt. At 10
users × 10 embeddings each = 100 cosine comparisons per attempt — acceptable. At
50+ users this should be batch-vectorised with NumPy. Deferred to P2.

**Latency gate for baseline:**
The PAD-ISO-LAB-v1 protocol requires P95 latency ≤ 1200 ms. This has not been
measured on the target hardware. It must be confirmed as part of the E3 harness
run before the baseline can be frozen.

---

## D. PAD Evaluation Status

### Dataset status (as of last QC run: 2026-05-30)

| Category | Have | Required (standard tier) | Gap |
|----------|------|--------------------------|-----|
| Bona fide (BF) | 4 presentations / 2 subjects | 150 / 30 | **146 / 28** |
| PA-01 (print) | 0 | 50 / 10 | **50 / 10** |
| PA-02 (static screen) | 1 / 1 | 50 / 10 | **49 / 9** |
| PA-03 (video replay) | 0 | 50 / 10 | **50 / 10** |
| PA-04 (virtual camera) | 0 | 30 / 6 (optional) | deferred |

**Current state: P0 pilot only. 5 total presentations. Not evaluable.**

### Eval framework readiness

| Component | Status | Location |
|-----------|--------|---------|
| E1: Protocol definitions | ✅ Complete | `eval/protocols/*.yaml` |
| E2: Metrics (APCER/BPCER, confusion, latency) | ✅ Complete + tested | `eval/metrics/` |
| E3: Evaluation harness | ✅ Complete + tested | `eval/harness/run_pad_eval.py` |
| E4: Reporting + tier gate | ✅ Complete + tested | `eval/reporting/report_builder.py` |
| Dataset collection tool | ✅ Complete | `tools/collect_pad_dataset.py` |
| Manifest generator | ✅ Complete | `tools/generate_manifest.py` |
| QC pipeline | ✅ Complete | `tools/qc_dataset.py` |
| PAD baseline runbook | ✅ Complete | `docs/PAD_BASELINE_RUNBOOK.md` |
| `tools/project_paths.py` `resolve_project_root` | ✅ VERIFIED | Required by `report_builder.py` |
| Approved baseline JSON | ❌ **MISSING** | `eval/baselines/heuristic_rgb_v1_standard.json` |
| Real dataset at P1 counts | ❌ **MISSING** | `eval/datasets/face_lock_v1/` |

### PASS criteria (standard tier)

All four gates must pass simultaneously:

| Gate | Metric | Threshold |
|------|--------|-----------|
| APCER | Pooled (PA-01 + PA-02 + PA-03) | ≤ 5.00% |
| BPCER | All bona fide | ≤ 5.00% |
| Latency P95 | ms | ≤ 1200 ms |
| Regression | APCER/BPCER vs baseline | ≤ +1.0 pp (first run: N/A) |

### PAD quality of existing data

The 5 existing presentations in `face_lock_v1` are **pilot-quality only**:
- mean brightness: 105.87 (acceptable)
- mean blur variance: 85.65 (acceptable)
- frame resolution: 1920×1080 (good)
- consistent dimensions: all True

The image quality is technically sound. The count is the only blocker for evaluation.

---

## E. Remaining Blockers

### BLOCKER 1 — PAD dataset at pilot scale (P0 collection required)

The `face_lock_v1` dataset has 5 presentations. The minimum for a valid standard-tier
evaluation is 300 (150 BF + 50 each PA-01/PA-02/PA-03). This is the single largest
gap between current state and a freeze-able baseline.

**Required action:** Complete P1 data collection per `docs/PAD_BASELINE_RUNBOOK.md`
§4. Estimated time: 1–2 weeks with a dedicated collector.

**Unblocks:** E3 harness run, E4 report, baseline freeze, production PAD claim.

---

### BLOCKER 2 — No approved baseline JSON

`eval/baselines/heuristic_rgb_v1_standard.json` does not exist. Without it:
- `report_builder.py` cannot perform regression comparison
- The E4 tier gate cannot produce an `overall: pass` result
- No signed-off `REPORT.md` exists
- The V1 release checklist remains unsigned

**Required action:** After Blocker 1, run E3 + E4 per runbook §7–§8, obtain dual
sign-off, then copy `summary.json` → `eval/baselines/heuristic_rgb_v1_standard.json`.

---

### BLOCKER 3 — Both security env vars default to unsafe mode

Neither `FACE_LOCK_OPERATOR_PIN_HASH` nor `FACE_LOCK_SIGNING_KEY` has a default
value. If these are not set before running the application on a shared machine,
operator authentication is bypassed and embedding files are unsigned.

**Required action:** Document env var setup in `CONFIGURATION.md` and add a startup
check that emits a visible WARNING (both already do) — and optionally gate the
application to require both vars for any non-demo deployment. This is an
**operational** blocker, not a code blocker — the controls are implemented, they
just need to be configured.

---

### WARNING 1 — `allow_virtual_camera: True` in default config

The default `PADSettings.allow_virtual_camera = True` means virtual camera attacks
are not actively blocked. The heuristic PAD may detect some but PA-04 has no
baseline data. For any deployment where the camera is accessible, this should be
`False`.

**Required action:** Either flip the default in `src/config.py` or document the
requirement to set it explicitly in `CONFIGURATION.md`.

---

### WARNING 2 — `load_all_user_embeddings` abort-all on corrupt file

If any `.pkl` file in `encodings/` raises an exception (corrupt bytes,
`RestrictedUnpicklingError`, or `IntegrityError`), `load_all_user_embeddings`
raises `RuntimeError` and recognition is denied for all users. This is
fail-closed (correct for security) but creates a denial-of-service vector if an
attacker drops a corrupt file into `encodings/`.

**Required action (P1):** Replace the abort-all with skip-and-continue + WARNING
per failed file.

---

### WARNING 3 — No system tests

`tests/system/` is empty. There are no end-to-end smoke tests that exercise a
full menu loop, stdin mocking, or real OpenCV window behavior. This means a
packaging or import regression could ship undetected.

**Required action (P1):** Add at least one `@pytest.mark.system` test that runs
`app.main()` with mocked inputs and confirms the exit path.

---

### WARNING 4 — `src/utility.py` is an empty file

The file exists and is tracked in git but contains no code. It creates a confusing
module that importers might accidentally reference.

**Required action (P3):** Delete or populate. No functional impact.

---

## F. Go / No-Go Recommendation

### Recommendation: **NO-GO for production deployment**

### GO for: continued development, demo use on single-operator trusted machines

| Dimension | Verdict | Reason |
|-----------|---------|--------|
| Security (critical) | ✅ GO | SEC-001, SEC-002, SEC-003 all resolved and tested |
| Security (remaining) | ⚠️ CONDITIONAL | High/medium findings open but acceptable for demo scope |
| Testing | ✅ GO | 218 tests passing; coverage adequate for demo |
| PAD evaluation | ❌ **NO-GO** | No approved baseline; dataset at 1.7% of required count |
| Performance | ✅ GO | Cold-start resolved; acceptable for single-user demo |
| Tooling / eval framework | ✅ GO | Full E1–E4 pipeline complete and tested |

**The sole hard blocker is the PAD dataset and baseline freeze.**
Everything else needed for a defensible production demo is either complete or
has a documented path to completion. Once the dataset is collected and the
baseline is frozen with dual sign-off, the remaining items are configuration
and polish, not architecture.

---

## Release Score

Scores are on a 0–100 scale. The scores below reflect the current audited state,
not the V1 execution plan targets.

| Dimension | Score | Change since V1 audit | Notes |
|-----------|-------|-----------------------|-------|
| **Security** | **72 / 100** | +30 (was 42) | SEC-001/002/003 resolved; high/medium findings remain |
| **Reliability** | **68 / 100** | +4 (was 64) | Abort-all on corrupt pkl is the main open item |
| **Maintainability** | **75 / 100** | — | Clean architecture; some DRY gaps; empty utility.py |
| **Performance** | **58 / 100** | +8 (was 50) | Cold-start fixed; PERF-002/003/010 open |
| **Test coverage** | **72 / 100** | +10 (was 62) | New test files for register, integrity, operator auth, warmup |
| **Evaluation readiness** | **35 / 100** | — | Framework 100% ready; dataset at P0 only |
| **Overall** | **63 / 100** | +11 (was 52) | Strong progress; blocked by PAD data |

---

## G. What Is Complete

The following items are implemented, tested, and ready for production use:

**Security controls:**
- `_SafeUnpickler` — blocks all known pickle RCE gadgets (os.system, subprocess, eval, importlib)
- `operator_auth` — SHA-256 PIN gate with timing-safe comparison; demo mode with audit trail
- `integrity` — HMAC-SHA256 embedding file signing; tamper detection at single-float granularity
- Backward compatibility for legacy files (load with WARNING, no data loss)
- `yaml.safe_load` throughout eval pipeline

**Application:**
- Full registration workflow with PAD gate on enrollment
- Full live recognition with PAD session (passive + active challenge)
- Model warm-up at startup (first-frame latency eliminated)
- Structured audit logging for auth, registration, and recognition events
- Rotating log file with configurable levels

**Eval framework (E1–E4):**
- Three protocol definitions (PAD-ISO-LAB, PAD-LIVE-SIM, PAD-REG)
- APCER/BPCER/latency/confusion metrics
- Clip loader, settings snapshot, evaluation harness
- Report builder, baseline regression detector, tier gate validator
- PAD baseline runbook (end-to-end procedure)
- Collection tool, manifest generator, QC pipeline

**Tests:**
- 218 unit + integration tests; 0 failures
- Adversarial tests for pickle gadgets, tamper detection, PIN timing
- Mocked webcam, DeepFace, and filesystem — no external dependencies in test suite

---

## H. What Still Requires Real-World Data

The following cannot be completed without human collectors and physical equipment:

1. **PAD dataset (P1 collection):**
   - 146 more bona fide presentations (28 subjects)
   - 50 PA-01 (print) presentations (10 subjects)
   - 49 PA-02 (static screen) presentations (9 more subjects)
   - 50 PA-03 (video replay) presentations (10 subjects)
   - Estimated: 1–2 weeks with a dedicated collector

2. **Baseline freeze:**
   - E3 harness run on completed dataset
   - E4 report generation and review
   - Dual sign-off (evaluation engineer + security reviewer)
   - Copy `summary.json` → `eval/baselines/heuristic_rgb_v1_standard.json`

3. **Latency measurement:**
   - P95 latency must be measured on target hardware (≤ 1200 ms gate)
   - Cannot be verified without a real webcam and the full DeepFace stack running

4. **System tests:**
   - Manual smoke test with real camera confirming registration → recognition flow
   - `tests/system/` tests require camera hardware

---

## I. What Can Wait Until v1.1

The following are improvements that do not block a v1 baseline freeze:

**Security (P2/P3):**
- SEC-007: Bind cosine threshold to signed enrollment record
- SEC-008: Apply low-light check in PAD session recognition path
- SEC-009: Full PadSession + active challenge on registration (not just single-frame burst)
- SEC-010: Reject username normalisation collisions explicitly
- SEC-012: Generic deny message in production mode (hide closest-user name)
- Embedding file permissions (OS-level `icacls` documentation)

**Performance (P2):**
- PERF-002: Haar cascade face presence pre-check before DeepFace inference
- PERF-003: Batch NumPy cosine similarity for > 20 users
- PERF-010: TF thread pool documentation (`TF_NUM_INTEROP_THREADS`)

**Testing (P2/P3):**
- `tests/system/` full-loop smoke tests
- `tests/unit/test_capture.py` direct unit tests
- Harness integration tests on `eval/fixtures/`
- CLI `main()` exit-code tests for tools

**Code quality:**
- `src/utility.py` empty file — delete or populate
- Shared `eval/datasets/layout.py` for `ManifestSample` dataclass (currently duplicated in tools and harness)
- Frame filename regex de-duplication (mismatch risk between `collect_pad_dataset` and `clip_loader`)
- `ruff` + `mypy` in dev dependencies
- CI GitHub Actions matrix (Python 3.10 + 3.11)

**Documentation:**
- Consolidate `CONFIGURATION.md` + `LOGGING.md` → README
- Security threat model documented in README for operators
- Demographics requirements for PAD dataset (age, skin tone, glasses)

---

## J. Risk Table

| Risk | Severity | Likelihood | Impact | Mitigation |
|------|----------|------------|--------|------------|
| Heuristic PAD fails PA-01 gate | High | Medium | Baseline fails; no production PAD claim | Expected weakness; plan ONNX model for v1.1 |
| BPCER > 5% from borderline→active on BF clips | Medium | Medium | Baseline fails | Longer clips (≥ 16 frames); review `borderline_high` setting |
| Capture SOP drift P0→P1 | Medium | Medium | Non-comparable metrics | Fix environment before P1; document in `collection_meta.json` |
| `FACE_LOCK_SIGNING_KEY` not set on production machine | High | High | Embeddings unsigned; tamper undetected | Operational: document mandatory setup; add startup check that refuses to run if both env vars missing and `--strict` flag is set |
| `FACE_LOCK_OPERATOR_PIN_HASH` not set on production machine | High | High | Registration unauthenticated | Same as above |
| Virtual camera attack bypasses PAD | High | Medium | Spoof accepted | Flip `allow_virtual_camera=False` for production config profile |
| Corrupt `.pkl` blocks all users (abort-all) | Medium | Low | Denial of service | Fix abort-all → skip-and-continue (P1 code change, 30 min) |
| Username collision silent overwrite | Medium | Low | Existing user's embeddings replaced | Add confirmation prompt or collision check (P1) |
| No CI gate; regression ships undetected | Medium | Medium | Regressions in test coverage | Add GitHub Actions `pytest -m "not system"` on push (P2) |
| TF thread contention starves OpenCV display | Low | Low | UI freezes | Document `TF_NUM_INTEROP_THREADS=1` |

---

## K. Actions Required Before Baseline Freeze

In order of dependency (must be sequential):

```
Step 1  Collect P1 dataset
        Target: 150 BF (30 subjects), 50 PA-01, 50 PA-02, 50 PA-03
        Tool:   python tools/collect_pad_dataset.py --collection-root eval/datasets/face_lock_v1 --collector-id <id>
        Time:   1–2 weeks

Step 2  Generate and validate manifest
        python tools/generate_manifest.py --collection-root eval/datasets/face_lock_v1 --strict
        Expected: 300 samples, 0 errors

Step 3  Run QC gate (must exit 0)
        python tools/qc_dataset.py --collection-root eval/datasets/face_lock_v1
        Exit 0 = PASS — only then proceed

Step 4  Run E3 evaluation harness
        python -m eval.harness.run_pad_eval \
          --protocol eval/protocols/pad_iso_lab_v1.yaml \
          --manifest eval/datasets/face_lock_v1/manifest.json \
          --collection-root eval/datasets/face_lock_v1 \
          --output-dir eval/reports/runs \
          --run-id face_lock_v1_heuristic_standard_001

Step 5  Generate E4 report
        python -m eval.reporting.report_builder \
          --run-dir eval/reports/runs/face_lock_v1_heuristic_standard_001
        Exit 0 = tier gate PASS — only then proceed

Step 6  Dual sign-off on REPORT.md
        Evaluation engineer + security reviewer — different people

Step 7  Freeze baseline
        copy eval\reports\runs\face_lock_v1_heuristic_standard_001\summary.json \
             eval\baselines\heuristic_rgb_v1_standard.json
        git add eval/baselines/heuristic_rgb_v1_standard.json
        git commit -m "freeze: PAD baseline heuristic_rgb_v1_standard"

Step 8  Set required env vars on all deployment machines
        FACE_LOCK_OPERATOR_PIN_HASH  (SHA-256 hex of operator PIN)
        FACE_LOCK_SIGNING_KEY        (secrets.token_hex(32))

Step 9  Final test suite verification on target hardware
        python -m pytest tests/ -m "not system" -q
        Expected: 218+ passing, 0 failures
```

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Engineering lead | | | ☐ |
| Security reviewer | | | ☐ |
| ML / PAD evaluation | | | ☐ |

**No production deployment may occur until all three roles have signed this document
and Step 7 (baseline freeze) above is complete.**

---

*Audit performed against repository state on 2026-06-03.  
Reference: `docs/security_audit_v1.md`, `docs/V1_RELEASE_CHECKLIST.md`,  
`docs/PAD_BASELINE_RUNBOOK.md`, `docs/V1_EXECUTION_PLAN.md`.*
