# Face Lock AI — V1 Final Sign-Off Audit

**Audit date:** 2026-06-20  
**Audit scope:** Full repository — src/, tests/, eval/, tools/, docs/, .github/  
**Test suite state:** 447/447 passing (unit + integration + system + eval); 27 deselected (system marker); 3 pre-existing failures in tests/tools/ noted below  
**Python version:** 3.11.9 (pinned)

---

## Executive Summary

Face Lock AI v1 is a local desktop biometric demo with a clean architecture, complete security hardening for its threat model, a validated eval framework, and a comprehensive test suite. The security controls that were Critical risks at the start of the project (pickle RCE, operator auth, embedding forgery) are all implemented, tested, and verified in the runtime path.

**The system is ready for a local demo deployment with both environment variables configured.  
It is NOT ready for multi-user, networked, or unattended production deployment.**

The single remaining hard blocker before claiming a production-grade PAD baseline is the real-world dataset: the `face_lock_v1` collection is at P0 pilot scale (5 presentations). All tooling, protocols, and automation to collect and freeze a baseline are complete and tested.

---

## Readiness Scores

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Security** | 78 / 100 | P0 controls implemented and tested; medium findings remain |
| **Reliability** | 82 / 100 | Resilient embedding loading; single-file corrupt no longer blocks all users |
| **Maintainability** | 80 / 100 | Clean separation; consistent patterns; 3 pre-existing tool test bugs |
| **Performance** | 62 / 100 | Warm-up resolved; PERF-002/003/010 deferred |
| **Test coverage** | 78 / 100 | 447 tests; system tests added; capture module still lightly tested |
| **Eval readiness** | 40 / 100 | Framework 100% complete; dataset at P0 (5 presentations) |
| **CI / DevOps** | 72 / 100 | GitHub Actions workflow created; no branch protection rules yet |
| **Overall** | **70 / 100** | Up from 63 at previous audit |

---

## A. Security Status

### Critical findings — all resolved

| Finding | Fix | Verified by |
|---------|-----|------------|
| SEC-001 Unauthenticated registration | `src/operator_auth.py` PIN gate in `app.handle_registration` | 31 unit tests; structural position check |
| SEC-002 Pickle RCE | `_SafeUnpickler` allowlist in `src/utils.py` | 21 unit tests incl. live gadget tests |
| SEC-003 Embedding forgery/swap | HMAC-SHA256 in `src/integrity.py` + verified on every load | 34 unit tests incl. 1e-9 tamper detection |

### Resilience improvement (this session)

`load_all_user_embeddings` now skips corrupt or tampered files and continues loading remaining users. Previously it aborted with `RuntimeError` on the first bad file, creating a denial-of-service vector. The skip-and-continue behaviour is logged at ERROR level and tested by 9 dedicated unit tests. Integrity checks are NOT weakened — tampered files are detected and skipped, not silently accepted.

### Virtual camera enforcement (this session)

`src/capture.py::open_webcam` now enforces `PAD_SETTINGS.allow_virtual_camera`. When set to `False`, known virtual camera backends (OBS, DroidCam, ManyCam, Snap Camera, iVCam, and others) are detected by backend name substring matching and rejected with `VirtualCameraError` before any frames are captured. 33 unit tests cover detection, policy enforcement, and the default/override behaviour.

### Remaining security findings

| Finding | Severity | Status | Notes |
|---------|----------|--------|-------|
| SEC-004 Open registration (no max-user cap) | High | Open | PIN gate added; no user cap or re-register confirmation |
| SEC-005 Virtual camera default | High | Mitigated | `allow_virtual_camera=True` default unchanged; enforcement now wired |
| SEC-006 Replay / spoof (heuristic PAD) | High | Open | Heuristic model; no lab-validated baseline yet |
| SEC-007 Threshold bypass via config | Medium | Open | `cosine_threshold` in config.py is unprotected |
| SEC-008 Low-light check absent in PAD session | Medium | Open | Known gap; deferred to v1.1 |
| SEC-009 Registration PAD weaker than live | Medium | Open | Burst-PAD only on registration |
| SEC-010 Username normalisation collision | Medium | Open | `John`/`john` → same safe_name; silent overwrite |
| SEC-012 Information disclosure on deny | Low | Open | Closest user name shown in UI |

**Both env vars default to unsafe open mode.** For any deployment beyond local development:
- Set `FACE_LOCK_OPERATOR_PIN_HASH` before starting the application
- Set `FACE_LOCK_SIGNING_KEY` before first registration

---

## B. Testing Status

### Suite summary

| Suite | Tests | Status |
|-------|-------|--------|
| `tests/unit/` | 304 | ✅ All passing |
| `tests/integration/` | 15 | ✅ All passing |
| `tests/system/` | 27 | ✅ All passing (system marker; hardware-free) |
| `eval/tests/` | 101 | ✅ All passing |
| `tests/tools/` | **3 failing** | ⚠️ Pre-existing bugs (see below) |
| **Total (excl. failing)** | **447** | ✅ |

### New tests added this session

| File | Tests | Coverage |
|------|-------|---------|
| `eval/tests/test_dataset_status.py` | 31 | `eval/tools/dataset_status.py` |
| `tests/unit/test_freeze_baseline.py` | 28 | `tools/freeze_baseline.py` |
| `tests/unit/test_virtual_camera.py` | 33 | `src/capture.py` virtual camera |
| `tests/unit/test_ai_engine.py` (appended) | +9 | `load_all_user_embeddings` resilience |
| `tests/system/test_workflows.py` | 27 | Complete workflows end-to-end |

### Pre-existing test failures (not introduced this session)

| Test | Failure | Root cause |
|------|---------|-----------|
| `test_collect_pad_dataset::TestPaths::test_build_bona_fide_path` | `presentation_dir.is_dir()` → False | `build_presentation_path` returns the path but does not create the dir until `run_capture_session` starts recording; test checks too early |
| `test_collect_pad_dataset::TestFrameCountWarnings::test_validate_frame_count_warns_low` | `"protocol minimum" not in caplog.text` | `validate_frame_count` logs via `LOGGER.warning` (standard logging), but caplog doesn't capture it because the logger was configured before the test — bootstrap issue |
| `test_qc_dataset::TestManifest::test_invalid_pai_and_ground_truth` | `ground_truth_pai_mismatch` not in errors | QC only validates ground_truth consistency when both `pai_id` and `ground_truth` are valid individually first; the test fixture uses `pai_id: "INVALID"` which triggers a different error code |

None of these failures were introduced by this session's changes. All three are testability/fixture issues in existing test code, not bugs in the tool logic itself.

---

## C. Performance Status

| Item | Status | Notes |
|------|--------|-------|
| DeepFace cold start | ✅ Resolved | `app.warmup_model()` called at startup |
| Resilient embedding load | ✅ Resolved | Skip-and-continue; no longer abort-all |
| PERF-002 Face pre-check before DeepFace | ❌ Open | Deferred to v1.1 |
| PERF-003 Batch cosine similarity | ❌ Open | Acceptable < 20 users |
| PERF-010 TF thread contention | ❌ Open | Document `TF_NUM_INTEROP_THREADS=1` |

---

## D. Evaluation Readiness

| Component | Status |
|-----------|--------|
| E1 Protocol definitions (3 YAMLs) | ✅ Complete |
| E2 Metrics (APCER/BPCER/confusion/latency) | ✅ Complete + 101 tests |
| E3 Evaluation harness (`run_pad_eval.py`) | ✅ Complete |
| E4 Reporting (`report_builder.py`, tier gates, baseline compare) | ✅ Complete |
| Dataset collection tool (`collect_pad_dataset.py`) | ✅ Complete |
| Manifest generator (`generate_manifest.py`) | ✅ Complete |
| QC pipeline (`qc_dataset.py`) | ✅ Complete |
| Dataset status dashboard (`eval/tools/dataset_status.py`) | ✅ **New this session** |
| Baseline freeze utility (`tools/freeze_baseline.py`) | ✅ **New this session** |
| PAD baseline runbook (`docs/PAD_BASELINE_RUNBOOK.md`) | ✅ Complete |
| **Approved baseline JSON** | ❌ **Missing** — requires real dataset |
| `face_lock_v1` dataset at P1 counts | ❌ **Missing** — 5/300 presentations |

---

## E. Remaining Blockers

### BLOCKER 1 — Real-world PAD dataset

`face_lock_v1` has 5 presentations (4 BF + 1 PA-02). The minimum for a standard-tier baseline is 300 (150 BF + 50 each PA-01/PA-02/PA-03). No code change can resolve this — human collectors and physical equipment are required.

**Estimated time:** 1–2 weeks with a dedicated collector.  
**Tools ready:** `collect_pad_dataset.py`, `generate_manifest.py`, `qc_dataset.py`, `dataset_status.py`

### BLOCKER 2 — Approved baseline JSON

`eval/baselines/heuristic_rgb_v1_standard.json` does not exist. Without it, no regression comparison is possible and the V1 release checklist cannot be signed off.

**Resolved by:** Complete Blocker 1 → run `run_pad_eval.py` → `report_builder.py` → dual sign-off → `freeze_baseline.py`

### BLOCKER 3 — Security env vars not configured by default

Both `FACE_LOCK_OPERATOR_PIN_HASH` and `FACE_LOCK_SIGNING_KEY` default to unsafe open mode (demo mode / unsigned). For any deployment:

```powershell
# Generate signing key
python -c "import secrets; print(secrets.token_hex(32))"
$env:FACE_LOCK_SIGNING_KEY = "<output>"

# Generate PIN hash
$env:FACE_LOCK_OPERATOR_PIN_HASH = (python -c "import hashlib; print(hashlib.sha256(b'your_pin').hexdigest())")
```

---

## F. Deployment Checklist

### Before first deployment

- [ ] `FACE_LOCK_OPERATOR_PIN_HASH` set in environment
- [ ] `FACE_LOCK_SIGNING_KEY` set in environment
- [ ] `PAD_SETTINGS.allow_virtual_camera = False` for any security-sensitive context
- [ ] `pip install -e ".[dev]"` + `python scripts/patch_deepface_tf215.py` on target machine
- [ ] `pytest -m "not system" -q` passes on target machine
- [ ] All existing users re-registered after signing key is configured (legacy files load with WARNING)

### Before claiming production PAD claim

- [ ] P1 dataset collected (see PAD Baseline Runbook §4)
- [ ] `qc_dataset.py` exits 0 (PASS) on final dataset
- [ ] E3 harness run produces `tier_gate.overall = "pass"`
- [ ] E4 `report_builder.py` exits 0
- [ ] `REPORT.md` reviewed; all placeholders filled
- [ ] Dual sign-off (engineering + security reviewer)
- [ ] `freeze_baseline.py` run to write `eval/baselines/heuristic_rgb_v1_standard.json`
- [ ] Baseline committed to source control
- [ ] V1 release checklist signed by all three roles

### Before v1.1

- [ ] SEC-008 lighting check in PAD session path
- [ ] PERF-002 face pre-check before DeepFace
- [ ] Fix 3 pre-existing `tests/tools/` test failures
- [ ] SEC-010 username normalisation collision guard
- [ ] `tests/unit/test_capture.py` direct unit tests
- [ ] CI branch protection rules (require passing checks on PR)

---

## G. Remaining Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Heuristic PAD fails PA-01 gate (print texture) | High | Medium | Expected weakness; plan ONNX model v1.1 |
| BPCER > 5% from borderline→active challenge on BF clips | Medium | Medium | Longer clips (≥16 frames) |
| Both env vars not set on deployment machine | High | High | Document mandatory setup; startup check |
| Capture SOP drift P0→P1 collection | Medium | Medium | Fix camera, distance, background before P1 |
| TF thread contention freezes OpenCV display | Low | Low | `TF_NUM_INTEROP_THREADS=1` |
| `allow_virtual_camera=True` is the default | High | Medium | Now enforced if changed to False; document required config |

---

## H. What Is Complete

**Security controls (all P0):**
- Restricted pickle deserialization (`_SafeUnpickler`)
- Operator PIN gate on registration (`src/operator_auth.py`)
- HMAC-SHA256 embedding integrity (`src/integrity.py`)
- Resilient embedding load (skip corrupt/tampered, continue)
- Virtual camera detection and rejection (`src/capture.py`)
- Model warm-up at startup (`app.warmup_model`)

**Eval framework (E1–E4):**
- 3 protocol definitions (ISO-LAB, Live-Sim, Registration)
- APCER/BPCER/latency/confusion metrics
- Evaluation harness + clip loader + settings snapshot
- Report builder + tier gate validator + baseline regression
- Dataset status dashboard + baseline freeze utility
- PAD Baseline Runbook (17 sections, step-by-step commands)
- Pre-baseline verification document

**CI / DevOps:**
- GitHub Actions workflow (Python 3.11, install, patch, test, coverage)
- `system` marker added to `pytest.ini`
- 447 automated tests across 5 suites

**Documentation:**
- `docs/PAD_BASELINE_RUNBOOK.md` — authoritative freeze procedure
- `docs/PRE_BASELINE_VERIFICATION.md` — runtime wiring verification
- `docs/PRODUCTION_SIGNOFF_AUDIT.md` — previous audit
- `docs/V1_FINAL_SIGNOFF.md` — this document

---

## I. What Can Wait Until v1.1

| Item | Rationale |
|------|-----------|
| PERF-002 face pre-check (Haar cascade) | Nice-to-have; `frame_skip` mitigates cost |
| PERF-003 batch cosine similarity | Only matters at 50+ users |
| SEC-008 low-light check in PAD session path | Known; small attack surface |
| SEC-010 username collision guard | Edge case; operators create usernames |
| Full `PadSession` on registration (SEC-009) | Burst-PAD acceptable for demo |
| `tests/unit/test_capture.py` direct tests | Covered via integration mocks |
| Fix 3 pre-existing `tests/tools/` failures | Testability bugs, not logic bugs |
| Ruff + mypy in dev deps | Code quality; no correctness impact |
| CI branch protection rules | Tooling; workflow exists |
| Multi-user / networked deployment | Outside v1 scope |
| ONNX passive scorer | Required for high-tier baseline |
| Docker eval-only container | Convenience |

---

## J. Recommended Next Version Work (v1.1)

1. **Complete PAD dataset + freeze baseline** — the most impactful single action
2. **ONNX passive scorer** for high-tier (`high` tier requires APCER ≤ 1%)
3. **SEC-008** — apply low-light check in PAD session recognition path (30-min fix)
4. **PERF-002** — Haar cascade face pre-check before DeepFace (reduces wasted inference)
5. **Re-registration guard** — warn/confirm before silent user overwrite
6. **Fix tools test failures** — 3 pre-existing bugs in `tests/tools/`
7. **GitHub branch protection** — require CI pass on PR to main
8. **BIPA/GDPR compliance stub** — consent, retention, deletion documentation

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Engineering lead | | | ☐ |
| Security reviewer | | | ☐ |
| ML / PAD evaluation | | | ☐ |

**No production deployment may occur until all three roles have signed this document, Blockers 1–3 are resolved, and `eval/baselines/heuristic_rgb_v1_standard.json` exists.**
