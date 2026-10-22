# Face Lock AI — Security Audit v1

**Date:** 2026-05-30  
**Scope:** Application (`app.py`, `src/`), dataset tooling (`tools/`), evaluation (`eval/`)  
**Auditor role:** Biometric MVP pre-release review  
**Threat model:** Single-user desktop demo; physical access to machine; local webcam; no network API

---

## Executive summary

Face Lock AI is a **local CLI biometric demo**, not a hardened access-control product. There is **no authentication layer** (no operator PIN, no OS integration, no session tokens). Security relies on **face embeddings + heuristic PAD** under the assumption that only trusted operators control the workstation.

**Critical gaps for production:** unsafe pickle deserialization, unauthenticated registration, weak PAD against replay/virtual camera, and embedding file integrity.

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High | 5 |
| Medium | 8 |
| Low | 6 |

---

## Findings

### SEC-001 — Unauthenticated operator access

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Impact** | Anyone at the console can register users, run recognition, or modify `encodings/` |
| **Location** | `app.py` (menu loop), `src/register.py`, `src/recognize.py` |
| **Recommendation** | Document OS-login assumption; add operator PIN or Windows Hello gate before register/unlock; restrict CLI to privileged account |
| **Fix priority** | P0 (before any non-demo deployment) |

---

### SEC-002 — Pickle deserialization (RCE risk)

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Impact** | Malicious or swapped `.pkl` in `encodings/` executes arbitrary code on `pickle.load` |
| **Location** | `src/utils.py` (`load_pickle`), `src/ai_engine.py` (`load_all_user_embeddings`) |
| **Recommendation** | Replace with JSON + NumPy `.npy` or signed protobuf; if pickle retained, HMAC-sign blobs with key in OS keychain; validate schema version and `model_name` |
| **Fix priority** | P0 |

---

### SEC-003 — Embedding forgery / swap attack

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Impact** | Attacker with write access to `encodings/` can impersonate any user without face match |
| **Location** | `src/ai_engine.py` L454–481 (glob load all `*.pkl`) |
| **Recommendation** | File permissions (0600), integrity hash, encryption at rest, audit log on register/load |
| **Fix priority** | P0 |

---

### SEC-004 — Open registration (enrollment abuse)

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Impact** | Attacker registers their face as trusted user; no approval workflow |
| **Location** | `src/register.py`, `app.py` choice `1` |
| **Recommendation** | Separate admin mode; require existing operator auth; cap max users; log enrollment events |
| **Fix priority** | P1 |

---

### SEC-005 — Virtual camera / replay feed allowed

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Impact** | OBS, ManyCam, or injected virtual devices can feed pre-recorded video |
| **Location** | `src/config.py` (`allow_virtual_camera: True`), PAD heuristics |
| **Recommendation** | Default `allow_virtual_camera=False` for production; device fingerprinting; PA-04 eval coverage before enabling |
| **Fix priority** | P1 |

---

### SEC-006 — Replay attack (screen / print / video)

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Impact** | Presentation attacks may pass heuristic PAD + face match |
| **Location** | `src/pad_engine.py` (`HeuristicPassiveScorer`), `eval/protocols/*.yaml` |
| **Recommendation** | Train/deploy passive model v1; enforce active challenge on unlock; tune APCER/BPCER gates from real data |
| **Fix priority** | P1 (model), P0 (baseline eval before release claim) |

---

### SEC-007 — Threshold bypass via config change

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Impact** | Lowering `cosine_threshold` in `config.py` accepts impostors; stored per-user threshold in pickle is **ignored** at runtime |
| **Location** | `src/config.py` L87, `src/recognize.py` L367, `src/ai_engine.py` L397 (stored but unused) |
| **Recommendation** | Bind threshold to signed enrollment record or require re-register on policy change; log threshold at match time |
| **Fix priority** | P2 |

---

### SEC-008 — PAD bypass path (low-light check skipped)

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Impact** | When PAD session is active, `recognize_identity` runs without `is_bad_lighting` gate |
| **Location** | `src/recognize.py` L421–429 vs L441–446 |
| **Recommendation** | Apply lighting check before identity match in PAD session path |
| **Fix priority** | P2 |

---

### SEC-009 — Registration PAD weaker than live PAD

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Impact** | Enrollment uses burst/single-frame PAD; attacker may register spoof images |
| **Location** | `src/register.py`, `pad_engine.check_single_image_pad` |
| **Recommendation** | Require full `PadSession` + active challenge on registration |
| **Fix priority** | P2 |

---

### SEC-010 — Username normalization collision

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Impact** | `John` / `john` / `john!` map to same `safe_name`; overwrite prior user |
| **Location** | `src/utils.py` `clean_name`, `src/ai_engine.py` `get_user_embedding_path` |
| **Recommendation** | Reject collisions; use UUID suffix; confirm overwrite interactively |
| **Fix priority** | P2 |

---

### SEC-011 — Path traversal via username (mitigated)

| Field | Value |
|-------|-------|
| **Severity** | Low (mitigated) |
| **Impact** | `../` in username could escape data dirs without sanitization |
| **Location** | `src/utils.py` `clean_name` |
| **Recommendation** | Keep `clean_name`; add unit test for traversal strings; resolve paths under `DATA_DIR` with `is_relative_to` |
| **Fix priority** | P3 |

---

### SEC-012 — Information disclosure on deny

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Impact** | UI/logs reveal closest registered user and distance on failed match |
| **Location** | `src/recognize.py` L263–276 |
| **Recommendation** | Generic deny message in production mode; detailed metrics in debug only |
| **Fix priority** | P3 |

---

### SEC-013 — Dataset poisoning (eval pipeline)

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Impact** | Malicious frames in `eval/datasets/` skew APCER/BPCER; false confidence in release |
| **Location** | `tools/collect_pad_dataset.py`, `tools/generate_manifest.py`, `eval/harness/` |
| **Recommendation** | QC gate (`qc_dataset.py`) before harness; manifest checksums; separate collector role; gitignore raw datasets |
| **Fix priority** | P2 |

---

### SEC-014 — YAML protocol loading

| Field | Value |
|-------|-------|
| **Severity** | Low (mitigated) |
| **Impact** | Unsafe `yaml.load` could execute tags |
| **Location** | `eval/harness/run_pad_eval.py`, `eval/reporting/tier_gate_validator.py` |
| **Status** | Uses `yaml.safe_load` ✓ |
| **Recommendation** | Pin PyYAML; validate protocol schema |
| **Fix priority** | P3 |

---

### SEC-015 — No file upload surface (N/A)

| Field | Value |
|-------|-------|
| **Severity** | N/A |
| **Impact** | No HTTP upload; images from local webcam/disk only |
| **Recommendation** | If web UI added later, enforce size limits, MIME validation, virus scan |
| **Fix priority** | Future |

---

### SEC-016 — Subprocess in eval snapshot

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Impact** | `git` subprocess for commit SHA in eval artifacts |
| **Location** | `eval/harness/settings_snapshot.py` |
| **Recommendation** | Acceptable for dev; sandbox in CI |
| **Fix priority** | P4 |

---

## Attack surface matrix

| Vector | Present? | Control today | Residual risk |
|--------|----------|---------------|---------------|
| Physical console | Yes | None | High |
| Pickle RCE | Yes | Trust local FS | Critical |
| Face spoof (print/screen) | Yes | Heuristic PAD | High |
| Virtual camera | Yes | Config allows | High |
| Replay video | Yes | Active challenge (partial) | Medium |
| Enrollment abuse | Yes | PAD on save (weak) | High |
| Threshold tampering | Yes | Config file | Medium |
| Dataset poisoning | Eval only | QC tool | Medium |

---

## Recommended security roadmap

1. **P0:** Replace pickle; operator auth; lock down `encodings/` permissions  
2. **P1:** Disable virtual camera default; run PAD eval on real dataset; baseline gates  
3. **P2:** Unified PAD on register + recognize; lighting gate; collision-safe names  
4. **P3:** Production deny messages; path hardening tests  

---

## Compliance note

This system processes **biometric data** (face embeddings). For GDPR/CCPA/BIPA contexts you will need: consent, retention policy, secure deletion, and purpose limitation documentation—not present in v1 prototype.
