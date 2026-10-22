# Face Lock AI — Code Quality Audit v1

**Date:** 2026-05-30  
**Repository:** `C:\face_detection`

---

## Summary

| Area | Grade | Notes |
|------|-------|-------|
| Structure | B+ | Clear `src/` / `tools/` / `eval/` split |
| Typing | B | Good in eval/tools; partial in src |
| Documentation | B | Module docstrings strong; some API gaps |
| DRY | C+ | Duplicate path/bootstrap/manifest patterns |
| Testability | B | Eval well tested; register/capture gaps |
| Config | A- | Centralized frozen dataclasses in `config.py` |

---

## Dead / low-use code

| Symbol | File | Status | Recommendation |
|--------|------|--------|----------------|
| `load_face_model` | `src/ai_engine.py` | Unused in runtime | Call at app startup (perf) or document as optional |
| `generate_embeddings_from_images` | `src/ai_engine.py` | Unused | Keep for batch tooling; documented |
| `euclidean_distance` | `src/ai_engine.py` | Tests only | Keep (alternate metric reference) |
| `build_path` | `src/utils.py` | Tests only | Keep or merge into `PATH_SETTINGS` helpers |
| `run_pad_check` | `src/recognize.py` | Used only in `recognize_frame` | Keep (non-session PAD path) |
| `recognize_frame` | `src/recognize.py` | PAD-disabled + integration tests | Keep |
| `tests/system/` | `tests/system/__init__.py` | Empty placeholder | Add system tests or remove package |

**Not removed:** Items above are intentional API surface or test helpers; removing would break tests or future hooks.

---

## Duplicate code

| Pattern | Locations | Severity |
|---------|-----------|----------|
| `resolve_project_root()` | `src/config.py`, `tools/project_paths.py` | Intentional decoupling |
| Script bootstrap | `tools/_script_bootstrap.py` + 3 tool entry scripts | Consolidated ✓ |
| `ManifestSample` dataclass | `tools/generate_manifest.py`, `eval/harness/run_pad_eval.py` | Medium — extract shared `eval/datasets/schema.py` |
| Frame filename regex | `tools/collect_pad_dataset`, `eval/harness/clip_loader` | Medium — mismatch risk (`frame_0001` vs `frame_1`) |
| Tier gate evaluation | `run_pad_eval._evaluate_tier_gates`, `tier_gate_validator` | Low — E3 vs E4 roles |
| OpenCV overlay loops | `capture.py`, `register.py`, `collect_pad_dataset.py` | Low — different UX |
| Laplacian blur | `qc_dataset.py`, `pad_engine.py` heuristic | Low |

---

## Unused / redundant files

| Path | Notes |
|------|-------|
| `REPOSITORY_CLEANUP.md` | Historical; consider archiving |
| `CONFIGURATION.md`, `LOGGING.md` | Overlap with README; consolidate |
| `.snapshots/` | Tooling artifacts; gitignore if not needed |
| `face_lock_ai.egg-info/` | Build artifact; should be gitignored |
| `pytest.ini` | Redundant with `pyproject.toml` pytest section |

---

## Configuration duplication

| Source | Role |
|--------|------|
| `src/config.py` | **Canonical** runtime settings |
| `pyproject.toml` | Packaging + pytest |
| `requirements.txt` / `requirements-dev.txt` | Pip install lists (mirror pyproject) |
| `eval/protocols/*.yaml` | Eval PAD overrides |

**Issue:** `requirements.txt` and `pyproject.toml` can drift. **Fix:** Single source in `pyproject.toml`; generate requirements via `pip freeze` or document sync policy.

---

## Naming issues

| Issue | Example | Suggestion |
|-------|---------|------------|
| Package name `eval` shadows builtin | `import eval.harness` | Accept with lint ignore; never add `eval.py` |
| `subj_NNN` vs folder `subject_NNN` | collector vs disk | Document mapping (already in tools) |
| `BF` vs `bona_fide` | manifest vs folders | Consistent in spec |

---

## Long functions (review candidates)

| Function | File | Lines (approx) | Action |
|----------|------|----------------|--------|
| `run_collector` | `tools/collect_pad_dataset.py` | ~80 | Split prompt vs capture |
| `run_capture_session` | `tools/collect_pad_dataset.py` | ~100 | OK for UI loop |
| `run_qc` / scan logic | `tools/qc_dataset.py` | Large file | Extract check classes |
| `recognize_live` | `src/recognize.py` | ~130 | Extract loop body |
| `HeuristicPassiveScorer.score` | `src/pad_engine.py` | Medium | Do not refactor (PAD frozen) |

---

## Type hints & docstrings

| Module | Type hints | Docstrings |
|--------|------------|------------|
| `eval/metrics/*` | Strong | Good |
| `eval/harness/*` | Strong | Good |
| `tools/*` | Strong | Good |
| `src/pad_engine.py` | Strong | Good |
| `src/recognize.py` | Partial (`Any` for frames) | Good |
| `src/register.py` | Partial | Adequate |
| `src/capture.py` | Partial | Adequate |
| `app.py` | Good | Good |

**Gaps:** OpenCV frames typed as `Any` throughout (acceptable without `cv2.typing`).

---

## Error handling patterns

| Pattern | Location | Quality |
|---------|----------|---------|
| Broad `except Exception` | `app.py`, `load_all_user_embeddings` | Acceptable at boundary; log context |
| Fail-closed PAD | `pad_engine`, eval mapper | Good |
| Validation errors | `tools/*` custom exceptions | Good |

---

## Packaging & imports

| Item | Status |
|------|--------|
| `src/__init__.py` | Present + `__version__` ✓ |
| `tools/__init__.py` | Present ✓ |
| `eval/__init__.py` | Present ✓ |
| `tests/__init__.py` | Added ✓ |
| `tests/tools/__init__.py` | Added ✓ |
| Editable install | `pyproject.toml` ✓ |

---

## Safe fixes applied (Phase 6)

- Consolidated tool script bootstrap → `tools/_script_bootstrap.py`
- Added `tests/__init__.py`, `tests/tools/__init__.py`
- Added `__version__` to `src/__init__.py`
- Security docstrings on `load_pickle` / `save_pickle`
- Clarified `load_face_model` docstring (optional warm-up API)
- Removed stale comment on `LOGGING_LEVEL` in `utils.py`

---

## Recommended refactors (post-v1, not applied)

1. Shared `eval/datasets/layout.py` for manifest types + frame patterns  
2. Merge docs (`CONFIGURATION.md` → README)  
3. Add `mypy` / `ruff` to dev dependencies  
4. Split `qc_dataset.py` into check modules  
