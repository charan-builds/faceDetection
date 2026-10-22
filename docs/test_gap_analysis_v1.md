# Face Lock AI — Test Gap Analysis v1

**Date:** 2026-05-30  
**Test roots:** `tests/`, `eval/tests/` (pytest `testpaths` in `pyproject.toml`)

---

## Coverage overview

| Area | Modules | Test files | Est. coverage |
|------|---------|------------|---------------|
| `src/` | 10 | 8 test files | ~65% logic |
| `tools/` | 6 | 4 test files | ~55% logic |
| `eval/` (excl. tests) | 13 | 10 test files | ~80% logic |
| **Total** | **29** | **22** | **~70%** |

*Estimate based on module/function mapping, not line coverage (run `pytest --cov` locally).*

---

## `src/` module matrix

| Module | Unit | Integration | System | Failure tests | Gap severity |
|--------|------|-------------|--------|---------------|--------------|
| `config.py` | ✓ `test_config.py` | — | — | Partial | Low |
| `utils.py` | ✓ `test_utils.py` | — | — | ✓ missing pickle | Low |
| `ai_engine.py` | ✓ `test_ai_engine.py` | — | — | Partial | Low |
| `pad_engine.py` | ✓ `test_pad_engine.py` | — | — | Good | Low |
| `display.py` | ✓ `test_display.py` | — | — | Partial | Low |
| `logging_config.py` | ✓ `test_logging_config.py` | — | — | Partial | Low |
| `recognize.py` | Partial | ✓ `test_recognize.py` | — | Good | Medium |
| `register.py` | **None** | **None** | **None** | **None** | **High** |
| `capture.py` | **None** | Mocked only | **None** | **None** | **High** |
| `app.py` | **None** | **None** | **None** | **None** | Medium |

---

## `tools/` module matrix

| Module | Tests | Gaps |
|--------|-------|------|
| `project_paths.py` | ✓ | — |
| `collect_pad_dataset.py` | ✓ validation, paths, timing | No camera loop, no `main()` CLI |
| `generate_manifest.py` | ✓ scan logic | No full CLI integration |
| `qc_dataset.py` | ✓ checks | No full CLI integration |
| `_repo_path.py` | — | Low value |
| `_script_bootstrap.py` | — | Add smoke test |

---

## `eval/` module matrix

| Module | Tests | Gaps |
|--------|-------|------|
| `metrics/apcer_bpcer.py` | ✓ | — |
| `metrics/confusion.py` | ✓ | — |
| `metrics/latency_stats.py` | ✓ | — |
| `harness/outcome_mapper.py` | ✓ | — |
| `harness/clip_loader.py` | ✓ | Frame pattern edge cases |
| `harness/settings_snapshot.py` | ✓ | git subprocess mocked? |
| `harness/run_pad_eval.py` | ✓ E2E mock | No real `src.pad_engine` integration |
| `reporting/report_builder.py` | ✓ | `append_baseline_section`, `main()` |
| `reporting/baseline_compare.py` | ✓ | — |
| `reporting/tier_gate_validator.py` | ✓ | — |

---

## Missing test categories

### Integration tests needed

1. **Register → encodings → recognize** golden path (mock DeepFace)  
2. **Manifest → clip_loader → harness** with `eval/fixtures/` images  
3. **collect → manifest → qc → harness** pipeline on temp dataset  
4. **PAD session path** in `recognize_live` (mock camera)

### System tests needed (`tests/system/` empty)

1. Manual/CI optional: webcam + TF marker `@pytest.mark.system`  
2. Full `app.py` menu smoke with stdin mock  
3. Console script entry points after `pip install -e .`

### Failure / adversarial tests needed

| Scenario | Target |
|----------|--------|
| Corrupt pickle | `load_all_user_embeddings` |
| Empty encodings dir | `recognize_live` |
| Camera open failure | `capture.open_webcam` |
| PAD ATTACK outcome | `recognize_live` deny path |
| Invalid manifest | `run_pad_eval` |
| Strict QC fail | `qc_dataset --strict` |
| Username collision | `clean_name` + register |
| Missing frames in clip | `clip_loader` |

---

## Test quality strengths

- Eval metrics thoroughly unit tested  
- PAD engine good unit coverage  
- `test_recognize.py` covers PAD + identity branches with mocks  
- Tool validation logic well covered without camera  

---

## Recommended test plan (priority)

| Priority | Test | File |
|----------|------|------|
| P0 | Register workflow (mock cv2 + DeepFace) | `tests/unit/test_register.py` |
| P0 | Capture helpers | `tests/unit/test_capture.py` |
| P1 | Pickle corrupt file handling | `tests/unit/test_ai_engine.py` |
| P1 | Harness on eval/fixtures (real pad_engine) | `eval/tests/test_harness_fixtures.py` |
| P2 | CLI `main()` exit codes | `tests/tools/test_*_cli.py` |
| P2 | `_script_bootstrap` | `tests/tools/test_script_bootstrap.py` |
| P3 | System marker webcam test | `tests/system/test_live_demo.py` |

---

## Commands

```powershell
pip install -e ".[dev]"
pytest --cov=src --cov=tools --cov=eval --cov-report=term-missing
pytest -m "not system"   # default CI
pytest -m system         # manual GPU/camera
```

---

## Test readiness score

| Metric | Score /10 |
|--------|-----------|
| Unit depth (eval) | 9 |
| Unit depth (src) | 6 |
| Integration | 5 |
| System | 1 |
| Failure/adversarial | 4 |
| **Overall test readiness** | **6/10** |
