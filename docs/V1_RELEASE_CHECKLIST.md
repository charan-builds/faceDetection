# Face Lock AI — V1 Release Checklist

**Target:** Production-ready v1 MVP (desktop demo with PAD eval pipeline)  
**Date:** 2026-05-30

Use this checklist before tagging `v1.0.0` or deploying beyond developer machines.

---

## Must fix before release (blockers)

- [ ] **SEC-002** Replace or sign pickle embedding storage (`encodings/*.pkl`)
- [ ] **SEC-001** Document threat model + add operator gate (PIN / OS user assumption in README)
- [ ] **SEC-003** Restrict `encodings/` file permissions; integrity validation on load
- [ ] **Run full test suite** — `pip install -e ".[dev]" && pytest` passes on Windows
- [ ] **PAD baseline eval** — run harness on `face_lock_v1` (or approved dataset); APCER/BPCER within protocol gates
- [ ] **Freeze baseline JSON** — copy approved `summary.json` → `eval/baselines/heuristic_rgb_v1_standard.json`
- [ ] **QC gate** — `python -m tools.qc_dataset` PASS on release dataset
- [ ] **Manifest strict** — `python -m tools.generate_manifest --strict`
- [ ] **Python 3.11 venv** — `python --version` shows 3.10 or 3.11 (not 3.12); see [docs/python_environment_audit.md](python_environment_audit.md)
- [ ] **Remove secrets** from repo — no API keys, no personal images in git
- [ ] **`.gitignore`** — `encodings/`, `data/`, `eval/datasets/*/`, `*.pkl`, `face_lock_ai.egg-info/`
- [ ] **Editable install documented** — README install section verified

---

## Should fix (high value, not blocking demo)

- [ ] **SEC-005** Default `allow_virtual_camera=False` for “production” profile in config
- [ ] **SEC-004** Registration approval / admin mode
- [ ] **SEC-008** Low-light check in PAD session recognition path
- [ ] **PERF-001** Warm up DeepFace model at startup
- [ ] **TEST** `test_register.py` + `test_capture.py`
- [ ] **TEST** Harness integration test on `eval/fixtures/`
- [ ] **CODE** Unify frame filename rules (`tools` vs `clip_loader`)
- [ ] **CODE** Shared manifest dataclass module
- [ ] **DOC** Security audit findings linked from README
- [ ] **CI** GitHub Actions: pytest on push (no system tests)

---

## Nice to have

- [ ] **PERF-003** Batch embedding comparison matrix
- [ ] **PERF-002** Face presence pre-check before DeepFace
- [ ] **SEC-012** Generic deny messages in production mode flag
- [ ] **DOC** Consolidate CONFIGURATION.md + LOGGING.md into README
- [ ] **Lint** Add `ruff` + `mypy` to dev deps
- [ ] **CLI** `face-lock` console script smoke test
- [ ] **Report** Automated `build-pad-report` in release pipeline
- [ ] **Docker** Optional CPU container for eval-only runs
- [ ] **Version** Bump `src/__version__` + git tag automation

---

## Release verification script (operator)

```powershell
cd C:\face_detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python -c "import src; import tools; import eval.harness; print('imports ok')"
pytest -m "not system" -q

python -m tools.qc_dataset --collection-root eval/datasets/face_lock_v1
python -m tools.generate_manifest --strict --collection-root eval/datasets/face_lock_v1

python -m eval.harness.run_pad_eval ^
  --protocol eval/protocols/pad_iso_lab_v1.yaml ^
  --manifest eval/datasets/face_lock_v1/manifest.json ^
  --collection-root eval/datasets/face_lock_v1

python app.py   # manual smoke: register + recognize
```

---

## Sign-off

| Role | Name | Date | OK |
|------|------|------|-----|
| Engineering | | | ☐ |
| Security review | | | ☐ |
| ML / PAD eval | | | ☐ |

---

## Post-release monitoring (v1.1)

- Collect field failure logs from `logs/face_lock.log`
- Track false accept / false reject reports
- Re-run eval when PAD model or threshold changes
