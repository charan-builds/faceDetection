# Face Lock AI — Python Environment Audit

**Date:** 2026-05-30  
**Reporter context:** `pip install -e ".[dev]"` fails on **Python 3.12.4**  
**Error:** `No matching distribution found for tensorflow==2.15.1`

---

## Executive summary

| Item | Finding |
|------|---------|
| **Root cause** | TensorFlow **2.15.1 publishes no wheels for Python 3.12** (only cp39, cp310, cp311) |
| **Recommended solution** | **Option A** — keep TF/Keras 2.15 stack; use **Python 3.11.x** |
| **Not recommended now** | **Option B** — upgrade to TF 2.16+ / Keras 3 for Python 3.12 (high regression risk) |
| **Config bug fixed** | `pyproject.toml` previously allowed `3.12` via `requires-python = "<3.13"` |

---

## Root cause (detailed)

When you run `pip install -e ".[dev]"` on Python 3.12:

1. Pip resolves dependencies from `pyproject.toml`.
2. It requests `tensorflow==2.15.1`.
3. PyPI has **no** `tensorflow-2.15.1-cp312-*` wheel for any platform.
4. Pip reports: **No matching distribution found for tensorflow==2.15.1**.
5. The entire install aborts — **pytest never installs** because dev extras never resolve.

This is **not** a broken pin, SSL issue, or Face Lock packaging bug. It is a **Python ABI mismatch**: the pinned TensorFlow release predates official Python 3.12 wheels.

### Evidence (PyPI wheel tags for `tensorflow==2.15.1`)

| Python tag | Wheels published |
|------------|------------------|
| cp39 | Yes |
| cp310 | Yes |
| cp311 | Yes |
| **cp312** | **No** |

Source: [tensorflow 2.15.1 on PyPI](https://pypi.org/project/tensorflow/2.15.1/) — classifiers list Python 3.9, 3.10, 3.11 only.

On Windows, `pip install tensorflow==2.15.1` resolves to `tensorflow-intel==2.15.1`, which has the **same cp311/cp310/cp39** constraint.

---

## Inspected dependency files

### `pyproject.toml` (canonical)

```toml
requires-python = ">=3.10,<3.12"   # updated — was "<3.13"
dependencies = [
    "numpy==1.26.4",
    "opencv-python==4.10.0.84",
    "tensorflow==2.15.1",
    "keras==2.15.0",
    "deepface==0.0.93",
    "PyYAML>=6.0.1,<7",
]
```

### `requirements.txt`

Mirrors runtime ML stack; header already recommends Python 3.11.x.

### `requirements-dev.txt`

pytest, pytest-cov, PyYAML — **not** the failure point; they never install because TensorFlow resolution fails first.

---

## Official / de-facto Python support matrix

| Package | Version | Declared Python | **Actual wheels (install reality)** |
|---------|---------|-----------------|-------------------------------------|
| **TensorFlow** | 2.15.1 | `>=3.9` (metadata) | **3.9, 3.10, 3.11 only** |
| **Keras** | 2.15.0 | 3.8–3.11 (classifiers) | Universal `py3` wheel; **must match TF runtime** |
| **DeepFace** | 0.0.93 | `>=3.7` | Loose deps: `tensorflow>=1.9.0`, `keras>=2.2.0` |
| **NumPy** | 1.26.4 | 3.9–3.12 | Compatible with TF 2.15 (`numpy<2`) |

### DeepFace 0.0.93 notes

- Does **not** pin TensorFlow to 2.15; it accepts a wide range.
- Known pain point: **TensorFlow 2.16+ ships Keras 3**; many DeepFace versions expect Keras 2 / `tf.keras` patterns.
- Pinning `deepface==0.0.93` + `tensorflow==2.15.1` is a **deliberate stable combo** for this project.
- Upgrading Python to 3.12 **forces** TensorFlow ≥2.16, which triggers Keras 3 migration work.

---

## Option comparison

### Option A — Keep TensorFlow 2.15 stack, require Python 3.11

| Dimension | Assessment |
|-----------|------------|
| **Risk** | **Low** — pins match published wheels; stack already chosen in README |
| **Effort** | **~1 hour** — install Python 3.11, recreate venv, `pip install -e ".[dev]"` |
| **Compatibility** | Full: app, DeepFace, PAD eval, tools, pytest |
| **PAD / thresholds / metrics** | Unchanged ✓ |

**Verdict:** **Recommended for v1 release.**

Also supported: **Python 3.10** (same TF wheels). **Python 3.11** preferred (latest in the 2.15 wheel set, better perf, longer support).

---

### Option B — Upgrade stack for Python 3.12

Typical target versions:

| Package | Example upgrade |
|---------|-----------------|
| Python | 3.12.x |
| TensorFlow | 2.16.x – 2.18.x (has cp312 wheels) |
| Keras | 3.x (bundled) + often **`tf-keras`** for legacy APIs |
| DeepFace | Latest (e.g. 0.0.9x+) — **must validate** |
| NumPy | May need `>=2.0` for newer TF |

| Dimension | Assessment |
|-----------|------------|
| **Risk** | **High** — Keras 3 breaking changes, DeepFace model load paths, embedding reproducibility |
| **Effort** | **1–2 weeks** — dependency matrix, fix imports, re-run full test suite, re-validate Facenet512 embeddings & PAD eval baselines |
| **Compatibility** | Unknown until tested; **embedding vectors may differ** → re-register all users |
| **PAD / thresholds / metrics** | PAD logic unchanged but **eval baselines may need re-run** if TF affects preprocessing |

**Verdict:** **Defer to v1.1+** — not the fastest path to stable v1.

---

## Risks of staying on Python 3.12 (without Option B)

| Risk | Impact |
|------|--------|
| Cannot install pinned TensorFlow | **Blocks all ML workflows** |
| pytest / dev tools missing | **Blocks CI and quality gates** |
| Workarounds (unpinned TF, manual wheels) | Fragile, unreproducible, unsupported |
| Using system Python 3.12 globally | Other projects OK; **this repo still broken** until venv uses 3.11 |

There is **no supported way** to run `tensorflow==2.15.1` on Python 3.12.

---

## Recommended solution

**Use Option A: Python 3.11.x virtual environment.**

Project policy after this audit:

- **Supported:** Python **3.10.x**, **3.11.x**
- **Recommended:** Python **3.11.x**
- **Unsupported:** Python **3.12+** with current pins (enforced in `pyproject.toml`)

---

## Exact recovery commands (Windows)

You currently have **Python 3.12.4**. Install **Python 3.11** alongside it (do not uninstall 3.12 unless you choose to).

### 1. Install Python 3.11

Download: [python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)  
During setup: enable **“Add python.exe to PATH”** and **“py launcher”**.

Verify:

```powershell
py -0p
py -3.11 --version
```

Expected: `Python 3.11.x`

### 2. Migrate project venv

From repository root:

```powershell
cd C:\face_detection

# Deactivate if an old venv is active
deactivate

# Remove broken 3.12 venv (optional but recommended)
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue

# Create venv with Python 3.11 explicitly
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Confirm venv Python — MUST NOT be 3.12
python --version
```

Expected: `Python 3.11.x`

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

### 4. Verify environment

```powershell
python -c "import sys; print(sys.version)"
python -c "import tensorflow as tf; print('tensorflow', tf.__version__)"
python -c "import keras; print('keras', keras.__version__)"
python -c "import deepface; print('deepface ok')"
python -c "import pytest; print('pytest', pytest.__version__)"
pytest --collect-only -q
```

### 5. Run application smoke test

```powershell
python app.py
```

---

## Migration steps (macOS / Linux)

```bash
cd /path/to/face_detection
deactivate 2>/dev/null || true
rm -rf .venv

# Install python3.11 via brew/apt if needed
python3.11 -m venv .venv
source .venv/bin/activate
python --version   # expect 3.11.x

python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
pytest -m "not system" -q
```

---

## Alternative install path (requirements files)

If editable install is not needed immediately:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

---

## Tooling-only workflow (no TensorFlow)

Dataset tools (`collect_pad_dataset`, `generate_manifest`, `qc_dataset`) use OpenCV + stdlib and **can run without TensorFlow**, but **`pip install -e .` still resolves full `[project] dependencies`** from `pyproject.toml`.

For eval harness without fixing Python version: **not viable** — harness imports `src.pad_engine` and full stack.

**Practical split (advanced):** separate optional extra `[project.optional-dependencies] ml` in a future refactor — **out of scope** for this audit; use Python 3.11 for the unified install.

---

## CI recommendation

```yaml
# Example GitHub Actions
runs-on: ubuntu-latest
steps:
  - uses: actions/setup-python@v5
    with:
      python-version: "3.11"
  - run: pip install -e ".[dev]"
  - run: pytest -m "not system"
```

---

## Changes applied in this audit

| File | Change |
|------|--------|
| `pyproject.toml` | `requires-python = ">=3.10,<3.12"` — pip rejects 3.12 early with clear message |
| `requirements.txt` | Reinforced Python 3.11 guidance |
| `README.md` | Windows `py -3.11` venv instructions |
| `.python-version` | `3.11` for pyenv / compatible tools |
| `docs/python_environment_audit.md` | This document |

**Not changed:** PAD logic, recognition thresholds, evaluation metrics, TensorFlow/DeepFace version pins.

---

## Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-30 | Option A (Python 3.11) | Zero ML stack change; fastest recovery; matches existing README intent |
| 2026-05-30 | Defer Option B | Keras 3 + DeepFace + embedding reproducibility = v1.1 epic |

---

## Quick reference

| Question | Answer |
|----------|--------|
| Why did install fail? | No TF 2.15.1 wheel for Python 3.12 |
| Which Python? | **3.11.x** (3.10.x also OK) |
| Change TensorFlow pin? | **No** for v1 |
| Fastest fix? | `py -3.11 -m venv .venv` → `pip install -e ".[dev]"` |
