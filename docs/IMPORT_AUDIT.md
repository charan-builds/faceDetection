# Face Lock AI — Import & Dependency Audit

**Date:** 2026-05-30  
**Repository root:** `C:\face_detection`

## Executive summary

| Issue | Root cause | Fix applied |
|-------|------------|-------------|
| `ModuleNotFoundError: No module named 'src'` | Running `python tools/*.py` puts `tools/` on `sys.path[0]`, not the repo root; `src/` was not an installable package | `pyproject.toml` editable install, `src/__init__.py`, `tools/project_paths.py` |
| `from tools.*` fails when running scripts directly | Same `sys.path[0]` issue | `pip install -e .` or `python -m tools.<module>` |
| pytest worked in many setups | `pytest.ini` / `pyproject.toml` set `pythonpath = .` | Consolidated in `pyproject.toml` |

## Project layout

```text
face_detection/                 ← repository root (must be on PYTHONPATH or installed)
├── pyproject.toml              ← packaging + pytest config (NEW)
├── app.py                      ← imports src.* (root on path when run as script)
├── src/                        ← application package (NEW: __init__.py)
├── tools/                      ← dataset CLI utilities (NEW: __init__.py)
├── eval/                       ← evaluation framework (local package name; see note)
├── tests/
└── requirements.txt
```

### Note on `eval` package name

The top-level folder `eval/` is a **local Python package** (`import eval.harness`). It shadows the built-in `eval()` name only if imported as a package — this is pre-existing by design. Do not add a top-level module named `eval.py`.

## Why `python tools/collect_pad_dataset.py` failed

Python inserts the **script directory** at `sys.path[0]`:

```text
python tools/collect_pad_dataset.py
→ sys.path[0] == .../face_detection/tools
```

Imports attempted:

```python
from src.config import resolve_project_root  # looks for tools/src/ — missing
from tools.collect_pad_dataset import ...     # looks for tools/tools/ — missing
```

`pytest` and `python app.py` worked because `sys.path[0]` is the **repository root**.

## Fixes applied

### 1. `src/__init__.py`

Marks `src` as a regular package for setuptools and static analyzers.

### 2. `tools/__init__.py` + `tools/project_paths.py`

`resolve_project_root()` lives in tools and does **not** import `src`:

```python
Path(__file__).resolve().parent.parent  # repo root
```

All dataset tools use `tools.project_paths` instead of `src.config` for path resolution only.

### 3. `pyproject.toml`

- Declares dependencies from `requirements.txt` (+ PyYAML for harness/manifest).
- `pip install -e .` registers `src`, `tools`, and `eval` on the environment path.
- Console entry points: `collect-pad-dataset`, `generate-pad-manifest`, `qc-pad-dataset`, etc.

### 4. Pytest configuration

Moved to `[tool.pytest.ini_options]` with `pythonpath = ["."]`.

### 5. `tools/_repo_path.py` (script bootstrap)

When `python tools/<script>.py` runs, Python sets `sys.path[0]` to `tools/`, not the repo root.
Each dataset tool loads `_repo_path.py` with `importlib.util` **before** `from tools.*` so the
repository root is prepended once. This is isolated to tooling scripts; application code still
relies on root scripts, `python -m`, or `pip install -e .`.

## Supported execution methods

| Command | Requires `pip install -e .`? | Notes |
|---------|------------------------------|-------|
| `pip install -e .` | — | **Recommended** once per venv |
| `python app.py` | No* | *Root script; repo root on path |
| `python -m tools.collect_pad_dataset` | No* | *Run from repo root |
| `python tools/collect_pad_dataset.py` | No* | *`tools/_repo_path.py` bootstrap via `importlib` |
| `collect-pad-dataset` (console script) | Yes | After editable install |
| `pytest` | No* | *With `pythonpath = .` in pyproject |
| `python -m eval.harness.run_pad_eval` | No* | *Imports `src.pad_engine` — needs root path or install |

## Dependency alignment

| File | Role |
|------|------|
| `requirements.txt` | Runtime ML stack (numpy, opencv, tensorflow, deepface) |
| `requirements-dev.txt` | pytest, pytest-cov (also in `[project.optional-dependencies].dev`) |
| `pyproject.toml` | Canonical dependency list for `pip install -e .` |

PyYAML moved into main `pyproject.toml` dependencies (required by manifest + harness).

## Import map (by area)

| Module | Imports `src`? | Imports `tools`? | Imports `eval`? |
|--------|----------------|------------------|-----------------|
| `app.py` | Yes | No | No |
| `tools/*` | No (paths only) | Yes (internal) | No |
| `eval/harness/*` | Yes (`pad_engine`, `config`) | No | Yes (internal) |
| `eval/reporting/*` | No | `project_paths` only | Yes (internal) |
| `tests/*` | Yes | Yes | Yes |

## Verification checklist

From repository root after `pip install -e ".[dev]"`:

```powershell
python -c "import src; import tools; import eval.harness; print('imports ok')"
python tools/collect_pad_dataset.py --help
python tools/generate_manifest.py --help
python tools/qc_dataset.py --help
python -c "from app import main; print('app ok')"
pytest tests/tools/test_collect_pad_dataset.py -q
```

## Remaining limitations

- **Heavy ML deps:** `app.py` and harness require TensorFlow/DeepFace installed.
- **Direct script path:** `python tools/<name>.py` works via `_repo_path` bootstrap; `python -m tools.<name>` and console scripts remain preferred for consistency.
- **No `src` re-export in tools:** Application settings remain in `src.config`; only path helpers were decoupled.
