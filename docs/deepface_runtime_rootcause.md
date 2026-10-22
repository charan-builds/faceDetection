# DeepFace Runtime Root-Cause Investigation

**Date:** 2026-06-01  
**Environment:** Windows, Python 3.11.9, TensorFlow 2.15.1, Keras 2.15.0, DeepFace 0.0.93  
**Symptom:** `from deepface import DeepFace` → `ModuleNotFoundError: No module named 'tensorflow.keras'`

---

## Executive summary

| Item | Finding |
|------|---------|
| **Root cause** | DeepFace 0.0.93 sets `TF_USE_LEGACY_KERAS=1` **unconditionally** before importing TensorFlow. On TF 2.15 this **removes** the `tensorflow.keras` namespace, but DeepFace model code still imports `tensorflow.keras.*` when `get_tf_major_version() == 2`. |
| **Classification** | **DeepFace 0.0.93 bug** + **TensorFlow/Keras interaction** (not Windows-specific, not a pip conflict) |
| **Minimum-risk fix** | Patch `DeepFace.py` to enable legacy Keras **only for TF ≥ 2.16**; optional fallback in `preprocessing.py` |
| **TensorFlow upgrade** | **Not required** |

---

## What works vs what fails

### Works (direct imports, no `DeepFace.py` entry)

```python
import tensorflow
import tensorflow.keras
import tensorflow.keras.preprocessing.image
from deepface.modules.preprocessing import ...
from deepface.modules.representation import ...
from deepface.modules.verification import ...
```

These succeed because **`DeepFace.py` is not executed**, so `TF_USE_LEGACY_KERAS` is never set.

### Fails

```python
from deepface import DeepFace
```

This loads `deepface/DeepFace.py`, which sets legacy mode, then imports modules that expect `tensorflow.keras`.

---

## Import chain (exact failure path)

```text
deepface/DeepFace.py:8
  os.environ["TF_USE_LEGACY_KERAS"] = "1"    ← ROOT TRIGGER

deepface/DeepFace.py:15
  import tensorflow as tf                     ← TF loads with legacy Keras

deepface/DeepFace.py:20
  from deepface.modules import modeling, ...

deepface/modules/modeling.py:5
  from deepface.models.facial_recognition import VGGFace, ...

deepface/models/facial_recognition/VGGFace.py:6
  from deepface.modules import verification

deepface/modules/verification.py:9
  from deepface.modules import representation, detection, modeling

deepface/modules/representation.py:9
  from deepface.modules import modeling, detection, preprocessing

deepface/modules/preprocessing.py:16          ← FIRST FAILURE (before patch)
  from tensorflow.keras.preprocessing import image
  ModuleNotFoundError: No module named 'tensorflow.keras'

(after preprocessing patch only, next failure:)

deepface/models/FacialRecognition.py:8
  from tensorflow.keras.models import Model
  ModuleNotFoundError: No module named 'tensorflow.keras'
```

**Failing module (primary):** `deepface/DeepFace.py` line 8 (environment side effect)  
**Failing import (symptom):** `deepface/modules/preprocessing.py` and ~15 model files using `tensorflow.keras` when TF major version is 2

---

## Proof: `TF_USE_LEGACY_KERAS` breaks `tensorflow.keras` on TF 2.15

```powershell
.\.venv\Scripts\Activate.ps1

# Without legacy flag — works
python -c "import tensorflow as tf; import tensorflow.keras; print('ok')"

# With DeepFace's flag — fails
python -c "import os; os.environ['TF_USE_LEGACY_KERAS']='1'; import tensorflow as tf; import tensorflow.keras"
# ModuleNotFoundError: No module named 'tensorflow.keras'

# Standalone keras still works under legacy mode
python -c "import os; os.environ['TF_USE_LEGACY_KERAS']='1'; import tensorflow; from keras.preprocessing import image; print('ok')"
```

DeepFace's own version branching uses:

```python
if tf_major_version == 1:
    from keras...
elif tf_major_version == 2:
    from tensorflow.keras...   # ← wrong branch when legacy mode is on
```

On TF 2.15 with `TF_USE_LEGACY_KERAS=1`, major version is still **2**, but Keras is exposed as standalone **`keras`**, not **`tensorflow.keras`**.

---

## Issue classification

| Hypothesis | Verdict |
|------------|---------|
| DeepFace 0.0.93 bug | **Yes** — unconditional `TF_USE_LEGACY_KERAS` intended for TF 2.16+/Keras 3, harmful on TF 2.15 |
| TensorFlow/Keras compatibility | **Yes** — legacy flag changes Keras namespace |
| Dependency conflict (wrong keras/tf versions) | **No** — pins are correct; imports work without DeepFace entry |
| Windows-specific | **No** — reproduces on any OS with same stack |

---

## Minimum-risk fix (recommended)

### Patch 1 — `deepface/DeepFace.py` (required)

Replace unconditional:

```python
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

With conditional enable **only for TensorFlow ≥ 2.16**:

```python
def _enable_legacy_keras_if_needed() -> None:
    try:
        import importlib.metadata
        version = importlib.metadata.version("tensorflow")
        major_str, minor_str, *_ = version.split(".")
        minor = int(minor_str.split("+")[0].split("-")[0])
        if int(major_str) == 2 and minor >= 16:
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
    except importlib.metadata.PackageNotFoundError:
        pass

_enable_legacy_keras_if_needed()
```

On **TF 2.15.1**, this does **not** set the flag → `tensorflow.keras` remains available → all DeepFace model imports succeed.

### Patch 2 — `deepface/modules/preprocessing.py` (optional defense-in-depth)

Add fallback when `tensorflow.keras` is missing:

```python
elif tf_major_version == 2:
    try:
        from tensorflow.keras.preprocessing import image
    except ModuleNotFoundError:
        from keras.preprocessing import image
```

---

## Exact commands

### Apply patch (automated)

From repository root with venv activated:

```powershell
cd C:\face_detection
.\.venv\Scripts\Activate.ps1
python scripts/patch_deepface_tf215.py
```

Run **after every** `pip install -e ".[dev]"` or `pip install --upgrade deepface`.

### Verify fix

```powershell
python -c "from deepface import DeepFace; print('DeepFace OK', DeepFace.__name__)"
python -c "from src.ai_engine import _get_deepface; print('Face Lock ai_engine OK')"
python -c "import tensorflow as tf; print('TF', tf.__version__); import tensorflow.keras; print('tf.keras OK')"
```

Expected: all three print success messages (TensorFlow may log oneDNN warnings — normal).

### Full app smoke (optional)

```powershell
python app.py
# Menu should load; registration/recognition require webcam
```

---

## What we did NOT change

- TensorFlow 2.15.1 pin — **unchanged**
- Keras 2.15.0 pin — **unchanged**
- DeepFace 0.0.93 pin — **unchanged**
- Face Lock PAD logic, thresholds, eval metrics — **unchanged**

---

## Alternative fixes (not chosen)

| Option | Why not |
|--------|---------|
| Upgrade TensorFlow to 2.16+ for Python 3.12 | High regression risk; Keras 3 migration; embedding drift |
| Upgrade DeepFace to latest | Requires compatibility matrix + full retest; may pull Keras 3 paths |
| Pin `TF_USE_LEGACY_KERAS=0` in app before import | DeepFace.py sets flag first when using `from deepface import DeepFace` |
| Vendor entire DeepFace | High maintenance |

---

## Files in this repository

| File | Purpose |
|------|---------|
| `scripts/patch_deepface_tf215.py` | Idempotent post-install patch |
| `docs/deepface_runtime_rootcause.md` | This document |

---

## Maintenance note

The patch modifies files under `.venv/Lib/site-packages/deepface/`. Re-run the script whenever the venv is recreated or DeepFace is reinstalled. Consider adding to your setup docs:

```powershell
pip install -e ".[dev]"
python scripts/patch_deepface_tf215.py
```

---

## Verification log (2026-06-01)

```text
python -c "from deepface import DeepFace"          → OK
python -c "from src.ai_engine import _get_deepface" → OK
```

After Patch 1 (+ optional Patch 2) on Windows / Python 3.11.9 / TF 2.15.1 / DeepFace 0.0.93.
