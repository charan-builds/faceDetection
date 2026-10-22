#!/usr/bin/env python3
"""
Apply DeepFace 0.0.93 compatibility patches for TensorFlow 2.15.x.

DeepFace unconditionally sets TF_USE_LEGACY_KERAS=1, which removes the
tensorflow.keras namespace on TF 2.15 while model code still imports
tensorflow.keras — causing ModuleNotFoundError.

Run after every ``pip install`` that installs or upgrades deepface:

    python scripts/patch_deepface_tf215.py

Idempotent: safe to run multiple times.
"""

from __future__ import annotations

import sys
from pathlib import Path


DEEPFACE_PY_LEGACY_BLOCK = '''# this has to be set before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"
'''

DEEPFACE_PY_PATCHED_BLOCK = '''# this has to be set before importing tensorflow — only TF 2.16+ (Keras 3) needs legacy mode.
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
'''

PREPROCESSING_OLD = """elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image
"""

PREPROCESSING_PATCHED = """elif tf_major_version == 2:
    try:
        from tensorflow.keras.preprocessing import image
    except ModuleNotFoundError:
        # TF 2.16+ with TF_USE_LEGACY_KERAS=1, or TF 2.15 after legacy mode is enabled.
        from keras.preprocessing import image
"""


def _site_packages() -> Path:
    for entry in sys.path:
        candidate = Path(entry) / "deepface" / "DeepFace.py"
        if candidate.is_file():
            return Path(entry)
    raise FileNotFoundError(
        "Could not locate deepface in the active Python environment. "
        "Activate the project venv and ensure deepface is installed."
    )


def _patch_file(path: Path, old: str, new: str, label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  {label}: already patched")
        return False
    if old not in text:
        # Optional patch: skip if file layout changed or already manually patched.
        if label == "preprocessing.py" and "except ModuleNotFoundError" in text:
            print(f"  {label}: already patched (manual or prior run)")
            return False
        raise RuntimeError(
            f"{label}: expected content not found in {path}. "
            "DeepFace version may differ from 0.0.93 — review manually."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  {label}: patched")
    return True


def main() -> int:
    site = _site_packages()
    deepface_py = site / "deepface" / "DeepFace.py"
    preprocessing_py = site / "deepface" / "modules" / "preprocessing.py"

    print(f"Patching DeepFace under: {site}")
    changed = False
    changed |= _patch_file(deepface_py, DEEPFACE_PY_LEGACY_BLOCK, DEEPFACE_PY_PATCHED_BLOCK, "DeepFace.py")
    changed |= _patch_file(preprocessing_py, PREPROCESSING_OLD, PREPROCESSING_PATCHED, "preprocessing.py")

    if changed:
        print("Done. Verify with: python -c \"from deepface import DeepFace; print('ok')\"")
    else:
        print("No changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
