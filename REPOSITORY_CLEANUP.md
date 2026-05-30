# Face Lock AI Repository Cleanup

## 1. Hygiene Audit Explanation

This audit focuses only on repository hygiene and maintainability. It does not add
AI features, change recognition behavior, or rewrite the application.

The project should separate source code from runtime output. Source files are the
files needed to understand, run, test, and maintain the application. Runtime files
are produced by local execution: captured face images, embeddings, logs, caches,
temporary frames, downloaded models, and local virtual environments.

Generated files are dangerous in this project because they can contain biometric
data, machine-specific paths, stale cache state, large binary blobs, and outputs
that cannot be reviewed meaningfully in pull requests. In a face-lock system,
captured images and encodings are especially sensitive because they identify real
people and should not be copied into source control by accident.

The `src/` directory should stay clean because it is the maintainable application
surface. Reviewers should be able to open `src/` and see only Python source files
and package metadata. Images such as `src/capture.png` and `src/temp.jpg` make the
source tree ambiguous: it becomes unclear whether a file is required by the app,
was produced by a test run, or is simply leftover local output.

The `data/` and `encodings/` folders should be ignored by Git because they are
runtime state. `data/` holds captured face images. `encodings/` holds derived face
embeddings, commonly as pickle or NumPy artifacts. Both are user-specific,
sensitive, and likely to change every time the app is used.

Cache and temp files are harmful because they create noisy diffs, hide real
changes, waste repository space, and can accidentally preserve outdated behavior.
Python bytecode in `__pycache__/` is created automatically for the local Python
version and should always be regenerated locally instead of committed.

Current hygiene findings:

- `.venv/` exists inside the project and must remain ignored.
- Root `__pycache__/` and `src/__pycache__/` exist and should be removed locally.
- `src/capture.png` and `src/temp.jpg` are generated images inside source code.
- `face_detection/data/charan/*.png` is a nested runtime data folder.
- `src/utility.py` is a zero-byte file and appears unused.
- `modules/` appears empty and should be removed unless it has a planned purpose.
- `data/`, `encodings/`, `logs/`, and `models/` currently use `.gitkeep`, which is
  appropriate if the folders are needed at runtime.

## 2. Final Recommended Folder Structure

```text
face-lock/
|-- .gitignore
|-- CONFIGURATION.md
|-- LOGGING.md
|-- README.md
|-- REPOSITORY_CLEANUP.md
|-- app.py
|-- requirements.txt
|-- data/
|   `-- .gitkeep
|-- encodings/
|   `-- .gitkeep
|-- logs/
|   `-- .gitkeep
|-- models/
|   `-- .gitkeep
`-- src/
    |-- ai_engine.py
    |-- capture.py
    |-- config.py
    |-- display.py
    |-- logging_config.py
    |-- recognize.py
    |-- register.py
    `-- utils.py
```

Recommended future additions when the project grows:

```text
tests/
docs/
configs/
scripts/
```

Use those only when they have real content. Empty or speculative folders should
not be added unless they clarify runtime expectations, such as `.gitkeep` files in
runtime directories.

## 3. Cleanup Action List

- Remove generated cache folders:
  - `__pycache__/`
  - `src/__pycache__/`
- Remove generated source-tree media:
  - `src/capture.png`
  - `src/temp.jpg`
- Move or delete nested runtime data:
  - `face_detection/data/charan/*.png`
  - If the images are still needed, move them under `data/charan/`.
  - If they are old samples, delete them from the repository workspace.
- Remove empty or unused development artifacts:
  - `modules/`, if it remains empty.
  - `src/utility.py`, because it is empty and no current source imports it.
- Keep these runtime folders but only commit their `.gitkeep` files:
  - `data/`
  - `encodings/`
  - `logs/`
  - `models/`
- Keep `.venv/` local and recreate it from `requirements.txt` when needed.
- Do not commit captured faces, embeddings, local logs, model binaries, temp
  frames, virtual environments, or bytecode caches.

## 4. Gitignore Policy

The repository `.gitignore` should block:

- Python bytecode and test caches.
- Virtual environments.
- Build and package outputs.
- Local secrets and environment files.
- Runtime biometric data.
- Face embeddings and ML binaries.
- Captured/generated media.
- Logs, temporary files, and local databases.
- OS and editor metadata.

The generated `.gitignore` in this repository follows that policy and keeps
`.gitkeep` files trackable for empty runtime folders.

## 5. Repository Best Practices

Naming conventions:

- Use lowercase snake_case for Python modules: `ai_engine.py`, `register.py`.
- Use descriptive singular names for modules that own one responsibility.
- Avoid duplicate names with similar meaning, such as `utility.py` and `utils.py`.
- Use lowercase folder names: `data`, `encodings`, `logs`, `models`, `src`.

Folder conventions:

- `src/` contains source code only.
- `data/` contains local captured images only.
- `encodings/` contains local generated embeddings only.
- `logs/` contains runtime logs only.
- `models/` contains downloaded or generated model files only.
- `docs/` contains human documentation.
- `tests/` contains automated tests.
- `scripts/` contains one-off maintenance commands.
- `configs/` contains non-secret configuration templates.

Runtime versus source separation:

- Source files should be deterministic and reviewable.
- Runtime files should be disposable and reproducible.
- Sensitive files should stay outside Git history.
- Large binary artifacts should be stored in artifact storage, model registries, or
  release assets, not in normal commits.

## 6. Future Maintenance Recommendations

- Run a cleanup check before commits:
  - `git status --short`
  - `git check-ignore -v data/some_file.jpg`
  - `rg --files -uu | rg "__pycache__|\\.pyc$|temp|capture|\\.pkl$|\\.npy$|\\.png$|\\.jpg$"`
- Add tests only after the current cleanup phase is complete.
- Add a small `docs/` page later for data privacy and local setup.
- Add pre-commit tooling later if the project becomes collaborative.
- Avoid committing personal face samples, even for demos.
- Prefer sanitized synthetic examples for documentation.
- Treat embeddings as sensitive biometric derivatives, not harmless cache files.
- Periodically review `.gitignore` after adding new tools or frameworks.

Production repositories do this differently by enforcing the separation with CI,
pre-commit hooks, secret scanning, artifact stores, test fixtures, documented data
retention rules, and reproducible setup scripts. They do not rely on developers to
remember which generated files are safe; the repository policy blocks unsafe files
by default.
