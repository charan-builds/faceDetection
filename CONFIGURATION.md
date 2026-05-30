# Face Lock AI Configuration Architecture

## 1. Explanation

Configuration management means keeping tunable values in one predictable place
instead of scattering them across feature code. In this project, settings such as
the DeepFace model name, detector backend, cosine threshold, frame skip, cooldown,
camera index, registration image count, storage folders, file extensions, and
logging level now belong to `src/config.py`.

Hardcoded values are risky because they make behavior difficult to audit. If the
threshold is repeated in multiple files, one file can be updated while another
keeps using the old value. In AI systems this is especially dangerous because
small changes to thresholds, detector backends, and frame timing can change user
experience, performance, and access decisions.

Centralized config improves maintainability by creating one source of truth. A
maintainer can inspect `src/config.py` and understand the operational defaults
without reading every module. Feature modules can focus on behavior, while config
owns the values that tune that behavior.

Source code is the logic of the system: how to capture frames, generate
embeddings, compare vectors, and save records. Configuration is the set of values
that tune that logic: which model to use, where runtime files live, how many
images to capture, and how strict the recognition threshold should be.

Production AI systems usually separate configuration even more strongly. They use
typed config objects, environment-specific config files, environment variables,
secret managers, model registries, CI validation, and deployment-time overrides.
The current design is intentionally lightweight, but it follows the same idea:
group settings by responsibility and keep them independent from application code.

## 2. Config Module

The central configuration module is:

```text
src/config.py
```

It uses frozen dataclasses so settings are grouped, typed, readable, and not
accidentally mutated at runtime. It also uses `pathlib.Path` to resolve project
paths from the file location instead of relying on the current working directory.

## 3. Recommended Config Categories

- `PathSettings`: project root, `data/`, `encodings/`, `logs/`, `models/`.
- `AISettings`: model name, detector backend, distance metric, cosine threshold,
  detection strictness, face alignment.
- `RecognitionSettings`: recognition window name, frame skip, cooldown, brightness
  threshold.
- `DisplaySettings`: OpenCV overlay colors, text positions, and text scale
  settings.
- `WebcamSettings`: camera index, capture window name, save key, quit key.
- `RegistrationSettings`: registration window name, image count, minimum valid
  embeddings.
- `StorageSettings`: default image extension, embedding extension, supported image
  extensions.
- `LoggingSettings`: logger name, log levels, rotating file settings, formatter,
  date format, and encoding.

## 4. Integration Strategy

The first integration step keeps backward-compatible constants in existing
modules, but points them to `src/config.py`. For example, `DEFAULT_MODEL_NAME`
still exists in `src/ai_engine.py`, but its value now comes from
`AI_SETTINGS.model_name`.

This avoids a risky broad rewrite. Existing code can keep importing familiar
names while the project gradually migrates toward direct config imports:

```python
from src.config import AI_SETTINGS, RECOGNITION_SETTINGS
```

Use direct config imports for new code. Keep old module-level constants only as
compatibility aliases until the project is ready for a full cleanup pass.

## 5. Migration Plan

1. Keep `src/config.py` as the source of truth.
2. Replace new hardcoded settings with imports from `src.config`.
3. Gradually update existing modules to accept settings objects or explicit
   values from config.
4. Remove compatibility aliases such as `DEFAULT_MODEL_NAME` only after all
   imports have migrated.
5. Add validation later for thresholds, frame skip values, file extensions, and
   camera index.
6. Add environment-specific overrides only when deployment needs them.

## 6. Future Scalability Recommendations

- Add config validation before application startup.
- Add `.env` support only for machine-specific or deployment-specific values.
- Keep secrets out of config files and Git.
- Store model binaries in `models/` or a model registry, not in source code.
- Add a production logging setup that reads `LoggingSettings.level`.
- Add test fixtures that override config cleanly without touching real biometric
  runtime folders.
- Consider a separate external config file later, such as `config/local.toml`, but
  keep `src/config.py` as the typed schema and defaults.
