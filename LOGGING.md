# Face Lock AI Logging Architecture

## 1. Explanation

Observability means being able to understand what the application did, when it
did it, and why something failed. For Face Lock AI, observability is especially
important because the system handles camera input, biometric-style embeddings,
access decisions, and user registration workflows.

`print()` is useful for simple terminal feedback, but it is not enough for a
maintainable backend system. A print statement usually disappears when the
terminal closes, has no standard severity level, does not rotate files, and is
hard to search later. Python logging gives timestamps, log levels, file output,
console output, traceback support, module names, and configurable verbosity.

Centralized logging matters because every module should follow one policy. If
each file creates its own handlers, logs become duplicated, inconsistent, and
hard to trust. In this project, `src/logging_config.py` owns setup and modules
should only request loggers.

AI and security systems need audit trails because decisions must be explainable.
Logs should show when registration succeeded, why recognition was denied, when
lighting was too low, when embedding generation failed, and when unexpected
exceptions happened. Logs should not store raw face images or full embeddings.

Log levels:

- `DEBUG`: Detailed developer diagnostics, such as config setup or internal flow.
- `INFO`: Normal successful events, such as registration completed.
- `WARNING`: Something recoverable or suspicious, such as low lighting.
- `ERROR`: A failed operation, such as embedding generation failure.
- `CRITICAL`: Severe failure where the application may not continue safely.

Rotating logs are important because long-running apps can produce large files.
`RotatingFileHandler` caps the active log file size and keeps a configured number
of backups. This prevents `logs/` from growing forever.

Logs should never mix with source code folders. `src/` must stay deterministic
and reviewable. Logs are runtime artifacts, so they belong in `logs/` and should
stay out of Git except for `logs/.gitkeep`.

## 2. Logging Module

The centralized logging module is:

```text
src/logging_config.py
```

It provides:

- `setup_logging()`
- `get_logger(module_name)`
- `log_event(logger, event_name, **fields)`
- `log_exception(logger, event_name, error, **fields)`
- `log_registration_event(event_name, **fields)`
- `log_recognition_event(event_name, **fields)`

## 3. Integration Examples

Basic module logger:

```python
from src.logging_config import get_logger

logger = get_logger(__name__)
logger.info("event=module_loaded")
```

Successful registration:

```python
from src.logging_config import log_registration_event

log_registration_event(
    "registration_completed",
    user_name=final_user_name,
    image_count=len(image_paths),
    embedding_count=len(embeddings),
)
```

Failed recognition:

```python
from src.logging_config import log_recognition_event

log_recognition_event(
    "recognition_denied",
    level=logging.WARNING,
    reason="distance_above_threshold",
    closest_user=closest_user,
    cosine_distance=distance,
    threshold=threshold,
)
```

Low lighting warning:

```python
log_recognition_event(
    "low_lighting",
    level=logging.WARNING,
    brightness=estimate_brightness(frame),
    threshold=LOW_LIGHT_THRESHOLD,
)
```

Embedding generation failure:

```python
from src.logging_config import get_logger, log_exception

logger = get_logger(__name__)

try:
    embedding = generate_embedding(image_input=image_path)
except Exception as error:
    log_exception(
        logger,
        "embedding_generation_failed",
        error,
        image_name=image_path.name,
    )
```

Unexpected exception:

```python
try:
    run_app()
except Exception as error:
    log_exception(logger, "application_unhandled_exception", error)
    raise
```

## 4. Example Log Outputs

```text
2026-05-27 23:10:11 | level=INFO | logger=face_lock.registration | module=logging_config | function=log_event | message=event=registration_completed user_name=charan image_count=10 embedding_count=10
2026-05-27 23:12:02 | level=WARNING | logger=face_lock.recognition | module=logging_config | function=log_event | message=event=recognition_denied reason=distance_above_threshold closest_user=charan cosine_distance=0.421 threshold=0.3
2026-05-27 23:12:15 | level=WARNING | logger=face_lock.recognition | module=logging_config | function=log_event | message=event=low_lighting brightness=31.7 threshold=45.0
2026-05-27 23:13:03 | level=ERROR | logger=face_lock.register | module=logging_config | function=log_exception | message=event=embedding_generation_failed error_type=RuntimeError error='No usable face was detected in the image.' image_name=charan_001.jpg
2026-05-27 23:14:20 | level=ERROR | logger=face_lock.app | module=logging_config | function=log_exception | message=event=application_unhandled_exception error_type=RuntimeError error='Could not open webcam with index 0.'
```

## 5. Migration Strategy

1. Keep `src/logging_config.py` as the only place that creates handlers.
2. Call `setup_logging()` once near application startup.
3. Add `logger = get_logger(__name__)` to one module at a time.
4. Keep `print_status()` for user-facing terminal messages during the transition.
5. Add audit logs around registration completion, recognition denial, low light,
   embedding failures, and unexpected exceptions.
6. Avoid logging raw images, face embeddings, secrets, or large binary data.
7. Once logging is mature, decide whether `print_status()` should also write to
   the logger or remain purely terminal output.

## 6. Production Observability Recommendations

- Use `INFO` in production and `DEBUG` during local diagnosis.
- Send logs to a central log system in production.
- Add request/session IDs when the app grows beyond a terminal workflow.
- Add metrics later for recognition latency, denial count, camera failures, and
  registration success rate.
- Add alerting for repeated camera failures, repeated recognition errors, and
  unexpected exceptions.
- Keep biometric data out of logs.
- Review log retention policy before production use.
