# Face Lock AI — Performance Audit v1

**Date:** 2026-05-30  
**Focus modules:** `src/recognize.py`, `src/pad_engine.py`, `src/ai_engine.py`

---

## Executive summary

The dominant cost is **DeepFace.represent()** (detect + align + embed) on CPU/GPU. PAD heuristics are comparatively cheap. Several **avoidable per-frame costs** exist: repeated full embedding scans, no model warm-up, and loading all pickle records once per session (acceptable) but parsing every user on each match.

**Estimated live loop budget (30 FPS camera, frame_skip=3, cooldown=2s):**

| Stage | Frequency | Relative cost |
|-------|-----------|---------------|
| `cap.read()` + display | Every frame | Low |
| `pad_session.add_frame` | Every frame | Low–medium (buffer copy) |
| `pad_session.evaluate_final` | Every ~3rd frame + cooldown | Medium |
| `generate_embedding` (DeepFace) | After PAD LIVE | **Very high** |
| `find_best_match` | After embedding | Low–medium (users × embeddings) |

---

## Findings

### PERF-001 — DeepFace model cold start

| Field | Value |
|-------|-------|
| **Severity** | High (UX) |
| **Location** | `src/ai_engine.py` `represent_face`, `load_face_model` (unused) |
| **Issue** | First `DeepFace.represent()` loads Facenet512 + detector; multi-second delay |
| **Recommendation** | Call `load_face_model()` once in `app.py` or start of `recognize_live` |
| **Priority** | P1 |

---

### PERF-002 — Full inference on every eligible frame

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Location** | `src/recognize.py` L416–448 |
| **Issue** | After cooldown, runs PAD final + embedding even if no face in frame |
| **Recommendation** | Cheap face presence check (Haar or detector pre-pass) before DeepFace |
| **Priority** | P2 |

---

### PERF-003 — Linear scan all embeddings every match

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Location** | `src/ai_engine.py` `find_best_match` L503–531 |
| **Issue** | O(users × embeddings_per_user) cosine compares per unlock attempt |
| **Recommendation** | Pre-stack embeddings to NumPy matrix; batch dot product; index map for user |
| **Priority** | P2 (matters at 50+ users) |

---

### PERF-004 — PAD frame buffer growth

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Location** | `src/pad_engine.py` `PadSession.add_frame`, `_ClipBuilder` |
| **Issue** | Stores frame copies for clip window; at 30 FPS × N seconds, memory grows |
| **Recommendation** | Cap buffer size; downsample for passive scorer; store grayscale thumb |
| **Priority** | P2 |

---

### PERF-005 — `frame.copy()` on every PAD frame

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Location** | `src/pad_engine.py` `build_pad_frame`, `recognize.py` loop |
| **Issue** | Full-resolution BGR copy per frame |
| **Recommendation** | Copy only when frame enters clip; pass reference for display path |
| **Priority** | P3 |

---

### PERF-006 — Repeated disk I/O on registration only

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Location** | `src/register.py`, `src/capture.py` |
| **Issue** | Each saved image written to disk then re-read for embedding |
| **Recommendation** | Embed from in-memory frame before optional save |
| **Priority** | P3 |

---

### PERF-007 — Pickle load at recognition start

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Location** | `src/recognize.py` `load_trusted_records` |
| **Issue** | Loads all `.pkl` once per session — acceptable |
| **Recommendation** | Cache in memory with mtime invalidation if files change mid-session |
| **Priority** | P4 |

---

### PERF-008 — Heuristic PAD per evaluate_final

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Location** | `src/pad_engine.py` `HeuristicPassiveScorer` |
| **Issue** | Laplacian, brightness, motion on clip — ~milliseconds at 640×480 |
| **Recommendation** | Already reasonable; profile if clip > 16 frames |
| **Priority** | P4 |

---

### PERF-009 — Integration test / logging overhead

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Location** | `src/logging_config.py`, `recognize.py` log_event |
| **Issue** | Structured logging on hot path |
| **Recommendation** | DEBUG-only for per-frame events |
| **Priority** | P4 |

---

### PERF-010 — TensorFlow thread contention

| Field | Value |
|-------|-------|
| **Severity** | Medium (environment) |
| **Location** | TensorFlow via DeepFace |
| **Issue** | Default TF thread pool may starve OpenCV UI thread |
| **Recommendation** | Set `TF_NUM_INTEROP_THREADS=1` for desktop; document in CONFIGURATION |
| **Priority** | P2 |

---

## Module-specific notes

### `recognize.py`

```text
while True:
  read_frame()           # every iteration
  pad_session.add_frame  # every iteration
  if should_run_inference:
    evaluate_final()     # PAD
    generate_embedding() # DeepFace — BOTTLENECK
    find_best_match()    # all users
  draw + imshow
```

**Bottleneck:** `generate_embedding` after PAD passes.  
**Frame skip + cooldown** correctly throttle but don't reduce per-attempt cost.

### `ai_engine.py`

- Lazy DeepFace import ✓  
- `represent_face` calls full pipeline each time  
- No embedding cache for live session  
- `load_face_model` exists but **never called**

### `pad_engine.py`

- Heuristic scorer: OpenCV ops on small clip — not primary bottleneck  
- Active challenge adds UX latency by design  
- Mock scorer for tests — zero cost

---

## Quick wins (no PAD/threshold changes)

1. Warm up DeepFace model at startup  
2. Batch cosine similarity with NumPy  
3. Reduce PAD clip copies / resolution for passive score  
4. Face presence gate before DeepFace  

---

## Profiling commands (operator)

```powershell
pip install py-spy
py-spy record -o profile.svg -- python app.py
# Select live recognition, reproduce slow path
```

```python
# Temporary timing in recognize_identity (dev only)
import time
t0 = time.perf_counter()
live_embedding = generate_embedding(...)
LOGGER.debug("embed_ms=%.1f", (time.perf_counter() - t0) * 1000)
```

---

## Performance score

| Category | Score /10 |
|----------|-----------|
| Latency (first unlock) | 4 |
| Steady-state unlock | 5 |
| Memory | 6 |
| Scalability (users) | 5 |
| **Overall perf readiness** | **5/10** |

Acceptable for **single-user demo**; not tuned for multi-user or sub-second SLA.
