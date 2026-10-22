# Face Lock Threat Model

## Threat Taxonomy

### 1. Presentation Attacks (Spoofing)
**Threat:** An attacker presents a photo, screen replay, or 3D mask of the owner to the camera.
- **Likelihood:** High. Photos and videos of most owners are easy to acquire.
- **Impact:** Complete system compromise (False Accept).
- **Current Defense:** `pad_engine.py` executes a heuristic passive check for texture (Laplacian), color distribution, and micro-motion. Borderline results force an active head-turn challenge.
- **Known Weakness:** The heuristic is not a trained anti-spoofing ML model. High-quality tablets or 3D masks might defeat the passive check.
- **Next Mitigation:** Integrate a dedicated ONNX-based anti-spoofing model (e.g., MiniFASNet) for robust passive checks.

### 2. Impostor Access (Zero-Effort Attack)
**Threat:** A non-owner sits in front of the laptop.
- **Likelihood:** High.
- **Impact:** Complete system compromise (False Accept).
- **Current Defense:** 1:1 facial embedding comparison (FaceNet512) bounded by a strict `0.30` cosine distance threshold.
- **Known Weakness:** The threshold has not yet been empirically validated for false acceptance on a large population. Look-alikes or family members may score below the threshold.
- **Next Mitigation:** Collect a robust impostor dataset (Stage B/C) to measure FAR and mathematically calibrate the threshold for strict 1:1 laptop security.

### 3. Template Substitution (Local Attack)
**Threat:** An attacker with filesystem access replaces `encodings/owner.pkl` with their own face embedding.
- **Likelihood:** Medium. Requires local filesystem access.
- **Impact:** The attacker can unlock the laptop using their own face.
- **Current Defense:** `src/integrity.py` computes an HMAC-SHA256 signature using `FACE_LOCK_SIGNING_KEY` during registration. If the template is modified, the signature check fails, blocking authentication.
- **Known Weakness:** The key is stored in a local environment variable. If the attacker has Administrator access or can read the environment variables, they can compute a valid signature for their forged template.
- **Next Mitigation:** Protect the signing key using Windows Credential Manager or DPAPI, preventing unauthorized local extraction.

### 4. Configuration Tampering
**Threat:** An attacker modifies `src/config.py` to change `owner_name`, adjust `cosine_threshold` to `1.0`, or set `PAD_SETTINGS.enabled = False`.
- **Likelihood:** Medium. Requires local filesystem access.
- **Impact:** System restrictions are completely disabled.
- **Current Defense:** None. The application trusts its own source code and configuration.
- **Known Weakness:** Python scripts are executed as plain text and can be edited.
- **Next Mitigation:** Compile the application (e.g., PyInstaller) or enforce filesystem ACLs to restrict write access to the Face Lock installation directory.

### 5. Hardware Interception (Virtual Camera Spoof)
**Threat:** An attacker uses a virtual webcam (e.g., OBS Virtual Camera) to inject a pre-recorded deepfake video stream into the application.
- **Likelihood:** Low to Medium.
- **Impact:** Bypasses presentation attack detection if the video passes the active challenge.
- **Current Defense:** None. `cv2.VideoCapture(0)` reads whatever stream Windows presents as the default camera.
- **Known Weakness:** No verification that the camera stream originates from the secure, physical laptop webcam.
- **Next Mitigation:** Implement Windows Media Foundation (WMF) device enumeration to explicitly block known virtual camera drivers.

### 6. DoS via Repeated Failures
**Threat:** An attacker repeatedly fails authentication, forcing the system to endlessly process embeddings.
- **Likelihood:** High.
- **Impact:** High CPU usage, battery drain, or potential thermal throttling.
- **Current Defense:** `RECOGNITION_SETTINGS.cooldown_seconds` (default 1.5s) limits the frequency of recognition attempts.
- **Known Weakness:** No lockout policy after *N* consecutive failures. The system will loop infinitely.
- **Next Mitigation:** Implement a rate limiter or exponential backoff (e.g., lock for 60 seconds after 5 consecutive failed attempts).

## Conclusion
The current implementation provides strong protection against casual impostors and basic photos. However, against an attacker with Administrator-level filesystem access, the system is fundamentally vulnerable because the Python scripts and environment variables can be altered. Future mitigations must focus on empirical threshold validation, ONNX-based PAD, and OS-level integration.
