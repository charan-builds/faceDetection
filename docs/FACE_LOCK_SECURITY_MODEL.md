# Face Lock Security Model

## 1. Primary Security Objective
The single purpose of Face Lock AI is **1:1 Owner Verification**. It is designed to act as a biometric padlock for a single, designated owner on a personal laptop. It is *not* a generalized multi-user (1:N) identification system. 

The security policy demands:
- **Owner Access:** If the physical person attempting access is the enrolled owner, grant access.
- **Impostor Denial:** If the physical person attempting access is *anyone else*, deny access.
- **Spoof Denial:** If the "person" is a photograph, video, or digital replay (a presentation attack), deny access.
- **Secure Fallback:** If the system is unsure, err on the side of denial (Fail Closed).

## 2. Authentication Policy
The authentication flow enforces the following sequence:

1. **Environmental Viability:**
   - Detect exactly one face in the frame. Multiple faces result in an automatic **DENY**.
   - Validate lighting conditions. Extremely dark environments result in an automatic **DENY**.

2. **Liveness / Presentation Attack Detection (PAD):**
   - The system executes a passive heuristic check (Laplacian variance, color distribution, and temporal motion).
   - If the check determines the face is non-live (e.g., a static photograph), the system issues a **DENY** without even computing identity.
   - Borderline liveness results in an Active Challenge (e.g., "turn head left").

3. **Identity Verification (1:1):**
   - The system retrieves the single trusted embedding assigned to `owner_name`.
   - A DeepFace embedding is generated for the live capture.
   - The system computes the cosine distance between the live embedding and the owner's trusted template.
   - If `distance <= 0.30`, the system issues an **ACCESS GRANTED**.
   - If `distance > 0.30`, or if the closest face in the `encodings/` database belongs to an unauthorized identity, the system issues a **DENY**.

## 3. Template Security
Biometric templates (embeddings) are stored as pickle files in the `encodings/` directory.
To prevent an attacker from swapping the owner's template with their own, Face Lock enforces an **HMAC-SHA256 signature** policy.

1. At enrollment time, the template is serialized into canonical JSON bytes.
2. A cryptographic digest is computed using `FACE_LOCK_SIGNING_KEY`.
3. At recognition time, the template is only loaded if the signature matches the payload.
4. Tampered, missing, or swapped signatures result in an `IntegrityError` and block authentication.

## 4. Threshold Calibration
The current acceptance threshold is **0.30** (cosine distance). 
*Security Notice:* This threshold is currently a DeepFace default and has *not* been empirically validated against real owner/impostor data on the target hardware. Real-world validation (Stage B and Stage C) is required before this threshold can be trusted in production.

## 5. Security Integration
Face Lock operates as an application layer. It currently runs in a loop via `recognize_live()`. It does not yet disable or replace Windows authentication (e.g., Windows Hello or password login). Face Lock serves as an *additional* layer of protection. It is crucial that failure of Face Lock does not permanently lock the user out of the Operating System.

## 6. What Face Lock Cannot Do
- Face Lock cannot protect the system if the attacker has Administrator access or can modify `src/config.py` to change `owner_name`.
- Face Lock's PAD is heuristic and may fail against sophisticated 3D masks or high-fidelity deepfake video injections.
- Face Lock cannot prevent an attacker with the `FACE_LOCK_SIGNING_KEY` from enrolling a new malicious template.
