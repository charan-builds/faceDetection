# Impostor Verification Protocol

The fundamental security question of Face Lock is: **"Can an unauthorized person unlock the laptop?"** This protocol defines the rigorous procedure for collecting and evaluating Impostor data to measure the False Accept Rate (FAR) against the legitimate owner's template.

## 1. Objective
Measure the system's ability to reject non-owners attempting to authenticate against the laptop owner's biometric template.

## 2. Directory Structure
All impostor data must be recorded into the `eval/datasets/face_lock_impostor` directory.

```text
eval/
  datasets/
    face_lock_impostor/
      impostor_001/
        session_01/  <-- IMPOSTOR ATTEMPT 1
          frame_00001.jpg
          ...
      impostor_002/
        session_01/  <-- IMPOSTOR ATTEMPT 1
          ...
```

**CRITICAL RULE:** Impostors NEVER undergo an "Enrollment" phase. Do not use their data to create trusted templates. They exist solely as Zero-Effort attackers against the `owner_001` template.

## 3. Data Collection Procedure

### Consent and Privacy
1. Ensure the individual explicitly consents to having their face captured.
2. Ensure the individual understands this data will only be used locally for Face Lock evaluation.
3. **DO NOT** use the person's real name. Use anonymous identifiers (`impostor_001`, `impostor_002`, etc.).

### Step 1: Impostor Capture
1. Run `python tools/collect_recognition_dataset.py`.
2. When prompted for an ID, enter `impostor_001` (or the next available sequential ID).
3. Have the impostor sit naturally in front of the laptop under standard lighting.
4. Press `s` to capture the session.
5. Ask the impostor to provide 10-20 valid frames, varying their pose slightly, to mimic a legitimate unlock attempt.
6. Press `q` to quit the collection tool when finished, or `y` to start the next impostor.

### Step 2: Finalize Dataset
1. Run `python tools/generate_manifest.py --collection-root eval/datasets/face_lock_impostor`.
2. Run `python tools/qc_dataset.py --manifest eval/datasets/face_lock_impostor/manifest.json`.
3. Ensure the dataset passes the cryptographic leakage audit.

## 4. Evaluation Execution
Once the Impostor dataset is collected, the evaluation harness will iterate through every frame captured from every impostor and compare it against the `owner_001` template collected per the `OWNER_VERIFICATION_PROTOCOL.md`.

The goal is to measure the **False Accept Rate (FAR)**.
- Expected Outcome: `ACCESS DENIED` for every single impostor frame.
- A False Accept (an impostor is granted access) is a critical security vulnerability and indicates that the threshold (`0.30`) is too lenient or the model cannot distinguish between the owner and the specific impostor.
- **Goal:** Strict 0% FAR for laptop security.
