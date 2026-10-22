# Owner Verification Protocol

The Face Lock system must be empirically proven to consistently grant access to the **legitimate laptop owner** while minimizing False Rejects (FRR). This document outlines the rigorous protocol for capturing and evaluating the owner's biometric data across multiple sessions.

## 1. Objective
Measure the system's ability to recognize the owner under varying, realistic laptop usage conditions.

## 2. Directory Structure
All owner data must be recorded into the `eval/datasets/face_lock_owner` directory.

```text
eval/
  datasets/
    face_lock_owner/
      owner_001/
        session_01/  <-- ENROLLMENT (System ground truth)
          frame_00001.jpg
          ...
        session_02/  <-- EVALUATION (Verification attempt 1)
          ...
        session_03/  <-- EVALUATION (Verification attempt 2)
          ...
```

## 3. Data Collection Procedure

**Important:** Do NOT use frames from `session_01` (Enrollment) to test the system in the evaluation phase. Verification must strictly compare later sessions against the enrollment baseline.

### Step 1: Owner Enrollment (`session_01`)
1. Run `python tools/collect_recognition_dataset.py`.
2. When prompted for an ID, enter `owner_001`.
3. Sit naturally in front of the laptop under standard lighting.
4. Press `s` to capture the **Enrollment** session.
5. Provide a minimum of 5, and ideally 10, valid frames. 
6. Wait for the `DeepFace` validation buffer to complete and save the frames.

### Step 2: Owner Evaluation (`session_02`)
1. Change a variable: wait 5 minutes, move to a different room, adjust the lighting, or take off/put on glasses.
2. In the same tool, press `s` to capture the **Evaluation** session.
3. Provide 10-20 valid frames. 
4. Move your head slightly, blink, and vary your expression to capture natural variations.

### Step 3: Owner Evaluation (`session_03+`)
1. Wait at least a few hours or until the next day.
2. Run `python tools/collect_recognition_dataset.py` again.
3. Enter `owner_001`. The tool will automatically resume and prepare to capture `session_03`.
4. Press `s` to capture the next **Evaluation** session.
5. Provide 10-20 valid frames.

### Step 4: Finalize Dataset
1. Press `q` to quit the collection tool.
2. Run `python tools/generate_manifest.py --collection-root eval/datasets/face_lock_owner`.
3. Run `python tools/qc_dataset.py --manifest eval/datasets/face_lock_owner/manifest.json`.
4. Ensure the dataset passes the cryptographic leakage audit (no shared frames between sessions).

## 4. Evaluation Execution
Once the owner dataset is cleanly collected, it can be evaluated by the standard recognition evaluation harness. The system will compare every frame in `session_02` and `session_03` against the `session_01` enrollment template.

The goal is to measure the **False Reject Rate (FRR)**.
- Expected Outcome: `ACCESS GRANTED` for all evaluation frames.
- A False Reject indicates the threshold (`0.30`) is too strict, or the enrollment template is not representative of the owner's variations.
