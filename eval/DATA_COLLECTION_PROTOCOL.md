# Face Lock AI — Data Collection Protocol

This document defines the standard operating procedure for collecting real-world Face Recognition evaluation datasets from consenting human participants.

## 1. Overview
The goal of this campaign is to collect a statistically significant number of participants (e.g., 50+ for Stage B, 500+ for Stage C) using internal webcams to create a zero-shot, leakage-free evaluation dataset.

## 2. Participant Privacy & Consent
- Ensure all participants consent to having their biometric data captured for internal R&D purposes.
- **NEVER** store names, emails, or PII in the dataset.
- Participants are strictly referred to by an anonymous identifier (e.g., `subj_001`, `subj_002`).

## 3. Collection Tool
Use the interactive capture tool:
```bash
python tools/collect_recognition_dataset.py
```
This tool enforces:
- Exactly 1 face per frame via live offline DeepFace validation.
- Silent duplicate frame dropping (via SHA-256) to handle camera stutter.
- Separation of `session_01` and `session_02`.
- Minimum capture limits.

### Dry-Run Mode
If you wish to practice the collection workflow, test the camera, or verify the UI without writing ANY files to disk, use the dry-run flag:
```bash
python tools/collect_recognition_dataset.py --dry-run
```

## 4. Enrollment Procedure (`session_01`)
**Goal:** Establish the ground-truth biometric template.
1. The tool will suggest an anonymous ID (e.g., `subj_015`). Accept it.
2. The tool will request capture for `session_01`. Minimum required valid frames: 5 (target: 10).
3. Ask the participant to look naturally at the camera in standard office lighting.
4. Press `s` to begin capture. The tool automatically buffers 10 frames over ~2 seconds.
5. The tool validates the frames. Ensure at least 5 valid frames are saved. If you save fewer than 5 frames, the tool will warn `INCOMPLETE PARTICIPANT`.

## 5. Evaluation Procedure (`session_02`)
**Goal:** Capture realistic evaluation probes.
1. Immediately following `session_01`, the tool automatically proceeds to `session_02`. Minimum required valid frames: 10 (target: 20).
2. Instruct the participant to introduce realistic variations:
   - Slight head pose (look slightly left/right/up/down).
   - Change expression (smile, neutral).
   - Move slightly closer to or farther from the camera.
   - (Optional) Change lighting if possible, e.g., turn off an overhead light.
3. Press `s` to begin capture. The tool automatically buffers 20 frames over ~4 seconds.
4. The tool validates the frames. Ensure at least 10 valid frames are saved.
5. The subject is now fully captured.

## 6. How to finish one participant & start the next
When the tool prompts:
`Capture another participant? [y/N]:`
Type `y` and press enter. The tool will automatically increment the ID (e.g., `subj_016`) and begin `session_01` again for the next person.

## 7. How to Resume or Fix Incomplete Participants
If a session is interrupted (e.g., crash, participant leaves early), simply run the tool again and type in the interrupted ID (e.g., `subj_015`). 
- If `session_01` has $< 5$ frames, the tool will resume capturing `session_01` and append to the existing sequence.
- If `session_01` has $\ge 5$ frames, the tool will advance to `session_02`.
- You will be warned about `INCOMPLETE PARTICIPANTS` until they meet the minimum thresholds.

## 8. Generating the Manifest
Once the capture session is over, you MUST generate the JSON manifest mapping the new files to the evaluation protocol.
```bash
python tools/generate_manifest.py --collection-root eval/datasets/face_lock_real --collection-id face_lock_real
```

## 9. Quality Control & Leakage Audit
After generating the manifest, run the dataset through the strict QC pipeline to verify zero cryptographic leakage and perfect face counts:
```bash
python tools/qc_dataset.py --collection-root eval/datasets/face_lock_real
```
If the output shows `PASS` or `WARNING` (with 0 Errors), the dataset is cryptographically sound. Any `data_leakage_duplicate_hash` or `no_face_detected` error strictly invalidates the dataset.
