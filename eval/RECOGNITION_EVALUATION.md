# Recognition Evaluation Methodology

This document outlines the pipeline used for validating the Face Recognition logic of Face Lock AI.

## Dataset

Currently, the evaluation is based on `face_lock_v1`.
- **Identities:** 2
- **Enrollment Samples:** 2 (subject_001 session_01, subject_002 session_01)
- **Evaluation Samples:** 2 (subject_001 session_02, subject_002 session_02)
- **Total Genuine Comparisons:** 10
- **Total Impostor Comparisons:** 10

*Note:* This dataset is scientifically insufficient for production threshold tuning due to its tiny size. 

## Methodology

1. **Enrollment/Evaluation Separation:** We enforce strict separation. `session_01` for each subject is used as the enrollment template, and `session_02` is strictly for evaluation.
2. **Production Pipeline Reuse:** The evaluation uses the exact `generate_embedding` and `cosine_distance` functions from `src.ai_engine.py` to ensure the measured system is identical to the deployed system.
3. **Metric Definitions:**
   - **False Accept Rate (FAR):** False Accepts / Total Impostor Attempts
   - **False Reject Rate (FRR):** False Rejects / Total Genuine Attempts
   - **Distance Metric:** Cosine Distance (Lower is more similar).
4. **Genuine Pairs:** Every evaluation frame is matched against the enrolled template of the *same* subject. The distance must fall at or below the threshold for a True Accept.
5. **Impostor Pairs:** Every evaluation frame is matched against the enrolled template of *different* subjects. The distance must fall at or below the threshold for a False Accept.

## Evaluation Script

The evaluation can be executed reproducibly:
```bash
python eval/harness/run_recognition_eval.py \
    --collection-root eval/datasets/face_lock_v1 \
    --output-dir eval/reports
```

## Results & Limitations

The output is written to `eval/reports/recognition_eval_report.json`.

**Limitations:**
- The dataset lacks demographic, pose, and lighting diversity.
- The sample size (2 subjects) cannot statistically validate a production threshold.
- The high FAR (0.70 at threshold 0.30) observed on this tiny dataset suggests that further tuning and a much larger dataset are required to establish an evidence-backed threshold.
