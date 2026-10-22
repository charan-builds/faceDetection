# Face Lock AI — Dataset Route Decision

## Comparison: Public Benchmark vs Internal Webcam

| Criterion                        | Public Benchmark (e.g., LFW) | Controlled Internal Webcam |
| -------------------------------- | ---------------------------- | -------------------------- |
| **Identity count**               | High (1,680 with $\ge$ 2 imgs) | Low currently (2). High effort to scale. |
| **Image/video format**           | Web-gathered JPEGs (news)    | Webcam frame sequences     |
| **Labels**                       | Subject names (Celebs)       | Anonymous (`subj_001`)     |
| **Enrollment/eval separation**   | Can be simulated if >= 2 imgs| Explicitly designed (`session_01` vs `session_02`) |
| **Webcam realism**               | Low (Professional photos)    | High (Actual target domain)|
| **Lighting variation**           | Unconstrained / Wild         | Realistic office/home      |
| **Pose variation**               | Unconstrained / Wild         | Constrained to screen-facing |
| **Reproducibility**              | High (Standard benchmark)    | High (Deterministic manifest) |
| **Licensing**                    | Non-commercial / Research    | Internal / Proprietary     |
| **Domain relevance**             | Moderate to Low              | Perfect                    |
| **Integration effort**           | Medium (Write importer script) | High (Manual collection campaign) |
| **Main limitations**             | Domain gap (Celeb vs Webcam) | Too small currently; impossible for AI to collect manually |

## Decision
**Route Selected:** Route A (Public Benchmark - Labeled Faces in the Wild subset) for **Stage B** validation, while continuing Route B for **Stage C** production.

**Why?**
The evaluation pipeline currently suffers from an extreme lack of impostor comparisons. We need at least 50 identities to reach Stage B (2,450 impostor pairs) to characterize the mathematical bounds of the 0.30 threshold. As an autonomous agent, I cannot physically collect 50 real humans for a webcam campaign. However, I *can* build an automated importer for LFW (Labeled Faces in the Wild) to instantly provide thousands of identities. 
To satisfy the Face Lock AI protocol, the importer will filter LFW for identities with exactly $\ge 2$ images, mapping image 1 to `session_01` (Enrollment) and image 2 to `session_02` (Evaluation), perfectly mirroring our canonical structure.

**Limitations:**
The domain gap between LFW (celebrity news photos) and Face Lock AI (close-up webcam frames) is significant. Thresholds derived from LFW will be scientifically rigorous in terms of sample size, but may not perfectly translate to webcam performance. Stage C must still rely on a large-scale internal webcam collection.

## Stage Requirements
- **Current Available (Stage A):** 2 identities.
- **Stage B Target:** ~50+ identities. We will target ~1,000 LFW identities for a strong intermediate validation.
- **Stage C Target:** 500+ real internal webcam identities.
