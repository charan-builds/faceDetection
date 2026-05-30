# Dataset collection template

Use this folder as a **layout reference** when creating a new PAD evaluation collection.

Copy the structure to:

```text
eval/datasets/<collection_id>/
```

Real biometric media must **not** be committed to git.

---

## Directory layout

```text
<collection_id>/
├── manifest.json              ← required; validates against manifest.schema.json
├── bona_fide/
│   └── subject_<nnn>/
│       └── session_<nn>/
│           ├── frame_0001.jpg
│           ├── frame_0002.jpg
│           └── meta.json      ← optional
├── attacks/
│   ├── PA-01_print/
│   │   └── subject_<nnn>/
│   │       └── attempt_<nn>/
│   ├── PA-02_screen_static/
│   ├── PA-03_screen_video/
│   └── PA-04_virtual_camera/
└── registration/              ← optional, for PAD-REG-v1
    ├── bona_fide/
    └── attacks/
```

---

## PAI categories

| Folder | PAI ID | Description |
|--------|--------|-------------|
| `bona_fide/` | BF | Live subject |
| `PA-01_print/` | PA-01 | Printed photo |
| `PA-02_screen_static/` | PA-02 | Static image on screen |
| `PA-03_screen_video/` | PA-03 | Video replay on screen |
| `PA-04_virtual_camera/` | PA-04 | Virtual camera injection |

---

## manifest.json

See `manifest.schema.json` in this folder.

Example entry:

```json
{
  "sample_id": "BF_subj001_s01",
  "pai_id": "BF",
  "ground_truth": "bona_fide",
  "subject_id": "subj_001",
  "path": "bona_fide/subject_001/session_01",
  "presentation_type": "clip",
  "capture_device": "laptop_integrated",
  "lux_bucket": "normal",
  "frame_count": 12
}
```

---

## Collection checklist

- [ ] Pseudonymous subject IDs only
- [ ] Minimum sample sizes per `eval/protocols/pad_iso_lab_v1.yaml`
- [ ] manifest validates against schema
- [ ] Collection stored outside public git remote
