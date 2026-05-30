# Face Lock AI System

A modular desktop face recognition system built with Python, OpenCV, and
DeepFace. The project captures trusted users through a webcam, converts their
faces into AI embeddings, and later compares a live webcam face against saved
embeddings to decide whether access should be granted or denied.

This is a beginner-friendly AI engineering project with a clean architecture
that separates camera capture, face embedding generation, registration,
recognition, and application orchestration.

> Security note: this project is for learning, demos, and portfolio use. It is
> not a production-grade operating system lock or biometric security product.

## Features

- Webcam-based user registration
- Multiple face images per registered user
- Face embeddings generated with `DeepFace.represent()`
- `Facenet512` model for face representation
- Cosine distance based matching
- Configurable recognition threshold
- Live webcam recognition
- On-screen OpenCV access status
- Green display for access granted
- Red display for access denied
- Frame skipping for better performance
- Cooldown between recognition attempts
- Modular project structure
- Beginner-friendly comments and functions

## Project Overview

The system has two main workflows:

1. Registration workflow

```text
User name
   -> Webcam capture
   -> Save face images
   -> Generate face embeddings
   -> Save embeddings in encodings/
```

2. Recognition workflow

```text
Live webcam frame
   -> Generate live embedding
   -> Load trusted embeddings
   -> Compare vectors
   -> Apply threshold
   -> ACCESS GRANTED or ACCESS DENIED
```

## AI Architecture

The project is split into small modules. Each file has one clear job.

```text
                 app.py
                   |
        -------------------------
        |                       |
  register.py             recognize.py
        |                       |
        |                       |
  capture.py              capture.py
        |                       |
        -------- ai_engine.py ---
                    |
                 utils.py
                    |
                 config.py
```

### Architecture Responsibilities

| Layer | File | Responsibility |
| --- | --- | --- |
| Entry point | `app.py` | Shows menu and calls workflows |
| Registration workflow | `src/register.py` | Captures user images and saves embeddings |
| Recognition workflow | `src/recognize.py` | Runs live webcam recognition |
| Camera layer | `src/capture.py` | Opens webcam, reads frames, saves images |
| Display layer | `src/display.py` | Draws OpenCV overlays, status text, warnings, and display helpers |
| PAD layer | `src/pad_engine.py` | Presentation attack detection (liveness) before identity matching |
| AI layer | `src/ai_engine.py` | Creates embeddings and compares vectors |
| Configuration layer | `src/config.py` | Centralizes paths, thresholds, model defaults, and runtime settings |
| Logging layer | `src/logging_config.py` | Centralizes console and rotating file logging |
| Utility layer | `src/utils.py` | Paths, folders, pickle files, messages |

## Folder Structure

```text
face-lock/
|
|-- data/
|   `-- <user_name>/
|       |-- user_20260527_101500_123456.jpg
|       `-- user_20260527_101505_654321.jpg
|
|-- encodings/
|   `-- <user_name>.pkl
|
|-- models/
|   `-- reserved for future model/config files
|
|-- src/
|   |-- capture.py
|   |-- pad_engine.py
|   |-- ai_engine.py
|   |-- config.py
|   |-- display.py
|   |-- logging_config.py
|   |-- utils.py
|   |-- register.py
|   `-- recognize.py
|
|-- app.py
|-- CONFIGURATION.md
|-- LOGGING.md
|-- requirements.txt
`-- README.md
```

### Folder Responsibilities

| Path | Purpose |
| --- | --- |
| `data/` | Stores captured face images for each user |
| `encodings/` | Stores saved face embeddings as `.pkl` files |
| `models/` | Reserved for future local models or configuration |
| `src/` | Contains the main project modules |
| `app.py` | Main terminal application |
| `CONFIGURATION.md` | Configuration architecture and migration guidance |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| OpenCV | Webcam access, image display, image saving |
| DeepFace | Face detection, alignment, and embedding generation |
| TensorFlow/Keras | Model backend used by DeepFace |
| NumPy | Vector math for embedding comparison |
| Pickle | Stores embedding records locally |

## Recommended Python Version

Use Python `3.11.x`.

This project intentionally uses a stable TensorFlow/Keras 2.15 stack to avoid
common TensorFlow 2.16+ and Keras 3 compatibility issues with DeepFace.

Avoid Python `3.12` or newer for this exact dependency set unless you update and
test the ML stack carefully.

## Installation

### 1. Clone or Open the Project

```powershell
cd C:\face_detection
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade Build Tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

The current dependency set is:

```text
numpy==1.26.4
opencv-python==4.10.0.84
tensorflow==2.15.1
keras==2.15.0
deepface==0.0.93
```

## Running the App

Start the application:

```powershell
python app.py
```

You will see:

```text
Face Lock AI System
------------------------------------------------

Choose an option:
1. Register new user
2. Start live recognition
3. Exit
```

## Usage Examples

### Register a New User

1. Run the app.
2. Choose option `1`.
3. Enter the user name.
4. A webcam window opens.
5. Press `s` to capture an image.
6. Capture around 10 images.
7. Press `q` to cancel if needed.
8. The system generates and saves embeddings.

Output files:

```text
data/<user_name>/*.jpg
encodings/<user_name>.pkl
```

### Start Live Recognition

1. Run the app.
2. Choose option `2`.
3. A webcam window opens.
4. Look at the camera.
5. The app displays one of:

```text
ACCESS GRANTED
ACCESS DENIED
```

6. Press `q` to stop recognition.

## How Registration Works

Registration creates trusted identity data.

```text
User enters name
   -> data/<user_name>/ folder is created
   -> webcam opens
   -> user captures several images
   -> each image is converted into an embedding
   -> embeddings are saved to encodings/<user_name>.pkl
```

Multiple registration images improve accuracy because a single face image can be
affected by lighting, blur, angle, expression, glasses, or camera quality. A set
of images gives the system several examples of the same person.

Registration quality matters. Clear, front-facing, well-lit images produce
better embeddings and more reliable recognition.

## How Recognition Works

Recognition compares a live face against saved trusted embeddings.

```text
Open webcam
   -> read live frame
   -> generate live embedding
   -> load trusted embeddings
   -> compare live embedding with saved embeddings
   -> find closest match
   -> apply threshold
   -> show access decision
```

Recognition does not compare raw images. It compares numeric vectors generated
by the AI model.

## AI Pipeline

```text
Image or webcam frame
        |
        v
Face detection
        |
        v
Face alignment
        |
        v
DeepFace.represent()
        |
        v
Embedding vector
        |
        v
Cosine distance comparison
        |
        v
Threshold decision
        |
        v
ACCESS GRANTED / ACCESS DENIED
```

## Key AI Concepts

### Embeddings

An embedding is a numeric representation of a face.

Example:

```text
[0.12, -0.44, 0.91, 0.03, ...]
```

The numbers are learned features created by the AI model. They represent
identity-related information in a compact vector.

### Why Embeddings Instead of Raw Images

Raw image comparison is unreliable because pixels change when lighting, camera
angle, background, or resolution changes.

Embeddings are better because they are designed to represent the identity of a
face rather than the exact pixels of one photo.

### Cosine Similarity and Cosine Distance

Cosine similarity compares the direction of two vectors.

```text
higher cosine similarity = more similar faces
```

Cosine distance is:

```text
cosine distance = 1 - cosine similarity
```

For this project:

```text
lower cosine distance = better match
higher cosine distance = worse match
```

### Threshold Logic

The threshold decides whether the closest face is close enough.

```text
if cosine_distance <= threshold:
    ACCESS GRANTED
else:
    ACCESS DENIED
```

The default threshold is:

```text
0.30
```

This value is used with:

```text
Facenet512 + cosine distance
```

### Why DeepFace.represent() Was Used

This project uses `DeepFace.represent()` instead of relying only on
`DeepFace.verify()` because `represent()` gives direct access to the embedding
pipeline.

Benefits:

| Benefit | Explanation |
| --- | --- |
| Reusable embeddings | Register once, compare many times |
| Faster recognition | Compare saved vectors instead of saved images |
| Clear AI design | Detection, embedding, comparison, and threshold are explicit |
| Easier debugging | You can inspect distances and thresholds |
| Scalable structure | More users can be added without changing the pipeline |

## Performance Optimizations

| Optimization | Why It Helps |
| --- | --- |
| Saved embeddings | Avoids recomputing trusted user embeddings every time |
| Frame skipping | Runs AI on selected frames instead of every webcam frame |
| Cooldown | Prevents repeated heavy inference too quickly |
| Brightness check | Avoids expensive AI calls on very dark frames |
| No temp image files | Passes live frames directly into the AI pipeline |
| Modular loading | Trusted embeddings are loaded once before the recognition loop |

Live recognition can become slow because face detection and `Facenet512`
embedding generation are expensive. A webcam may provide many frames per second,
but the AI model may not process every frame at that speed. Frame skipping keeps
the application responsive.

## Troubleshooting

| Problem | Likely Cause | Fix |
| --- | --- | --- |
| `No matching distribution found for tensorflow==2.15.1` | Wrong Python version | Use Python `3.11.x` |
| Keras or `tf-keras` errors | TensorFlow/Keras version mismatch | Use the pinned `requirements.txt` |
| Webcam does not open | Camera index or permissions issue | Close other camera apps, check permissions, try camera index `1` |
| First DeepFace run is slow | Model weights are downloading | Wait for the first model download to finish |
| `No face detected` | Poor lighting, face angle, blur | Face the camera and improve lighting |
| `Multiple faces detected` | More than one face in frame | Keep only one face visible |
| Registered user denied | Poor registration images or strict threshold | Re-register with clearer images |
| Recognition is slow | Model inference is heavy | Increase frame skip or cooldown |
| Wrong person accepted | Threshold too loose or weak registration | Lower threshold and add liveness checks |

## Security Limitations

This project demonstrates face recognition, but it is not a complete security
system.

Important limitations:

- PAD uses heuristic RGB scoring (not certified anti-spoof hardware)
- Advanced replays, 3D masks, and virtual cameras may still bypass PAD
- Embeddings are stored locally without encryption
- It does not lock or unlock the actual operating system
- It does not include audit logs or tamper protection
- Webcam quality and lighting strongly affect reliability

For production security, add certified PAD models, encrypted storage, audit logging,
multi-factor authentication, and operating-system-level integration.

## Screenshots

Add screenshots here when the app is running locally.

| Screen | Placeholder |
| --- | --- |
| Main menu | `docs/screenshots/main-menu.png` |
| Registration capture | `docs/screenshots/registration.png` |
| Access granted | `docs/screenshots/access-granted.png` |
| Access denied | `docs/screenshots/access-denied.png` |

## Future Improvements

- Upgrade PAD to ONNX/certified anti-spoof models
- Encrypt saved embeddings
- Add user deletion workflow
- Add user listing workflow
- Add SQLite metadata storage
- Add confidence reporting and logs
- Add GUI with Tkinter, PyQt, or a web frontend
- Add configurable thresholds per model
- Add support for multiple camera devices
- Add unit tests for utility and vector comparison functions
- Add OS-level lock/unlock integration

## Why Modular Architecture Matters

Modular architecture keeps each part of the project focused.

```text
capture.py   -> camera only
pad_engine.py -> PAD / liveness only
ai_engine.py -> AI only
config.py    -> centralized settings only
display.py   -> OpenCV UI rendering only
logging_config.py -> logging setup only
register.py  -> registration workflow only
recognize.py -> recognition workflow only
app.py       -> menu and orchestration only
utils.py     -> shared helpers only
```

This makes the project easier to:

- understand
- debug
- test
- extend
- replace one part without breaking the others

For example, you can switch from `Facenet512` to another DeepFace model inside
`src/config.py` without rewriting the menu, camera code, or AI workflow code.

## Development Notes

Run syntax checks:

```powershell
python -m py_compile app.py src\config.py src\utils.py src\capture.py src\pad_engine.py src\ai_engine.py src\display.py src\logging_config.py src\register.py src\recognize.py
```

Run the main app:

```powershell
python app.py
```

Direct module testing:

```powershell
python src\capture.py
python src\register.py
python src\recognize.py
```

## Disclaimer

This project is intended for educational use. Do not rely on it as the only
security mechanism for protecting real devices, accounts, or sensitive data.
