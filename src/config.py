"""
Centralized configuration for the Face Lock AI project.

This module contains settings only. It should not import application modules such
as ai_engine, capture, register, recognize, or utils. Keeping config independent
prevents circular imports and makes settings reusable across the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


def resolve_project_root() -> Path:
    """
    Resolve the project root without depending on the current working directory.

    src/config.py -> src/ -> project root
    """
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PathSettings:
    """
    Filesystem locations used by the application.

    Paths are derived from the location of this file, not from where Python is
    launched. This keeps the app portable across terminals, IDEs, and OSes.
    """

    project_root: Path = field(default_factory=resolve_project_root)
    data_dir_name: str = "data"
    encodings_dir_name: str = "encodings"
    logs_dir_name: str = "logs"
    models_dir_name: str = "models"

    @property
    def data_dir(self) -> Path:
        """Captured registration images live here at runtime."""
        return self.project_root / self.data_dir_name

    @property
    def encodings_dir(self) -> Path:
        """Generated face embedding records live here at runtime."""
        return self.project_root / self.encodings_dir_name

    @property
    def logs_dir(self) -> Path:
        """Runtime logs should be written here when logging is expanded."""
        return self.project_root / self.logs_dir_name

    @property
    def models_dir(self) -> Path:
        """Local model artifacts should live here if the project stores any."""
        return self.project_root / self.models_dir_name

    @property
    def runtime_directories(self) -> tuple[Path, ...]:
        """Directories the app may create or write to while running."""
        return (
            self.data_dir,
            self.encodings_dir,
            self.logs_dir,
            self.models_dir,
        )


@dataclass(frozen=True)
class AISettings:
    """
    Face embedding and comparison settings.
    """

    # DeepFace model used to generate face embeddings.
    model_name: str = "Facenet512"

    # Detector backend passed into DeepFace.represent().
    detector_backend: str = "opencv"

    # Current comparison metric. The threshold below is tuned for cosine distance.
    distance_metric: str = "cosine"

    # Lower cosine distance means a closer match. Distance <= threshold grants.
    cosine_threshold: float = 0.30

    # Keep face detection strict for a lock-style workflow.
    enforce_detection: bool = True

    # Align faces before embedding generation for more stable comparisons.
    align_faces: bool = True


@dataclass(frozen=True)
class RecognitionSettings:
    """
    Live recognition loop settings.
    """

    window_name: str = "Face Lock Recognition"

    # Run AI every N frames instead of every frame to reduce CPU/GPU load.
    frame_skip: int = 10

    # Minimum seconds between recognition attempts.
    cooldown_seconds: float = 1.5

    # Frames darker than this average brightness are rejected before inference.
    brightness_threshold: float = 45.0


@dataclass(frozen=True)
class DisplaySettings:
    """
    OpenCV display and overlay settings.
    """

    # OpenCV uses BGR color order, not RGB.
    color_granted: tuple[int, int, int] = (0, 180, 0)
    color_denied: tuple[int, int, int] = (0, 0, 255)
    color_info: tuple[int, int, int] = (255, 255, 255)
    color_warning: tuple[int, int, int] = (0, 200, 255)
    color_shadow: tuple[int, int, int] = (0, 0, 0)

    text_margin_x: int = 20
    primary_text_y: int = 40
    secondary_text_y: int = 78
    detail_text_y: int = 108
    footer_offset_y: int = 48
    quit_offset_y: int = 20

    primary_scale: float = 1.0
    secondary_scale: float = 0.7
    detail_scale: float = 0.6
    footer_scale: float = 0.55


@dataclass(frozen=True)
class WebcamSettings:
    """
    Camera and capture-window settings.
    """

    # OpenCV camera index. Usually 0 is the default laptop webcam.
    camera_index: int = 0

    capture_window_name: str = "Face Lock Capture"
    save_key: str = "s"
    quit_key: str = "q"


@dataclass(frozen=True)
class RegistrationSettings:
    """
    User registration workflow settings.
    """

    window_name: str = "Face Lock Registration"

    # Number of images to collect for a new user registration.
    image_count: int = 10

    # Reject registration if fewer than this many usable embeddings are created.
    minimum_embeddings: int = 3


@dataclass(frozen=True)
class StorageSettings:
    """
    File format settings for generated runtime data.
    """

    default_image_extension: str = "jpg"
    embedding_file_extension: str = "pkl"
    supported_image_extensions: tuple[str, ...] = ("jpg", "jpeg", "png")


@dataclass(frozen=True)
class LoggingSettings:
    """
    Logging-related defaults.
    """

    # Root application logger name. Module loggers become children of this name.
    logger_name: str = "face_lock"

    # DEBUG is useful during development; INFO is safer and quieter in production.
    level: str = "INFO"

    # Console stays readable, while file logs keep deeper diagnostic detail.
    console_level: str = "INFO"
    file_level: str = "DEBUG"

    # Rotating log file settings keep logs from growing without limit.
    file_name: str = "face_lock.log"
    max_bytes: int = 1_000_000
    backup_count: int = 5
    encoding: str = "utf-8"

    # Key-value style formatting is easy to scan and parse later.
    message_format: str = (
        "%(asctime)s | level=%(levelname)s | logger=%(name)s | "
        "module=%(module)s | function=%(funcName)s | message=%(message)s"
    )
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class PADSettings:
    """
    Presentation Attack Detection (PAD) / liveness settings.
    """

    enabled: bool = True
    assurance_tier: str = "standard"
    mode: str = "passive_then_active_on_borderline"

    min_frames_passive: int = 8
    max_frames_buffer: int = 24
    sample_every_n_frames: int = 1

    passive_pass_threshold: float = 0.62
    passive_reject_threshold: float = 0.38
    borderline_low: float = 0.45
    borderline_high: float = 0.58

    challenge_type: str = "head_turn_left"
    challenge_timeout_seconds: float = 3.0
    head_turn_min_shift_px: int = 12
    max_challenges_per_unlock: int = 1

    hardware_profile: str = "rgb_only"
    allow_virtual_camera: bool = True

    passive_model_id: str = "heuristic_rgb_v1"
    log_passive_scores: bool = False

    enforce_on_registration: bool = True
    registration_pad_burst_frames: int = 3


@dataclass(frozen=True)
class AppConfig:
    """
    Complete application configuration grouped by responsibility.
    """

    paths: PathSettings = field(default_factory=PathSettings)
    ai: AISettings = field(default_factory=AISettings)
    recognition: RecognitionSettings = field(default_factory=RecognitionSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    webcam: WebcamSettings = field(default_factory=WebcamSettings)
    registration: RegistrationSettings = field(default_factory=RegistrationSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    pad: PADSettings = field(default_factory=PADSettings)


CONFIG: Final[AppConfig] = AppConfig()

# Short aliases make imports readable in feature modules.
PATH_SETTINGS: Final[PathSettings] = CONFIG.paths
AI_SETTINGS: Final[AISettings] = CONFIG.ai
RECOGNITION_SETTINGS: Final[RecognitionSettings] = CONFIG.recognition
DISPLAY_SETTINGS: Final[DisplaySettings] = CONFIG.display
WEBCAM_SETTINGS: Final[WebcamSettings] = CONFIG.webcam
REGISTRATION_SETTINGS: Final[RegistrationSettings] = CONFIG.registration
STORAGE_SETTINGS: Final[StorageSettings] = CONFIG.storage
LOGGING_SETTINGS: Final[LoggingSettings] = CONFIG.logging
PAD_SETTINGS: Final[PADSettings] = CONFIG.pad
