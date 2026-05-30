"""
Load PAD evaluation clips and still images from dataset manifests.

Supports numbered frame sequences (frame_0001.jpg) and single-image stills.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PresentationType = Literal["clip", "still"]

_FRAME_PATTERN = re.compile(r"^frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
_SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True, slots=True)
class LoadedPresentation:
    """Frames loaded for one manifest sample."""

    frames: tuple[Any, ...]
    source_paths: tuple[Path, ...]
    presentation_type: PresentationType

    @property
    def frame_count(self) -> int:
        return len(self.frames)


class ClipLoadError(FileNotFoundError):
    """Raised when presentation media cannot be resolved."""


def resolve_sample_path(collection_root: Path, relative_path: str) -> Path:
    """Resolve a manifest-relative path under the collection root."""
    root = collection_root.resolve()
    candidate = (root / relative_path).resolve()
    if not str(candidate).startswith(str(root)):
        raise ClipLoadError(
            f"Sample path escapes collection root: {relative_path!r}"
        )
    return candidate


def discover_frame_paths(session_dir: Path) -> list[Path]:
    """
    Discover numbered frame files in a session directory.

    Prefer frame_NNNN.ext naming; fall back to sorted image files.
    """
    if not session_dir.is_dir():
        raise ClipLoadError(f"Session directory not found: {session_dir}")

    numbered: list[tuple[int, Path]] = []
    fallback: list[Path] = []

    for path in session_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        match = _FRAME_PATTERN.match(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
        else:
            fallback.append(path)

    if numbered:
        numbered.sort(key=lambda item: item[0])
        return [path for _, path in numbered]

    if fallback:
        fallback.sort(key=lambda path: path.name.lower())
        return fallback

    raise ClipLoadError(f"No image frames found in {session_dir}")


def load_image_bgr(path: Path) -> Any:
    """Load one image as a BGR numpy array."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ClipLoadError(f"Failed to decode image: {path}")
    return image


def load_still_frame(
    collection_root: Path,
    relative_path: str,
    *,
    image_loader: Callable[[Path], Any] | None = None,
) -> LoadedPresentation:
    """Load a single still image presentation."""
    loader = image_loader or load_image_bgr
    path = resolve_sample_path(collection_root, relative_path)

    if path.is_dir():
        frame_paths = discover_frame_paths(path)
        if len(frame_paths) != 1:
            raise ClipLoadError(
                f"Still presentation path is a directory with "
                f"{len(frame_paths)} images; expected one: {path}"
            )
        path = frame_paths[0]
    elif not path.is_file():
        raise ClipLoadError(f"Still image not found: {path}")

    return LoadedPresentation(
        frames=(loader(path),),
        source_paths=(path,),
        presentation_type="still",
    )


def load_clip_frames(
    collection_root: Path,
    relative_path: str,
    *,
    image_loader: Callable[[Path], Any] | None = None,
    max_frames: int | None = None,
) -> LoadedPresentation:
    """Load a multi-frame clip from a session directory."""
    loader = image_loader or load_image_bgr
    session_dir = resolve_sample_path(collection_root, relative_path)

    if session_dir.is_file():
        return LoadedPresentation(
            frames=(loader(session_dir),),
            source_paths=(session_dir,),
            presentation_type="clip",
        )

    frame_paths = discover_frame_paths(session_dir)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    if not frame_paths:
        raise ClipLoadError(f"No frames loaded for clip: {session_dir}")

    frames = tuple(loader(path) for path in frame_paths)
    return LoadedPresentation(
        frames=frames,
        source_paths=tuple(frame_paths),
        presentation_type="clip",
    )


def load_presentation(
    collection_root: Path,
    relative_path: str,
    presentation_type: PresentationType,
    *,
    image_loader: Callable[[Path], Any] | None = None,
    max_frames: int | None = None,
) -> LoadedPresentation:
    """Load frames for a manifest sample."""
    if presentation_type == "still":
        return load_still_frame(
            collection_root,
            relative_path,
            image_loader=image_loader,
        )
    return load_clip_frames(
        collection_root,
        relative_path,
        image_loader=image_loader,
        max_frames=max_frames,
    )


def frames_from_arrays(arrays: Sequence[Any]) -> LoadedPresentation:
    """Build an in-memory presentation (used in tests)."""
    if not arrays:
        raise ClipLoadError("Cannot build presentation from empty frame list.")
    return LoadedPresentation(
        frames=tuple(arrays),
        source_paths=tuple(),
        presentation_type="clip",
    )
