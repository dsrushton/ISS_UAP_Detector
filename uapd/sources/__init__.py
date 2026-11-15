"""Frame sources: anything that yields BGR frames in order."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class SourceError(RuntimeError):
    """A source could not be opened or read."""


class FrameSource(Protocol):
    def describe(self) -> dict[str, Any]: ...

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (zero-based index, BGR frame)."""
        ...


class VideoFileSource:
    """Frames of a video file, decoded with OpenCV."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise SourceError(f"video file not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened():
                raise SourceError(f"could not open video: {self.path}")
            self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "video",
            "path": str(self.path),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(str(self.path))
        try:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield index, frame
                index += 1
        finally:
            cap.release()


class FrameFolderSource:
    """Image files in a folder, in filename order."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise SourceError(f"folder not found: {self.path}")
        self.files = sorted(
            p for p in self.path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.files:
            raise SourceError(f"no image files in folder: {self.path}")

    def describe(self) -> dict[str, Any]:
        return {"kind": "folder", "path": str(self.path), "files": len(self.files)}

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for index, file in enumerate(self.files):
            frame = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if frame is None:
                raise SourceError(f"could not read image: {file}")
            yield index, frame


def open_source(path: str | Path) -> FrameSource:
    """Open a folder of frames or a video file."""
    p = Path(path)
    if p.is_dir():
        return FrameFolderSource(p)
    if p.is_file():
        return VideoFileSource(p)
    raise SourceError(f"source not found: {p}")
