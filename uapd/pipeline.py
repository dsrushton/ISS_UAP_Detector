"""The contract every detection pipeline implements, and the results it returns.

The headless harness only knows this interface, so V1 (through its adapter) and
each stage of V2 can be run over the same footage and compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

# Frame states. "ok" means the frame was searched for candidates.
STATE_OK = "ok"
STATE_DARK = "dark"
STATE_NOFEED = "nofeed"
STATE_NO_SPACE = "no_space"
STATE_ERROR = "error"


class PipelineError(RuntimeError):
    """A pipeline could not be built or could not process a frame."""


@dataclass(frozen=True)
class Region:
    """A scene region found by the region model."""

    label: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float


@dataclass(frozen=True)
class Detection:
    """A candidate object that the pipeline accepted on this frame."""

    box: tuple[int, int, int, int]  # x, y, w, h
    metrics: dict[str, Any] = field(default_factory=dict)
    track: int | None = None  # id of the track it belongs to, when the pipeline tracks


@dataclass(frozen=True)
class TrackSummary:
    """A track that has ended."""

    id: int
    confirmed: bool  # False for a transient: it ended before it persisted long enough
    first_index: int  # frame it was first seen on
    last_index: int  # frame it was last seen on
    frames_seen: int  # frames it was seen on
    first_box: tuple[int, int, int, int]
    last_box: tuple[int, int, int, int]
    max_contrast: float | None


@dataclass(frozen=True)
class Rejected:
    """A candidate object that failed one or more gates on this frame."""

    box: tuple[int, int, int, int]  # x, y, w, h
    gates: tuple[str, ...]  # every gate it failed, in the order they are checked
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameResult:
    index: int
    state: str
    regions: list[Region] = field(default_factory=list)
    # Accepted candidates. With tracking, only those on a confirmed track.
    detections: list[Detection] = field(default_factory=list)
    # Accepted candidates whose track has not persisted long enough yet.
    pending: list[Detection] = field(default_factory=list)
    confirmed_tracks: list[int] = field(default_factory=list)  # confirmed on this frame
    ended_tracks: list[TrackSummary] = field(default_factory=list)  # ended on this frame
    rejected: list[Rejected] = field(default_factory=list)
    faint: int = 0  # candidates too faint to pass, counted without being measured
    notes: dict[str, Any] = field(default_factory=dict)


class Pipeline(Protocol):
    name: str

    def describe(self) -> dict[str, Any]:
        """Everything needed to reproduce a run: versions, weights, parameter values."""
        ...

    def process(self, frame: np.ndarray, index: int) -> FrameResult:
        """Process one BGR frame as delivered by the source (uncropped)."""
        ...

    def close(self) -> None: ...
