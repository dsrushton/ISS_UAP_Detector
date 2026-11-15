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


@dataclass
class FrameResult:
    index: int
    state: str
    regions: list[Region] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
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
