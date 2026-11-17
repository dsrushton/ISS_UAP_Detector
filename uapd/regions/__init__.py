"""Region models: anything that labels the scene regions of a cropped frame."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from uapd.pipeline import Region


class RegionModel(Protocol):
    def describe(self) -> dict[str, Any]:
        """Model identity: kind, weights file and hash, device."""
        ...

    def detect(self, frame: np.ndarray) -> list[Region]:
        """Regions of one cropped BGR frame, already filtered by score."""
        ...

    def close(self) -> None: ...
