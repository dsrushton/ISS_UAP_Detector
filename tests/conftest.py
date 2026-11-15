from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from uapd.pipeline import STATE_OK, Detection, FrameResult, Region


class FakePipeline:
    """Flags a frame when its brightest pixel is above 100."""

    name = "fake"

    def __init__(self) -> None:
        self.closed = False

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}

    def process(self, frame: np.ndarray, index: int) -> FrameResult:
        result = FrameResult(
            index=index,
            state=STATE_OK,
            regions=[Region("space", (0.0, 0.0, 10.0, 10.0), 0.9)],
        )
        if frame.max() > 100:
            # numpy scalars on purpose: reports must serialise them.
            result.detections.append(
                Detection(box=(1, 2, 3, 4), metrics={"contrast": np.float32(42.5)})
            )
        return result

    def close(self) -> None:
        self.closed = True


class ArraySource:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames

    def describe(self) -> dict[str, Any]:
        return {"kind": "array", "frames": len(self._frames)}

    def frames(self):
        yield from enumerate(self._frames)


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def five_frames() -> ArraySource:
    """Five small frames; indices 1 and 3 are bright."""
    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(5)]
    frames[1][:] = 200
    frames[3][:] = 200
    return ArraySource(frames)
