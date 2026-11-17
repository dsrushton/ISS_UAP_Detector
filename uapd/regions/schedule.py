"""When the region model runs, and which regions a frame uses.

Regions are refreshed every ``cycle_frames`` and reused in between.

* ``SyncRegions`` runs the model in step with the frames. A run over the same
  footage always gives the same output; used by ``uapd eval``.
* ``BackgroundRegions`` runs the model in a worker thread with a one-frame queue,
  so a live loop never waits for it (except for the very first result).
"""

from __future__ import annotations

import queue
import threading
from typing import Protocol

import numpy as np

from uapd.pipeline import PipelineError, Region
from uapd.regions import RegionModel

FIRST_RESULT_TIMEOUT = 120.0


class RegionSchedule(Protocol):
    def update(
        self, frame: np.ndarray, count: int, force: bool = False
    ) -> tuple[list[Region], int]:
        """Regions to use for frame number ``count``, and the frame they were computed from.

        ``force`` computes them from this frame before returning.
        """
        ...

    def close(self) -> None: ...


class SyncRegions:
    def __init__(self, model: RegionModel, cycle_frames: int):
        self.model = model
        self.cycle_frames = cycle_frames
        self._regions: list[Region] = []
        self._source: int | None = None

    def update(
        self, frame: np.ndarray, count: int, force: bool = False
    ) -> tuple[list[Region], int]:
        if force or self._source is None or count - self._source >= self.cycle_frames:
            self._regions = self.model.detect(frame)
            self._source = count
        return self._regions, self._source

    def close(self) -> None:
        self.model.close()


class BackgroundRegions:
    def __init__(self, model: RegionModel, cycle_frames: int):
        self.model = model
        self.cycle_frames = cycle_frames
        self._lock = threading.Lock()
        self._regions: list[Region] = []
        self._source: int | None = None
        self._submitted: int | None = None
        self._error: BaseException | None = None
        self._first = threading.Event()
        self._queue: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._work, name="uapd-regions", daemon=True)
        self._thread.start()

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame, count = item
            try:
                regions = self.model.detect(frame)
            except BaseException as exc:  # reported to the caller on its next update
                self._error = exc
                self._first.set()
                return
            with self._lock:
                # A forced result from a later frame must not be replaced by an older one.
                if self._source is None or count >= self._source:
                    self._regions, self._source = regions, count
            self._first.set()

    def update(
        self, frame: np.ndarray, count: int, force: bool = False
    ) -> tuple[list[Region], int]:
        if self._error is not None:
            raise PipelineError(f"region model failed: {self._error}") from self._error
        if force:
            regions = self.model.detect(frame)
            with self._lock:
                self._regions, self._source = regions, count
            self._submitted = count
            self._first.set()
        elif self._submitted is None or count - self._submitted >= self.cycle_frames:
            try:
                # The frame buffer is reused by the caller, so the worker gets a copy.
                self._queue.put_nowait((frame.copy(), count))
                self._submitted = count
            except queue.Full:
                pass  # still busy with the previous frame; try again on the next one
        if not self._first.is_set() and not self._first.wait(FIRST_RESULT_TIMEOUT):
            raise PipelineError("region model produced no result")
        if self._error is not None:
            raise PipelineError(f"region model failed: {self._error}") from self._error
        with self._lock:
            return self._regions, self._source

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put(None)
        self._thread.join(timeout=10.0)
        self.model.close()
