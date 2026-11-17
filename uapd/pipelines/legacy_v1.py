"""Runs the V1 detector (legacy/SOD_Detections.py) behind the Pipeline interface.

This is the baseline every V2 stage is compared against. ``legacy/`` holds V1
with the July 2026 patch set (scale fix, flare guard, truthful area); the
unpatched original is on the ``dev`` branch. This adapter does not alter the V1
code, it only controls how it is driven:

* V1 runs the RCNN in a background thread and uses whatever result has arrived.
  With ``sync=True`` (the default) the adapter waits for each RCNN run to finish
  before the next frame, so a run over the same footage always gives the same
  output. The RCNN result therefore applies from the frame after it was queued,
  where a live V1 run would pick it up a few frames later.
* Display, recording and saving are not involved; only detection runs.

Boxes are in V1's cropped-frame coordinates.
"""

from __future__ import annotations

import contextlib
import hashlib
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from uapd.pipeline import (
    STATE_DARK,
    STATE_ERROR,
    STATE_NO_SPACE,
    STATE_NOFEED,
    STATE_OK,
    Detection,
    FrameResult,
    PipelineError,
    Region,
)

DEFAULT_LEGACY_DIR = Path(__file__).resolve().parents[2] / "legacy"
RCNN_WAIT_SECONDS = 300.0


class LegacyV1Pipeline:
    name = "legacy-v1"

    def __init__(
        self,
        weights: str | Path = "best_map_model.pth",
        legacy_dir: str | Path | None = None,
        rcnn_cycle: int | None = None,
        sync: bool = True,
    ):
        self.weights = Path(weights)
        self.legacy_dir = Path(legacy_dir) if legacy_dir else DEFAULT_LEGACY_DIR
        self.sync = sync
        if not self.weights.is_file():
            raise PipelineError(
                f"model weights not found: {self.weights} "
                "(download link is in README.md; pass the path with --weights)"
            )
        if not (self.legacy_dir / "SOD_Detections.py").is_file():
            raise PipelineError(f"V1 code not found in {self.legacy_dir}")

        if str(self.legacy_dir) not in sys.path:
            sys.path.insert(0, str(self.legacy_dir))

        # V1 prints progress to stdout; keep stdout clean for the report.
        with contextlib.redirect_stdout(sys.stderr):
            try:
                import SOD_Constants
            except ImportError as exc:
                raise PipelineError(
                    f"the legacy pipeline needs PyTorch and torchvision: {exc} "
                    "(install with: pip install -e '.[torch]')"
                ) from exc

            # V1 modules copy constants at import, so set them before importing the
            # detector and again on the module in case it was imported earlier.
            SOD_Constants.update_value("MODEL_PATH", str(self.weights))
            if rcnn_cycle is not None:
                if rcnn_cycle < 1:
                    raise PipelineError("rcnn_cycle must be at least 1")
                SOD_Constants.update_value("RCNN_DETECTION_CYCLE", int(rcnn_cycle))

            import SOD_Detections

            SOD_Detections.MODEL_PATH = str(self.weights)
            self._constants = SOD_Constants
            self._detector = SOD_Detections.SpaceObjectDetector()
            if not self._detector.initialize_model():
                self._detector.cleanup()
                raise PipelineError(f"V1 could not load the model from {self.weights}")

        self._rcnn_done = threading.Event()
        self._wrap_worker_completion()

    def _wrap_worker_completion(self) -> None:
        """Signal when the RCNN worker has stored a result and updated darkness state."""
        detector = self._detector
        original = detector._update_darkness_state

        def update_darkness_state(frame: np.ndarray) -> None:
            original(frame)
            if threading.current_thread() is detector.rcnn_worker_thread:
                self._rcnn_done.set()

        detector._update_darkness_state = update_darkness_state

    def describe(self) -> dict[str, Any]:
        const = self._constants.const
        constants = {
            name: _plain(getattr(const, name))
            for name in sorted(dir(const))
            if name.isupper() and _is_plain(getattr(const, name))
        }
        return {
            "name": self.name,
            "weights": str(self.weights),
            "weights_sha256": _sha256(self.weights),
            "device": str(const.DEVICE),
            "sync": self.sync,
            "coordinates": "cropped frame",
            "constants": constants,
        }

    def process(self, frame: np.ndarray, index: int) -> FrameResult:
        const = self._constants.const
        if frame.shape[1] <= const.CROP_LEFT + const.CROP_RIGHT:
            raise PipelineError(
                f"frame {index} is {frame.shape[1]} px wide; V1 crops "
                f"{const.CROP_LEFT}+{const.CROP_RIGHT} px and expects the 1280 px feed"
            )

        self._rcnn_done.clear()
        with contextlib.redirect_stdout(sys.stderr):
            results = self._detector.process_frame(frame)
        if results is None:
            return FrameResult(index=index, state=STATE_ERROR)

        if self.sync and results.metadata.get("rcnn_frame"):
            if not self._rcnn_done.wait(RCNN_WAIT_SECONDS):
                raise PipelineError(f"RCNN run queued at frame {index} did not finish")

        regions = [
            Region(label=label, box=tuple(float(v) for v in box), score=float(score))
            for label, boxes in results.rcnn_boxes.items()
            for box, score in zip(boxes, results.rcnn_scores.get(label, []), strict=False)
        ]

        if results.darkness_detected:
            state = STATE_DARK
        elif "nofeed" in results.rcnn_boxes:
            state = STATE_NOFEED
        elif not results.rcnn_boxes.get("space"):
            state = STATE_NO_SPACE
        else:
            state = STATE_OK

        metrics = results.metadata.get("anomaly_metrics", [])
        detections = [
            Detection(
                box=tuple(int(v) for v in box),
                metrics=dict(metrics[i]) if i < len(metrics) else {},
            )
            for i, box in enumerate(results.anomalies)
        ]

        notes = {
            key: results.metadata[key]
            for key in ("skip_save", "rcnn_frame", "rcnn_queue_full", "darkness_area_ratio")
            if key in results.metadata
        }
        return FrameResult(
            index=index, state=state, regions=regions, detections=detections, notes=notes
        )

    def close(self) -> None:
        self._detector.cleanup()


def _is_plain(value: Any) -> bool:
    if isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_plain(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_plain(v) for k, v in value.items())
    return False


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
