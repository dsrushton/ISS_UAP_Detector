"""The live app: V2 detection behind the V1 window, recorder and YouTube restream.

The main loop, display, recording, capture and restream are the modules in
``legacy/``, used as they are. Only the detector is replaced: ``CoreDetector``
runs the V2 core pipeline and hands the legacy code results in the shape it
expects. A recording starts only for a confirmed track, so brief flashes no
longer trigger one. Each recording gets a data file beside it (``uapd.events``).

Run from the repository root; the legacy modules use paths relative to it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from uapd import __version__
from uapd.config import Config
from uapd.eval.headless import _git_commit
from uapd.events import EventLog
from uapd.pipeline import STATE_DARK, FrameResult, PipelineError
from uapd.pipelines.core import CorePipeline, build_core_pipeline

LEGACY_DIR = Path(__file__).resolve().parents[2] / "legacy"


class CoreDetector:
    """Stands in for V1's ``SpaceObjectDetector`` in the legacy main loop."""

    def __init__(self, pipeline_factory, results_class: type):
        self._pipeline_factory = pipeline_factory
        self._results_class = results_class
        self.pipeline: CorePipeline | None = None
        self.last_result: FrameResult | None = None
        self.transients = 0
        self._count = 0
        self._space_contours: tuple[Any, list] = (None, [])

    # --- the interface the legacy main loop uses -------------------------------------

    def set_logger(self, logger: Any) -> None:
        pass

    def initialize_model(self) -> bool:
        try:
            self.pipeline = self._pipeline_factory()
        except PipelineError as exc:
            print(f"Could not start the V2 pipeline: {exc}")
            return False
        info = self.pipeline.describe()
        print(f"V2 core pipeline ready (config {info['config_fingerprint']})")
        return True

    def process_frame(
        self,
        frame: np.ndarray,
        avoid_boxes: list | None = None,
        is_test_frame: bool = False,
        crop_left: int | None = None,
        crop_right: int | None = None,
    ):
        count = self._count
        self._count += 1
        if is_test_frame or crop_left is None or crop_right is None:
            cropped = frame  # test frames arrive already cropped
        else:
            cropped = frame[:, crop_left : frame.shape[1] - crop_right]
        result = self.pipeline.process_cropped(
            cropped, count, avoid_boxes=avoid_boxes or (), refresh_regions=is_test_frame
        )
        self.last_result = result
        for track in result.ended_tracks:
            if not track.confirmed:
                self.transients += 1
                x, y, w, h = track.first_box
                print(
                    f"Transient ignored: {w}x{h} px at ({x}, {y}), "
                    f"seen on {track.frames_seen} frame(s)"
                )
        for track_id in result.confirmed_tracks:
            print(f"Track {track_id} confirmed")
        return self._to_legacy(result, count, is_test_frame)

    def cleanup(self) -> None:
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None

    # ---------------------------------------------------------------------------------

    def _to_legacy(self, result: FrameResult, count: int, is_test_frame: bool):
        boxes: dict[str, list] = {}
        scores: dict[str, list] = {}
        for region in result.regions:
            boxes.setdefault(region.label, []).append(list(region.box))
            scores.setdefault(region.label, []).append(region.score)

        out = self._results_class(frame_number=count, rcnn_boxes=boxes, rcnn_scores=scores)
        out.darkness_detected = result.state == STATE_DARK
        out.metadata = {"test_frame": is_test_frame, **result.notes}

        # Confirmed detections are the legacy "anomalies": they are boxed in red and
        # start a recording. Pending and rejected candidates are outlined only.
        out.anomalies = [d.box for d in result.detections]
        out.metadata["anomaly_metrics"] = [
            {**d.metrics, "contrast": d.metrics.get("contrast") or 0.0} for d in result.detections
        ]
        out.contours = [
            _outline(c.box) for c in (*result.detections, *result.pending, *result.rejected)
        ]

        space = sorted(boxes.get("space", []), key=lambda box: box[1])
        area = self.pipeline.search_area
        if space and area is not None:
            out.space_box = space[0]
            out.metadata["all_space_boxes"] = space
            out.space_mask = area.mask
            if self._space_contours[0] is not area:
                contours, _ = cv2.findContours(
                    area.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                self._space_contours = (area, list(contours))
            out.space_contours = self._space_contours[1]
        return out


def _outline(box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    corners = [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]]
    return np.array(corners, np.int32).reshape(-1, 1, 2)


def run_live(config: Config, weights: str | Path, source: str | None = None) -> None:
    """Run the live app until it is quit. ``source`` is a video file or a YouTube URL."""
    if not (LEGACY_DIR / "SOD_Main.py").is_file():
        raise PipelineError(f"legacy app not found in {LEGACY_DIR}")
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))

    # The legacy modules copy constants when they are imported, so set them first.
    import SOD_Constants

    SOD_Constants.update_value("CROP_LEFT", config.frame.crop_left)
    SOD_Constants.update_value("CROP_RIGHT", config.frame.crop_right)

    import SOD_Detections
    import SOD_Main

    def build() -> CorePipeline:
        return build_core_pipeline(weights, config, background=True)

    source = source or config.source.live_url
    is_file = Path(source).is_file()

    class LiveSystem(SOD_Main.SpaceObjectDetectionSystem):
        def __init__(self) -> None:
            super().__init__()
            self.detector.cleanup()  # the V1 detector the base class made
            self.detector = CoreDetector(build, SOD_Detections.DetectionResults)
            self.events: EventLog | None = None

        def process_frame(self, *args: Any, **kwargs: Any):
            outcome = super().process_frame(*args, **kwargs)
            result = self.detector.last_result
            if result is not None and self.video is not None:
                if self.events is None:
                    self.events = self._event_log()
                recording = bool(self.video.recording)
                number = getattr(self.video, "current_video_number", None)
                event_id = f"{number:05d}" if recording and number is not None else None
                written = len(self.events.written)
                self.events.observe(result, recording, event_id)
                for path in self.events.written[written:]:
                    print(f"Saved event data to {path}")
            return outcome

        def _event_log(self) -> EventLog:
            header = {
                "source": source,
                "fps": self.fps,
                "uapd_version": __version__,
                "commit": _git_commit(),
                "pipeline": self.detector.pipeline.describe(),
            }
            buffer_frames = int(config.capture.buffer_seconds * self.fps)
            return EventLog(SOD_Constants.VIDEO_SAVE_DIR, header, buffer_frames)

    system = LiveSystem()
    try:
        if not system.initialize():
            raise PipelineError("the live app could not start")
        print(f"\nConnecting to {source} ...")
        system.process_video_stream(source, is_youtube=not is_file)
    finally:
        if system.events is not None:
            system.events.close()
            for path in system.events.written[-1:]:
                print(f"Last event data: {path}")
        print(f"Transients ignored this run: {system.detector.transients}")
        system.cleanup()
