"""The V2 core pipeline: crop, regions, frame state, candidates, tracks.

Differences from V1 that are visible in the output:

* Accepted candidates are linked into tracks. Only candidates on a track that has
  persisted (``tracking.confirm_frames``) are detections; the rest are pending,
  and a track that ends unconfirmed is reported as a transient. See
  ``uapd.tracking``.

* The region model runs on the first frame and then every ``cycle_frames``. V1 has
  no regions until frame 120. By default it runs in step with the frames, so a run
  is repeatable; with ``background=True`` (the live app) it runs in a worker
  thread. Between runs the regions are reused; ``regions_age`` records how stale
  they are.
* Candidates are measured at full resolution (see ``uapd.candidates``).
* Rejected candidates are returned with the gates they failed.
* Lens flares are counted, not acted on. V1's guard never did anything, and the
  plan decides flare handling from labelled data.
* No limit on accepted candidates per frame (V1 stops at 2).

Boxes are in cropped-frame coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from uapd.candidates import Box, SearchArea, build_search_area, find_candidates
from uapd.config import Config
from uapd.framestate import frame_state
from uapd.pipeline import (
    STATE_OK,
    Detection,
    FrameResult,
    PipelineError,
    Rejected,
    TrackSummary,
)
from uapd.regions import RegionModel
from uapd.regions.schedule import BackgroundRegions, SyncRegions
from uapd.tracking import Tracker


class CorePipeline:
    name = "core"

    def __init__(
        self,
        config: Config,
        region_model: RegionModel,
        avoid_boxes: Sequence[Box] = (),
        background: bool = False,
    ):
        if config.regions.cycle_frames < 1:
            raise PipelineError("regions.cycle_frames must be at least 1")
        self.config = config
        self.region_model = region_model
        self.avoid_boxes = tuple(avoid_boxes)
        self.background = background
        schedule = BackgroundRegions if background else SyncRegions
        self._schedule = schedule(region_model, config.regions.cycle_frames)
        self._tracker = Tracker(config.tracking)
        self._frames_seen = 0
        self._area_source: int | None = None  # frame the search area was built from
        self.search_area: SearchArea | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "coordinates": "cropped frame",
            "config_fingerprint": self.config.fingerprint(),
            "config": self.config.to_dict(),
            "region_model": self.region_model.describe(),
            "regions_in_background": self.background,
            "avoid_boxes": [list(box) for box in self.avoid_boxes],
        }

    def process(self, frame: np.ndarray, index: int) -> FrameResult:
        crop = self.config.frame
        width = frame.shape[1]
        if width <= crop.crop_left + crop.crop_right:
            raise PipelineError(
                f"frame {index} is {width} px wide; the configured crop removes "
                f"{crop.crop_left}+{crop.crop_right} px"
            )
        return self.process_cropped(frame[:, crop.crop_left : width - crop.crop_right], index)

    def process_cropped(
        self,
        cropped: np.ndarray,
        index: int,
        avoid_boxes: Sequence[Box] | None = None,
        refresh_regions: bool = False,
    ) -> FrameResult:
        """Process a frame that is already cropped.

        ``avoid_boxes`` replaces the pipeline's own for this frame.
        ``refresh_regions`` runs the region model on this frame first.
        """
        shape = cropped.shape[:2]
        count = self._frames_seen
        self._frames_seen += 1

        resized = self.search_area is not None and self.search_area.mask.shape != shape
        regions, source = self._schedule.update(cropped, count, force=refresh_regions or resized)
        if self.search_area is None or source != self._area_source:
            space = [r.box for r in regions if r.label == "space"]
            self.search_area = build_search_area(shape, space)
            self._area_source = source

        regions = list(regions)
        state, notes = frame_state(regions, shape, self.config.regions)
        notes["regions_age"] = count - source
        notes["lens_flares"] = sum(1 for r in regions if r.label == "lf")
        result = FrameResult(index=index, state=state, regions=regions, notes=notes)

        accepted: list[Detection] = []
        if state == STATE_OK:
            region_boxes: dict[str, list[Box]] = {}
            for region in regions:
                region_boxes.setdefault(region.label, []).append(region.box)
            found = find_candidates(
                cropped,
                self.search_area,
                self.config.candidates,
                region_boxes,
                self.avoid_boxes if avoid_boxes is None else avoid_boxes,
            )
            for candidate in found.candidates:
                if candidate.accepted:
                    accepted.append(Detection(box=candidate.box, metrics=candidate.metrics))
                else:
                    result.rejected.append(
                        Rejected(
                            box=candidate.box, gates=candidate.failed, metrics=candidate.metrics
                        )
                    )
            result.faint = found.faint
            if found.truncated:
                notes["candidates_found"] = found.found
                notes["candidates_measured"] = len(found.candidates)

        # Frames that are not searched still move the tracker on, so tracks can end.
        tracked = self._tracker.update(accepted, index)
        result.detections = tracked.detections
        result.pending = tracked.pending
        result.confirmed_tracks = tracked.confirmed
        result.ended_tracks = tracked.ended
        return result

    def finish(self) -> list[TrackSummary]:
        """End the open tracks at the end of a clip and return them."""
        return self._tracker.finish()

    def close(self) -> None:
        self._schedule.close()


def build_core_pipeline(
    weights: str | Path = "best_map_model.pth",
    config: Config | None = None,
    background: bool = False,
) -> CorePipeline:
    """The core pipeline with the region model in ``weights``."""
    from uapd.regions import load_region_model

    config = config or Config()
    model = load_region_model(weights, config.regions)
    return CorePipeline(config, model, background=background)
