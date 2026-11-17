"""The V2 core pipeline: crop, regions, frame state, candidates.

Per-frame only; tracking and events come in the next milestone. Differences from
V1 that are visible in the output:

* The region model runs on the first frame and then every ``cycle_frames``, in
  step with the frames, so a run is repeatable. V1 has no regions until frame 120.
  Between runs the regions are reused; ``regions_age`` records how stale they are.
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
    Region,
    Rejected,
)
from uapd.regions import RegionModel


class CorePipeline:
    name = "core"

    def __init__(
        self,
        config: Config,
        region_model: RegionModel,
        avoid_boxes: Sequence[Box] = (),
    ):
        if config.regions.cycle_frames < 1:
            raise PipelineError("regions.cycle_frames must be at least 1")
        self.config = config
        self.region_model = region_model
        self.avoid_boxes = tuple(avoid_boxes)
        self._frames_seen = 0
        self._regions: list[Region] = []
        self._regions_frame: int | None = None  # count of the frame the regions came from
        self._search_area: SearchArea | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "coordinates": "cropped frame",
            "config_fingerprint": self.config.fingerprint(),
            "config": self.config.to_dict(),
            "region_model": self.region_model.describe(),
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
        cropped = frame[:, crop.crop_left : width - crop.crop_right]
        shape = cropped.shape[:2]

        count = self._frames_seen
        self._frames_seen += 1
        refreshed = (
            self._regions_frame is None
            or count - self._regions_frame >= self.config.regions.cycle_frames
            or self._search_area is None
            or self._search_area.mask.shape != shape
        )
        if refreshed:
            self._regions = self.region_model.detect(cropped)
            self._regions_frame = count
            space = [r.box for r in self._regions if r.label == "space"]
            self._search_area = build_search_area(shape, space)

        regions = list(self._regions)
        state, notes = frame_state(regions, shape, self.config.regions)
        notes["regions_age"] = count - self._regions_frame
        notes["lens_flares"] = sum(1 for r in regions if r.label == "lf")
        result = FrameResult(index=index, state=state, regions=regions, notes=notes)
        if state != STATE_OK:
            return result

        region_boxes: dict[str, list[Box]] = {}
        for region in regions:
            region_boxes.setdefault(region.label, []).append(region.box)
        found = find_candidates(
            cropped, self._search_area, self.config.candidates, region_boxes, self.avoid_boxes
        )
        for candidate in found.candidates:
            if candidate.accepted:
                result.detections.append(Detection(box=candidate.box, metrics=candidate.metrics))
            else:
                result.rejected.append(
                    Rejected(box=candidate.box, gates=candidate.failed, metrics=candidate.metrics)
                )
        if found.truncated:
            notes["candidates_found"] = found.found
            notes["candidates_measured"] = len(found.candidates)
        return result

    def close(self) -> None:
        self.region_model.close()


def build_core_pipeline(
    weights: str | Path = "best_map_model.pth",
    config: Config | None = None,
) -> CorePipeline:
    """The core pipeline with the V1 Faster R-CNN as its region model."""
    from uapd.regions.rcnn import RcnnRegionModel

    config = config or Config()
    return CorePipeline(config, RcnnRegionModel(weights, config.regions))
