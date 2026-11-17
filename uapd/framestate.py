"""Frame state: is this frame worth searching?

For now the state comes from the region model, as in V1: a large ``td`` (total
darkness) region or any ``nofeed`` region stops the search. The plan replaces
this with a cheap per-frame histogram check once there is footage to tune it on.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from uapd.config import RegionConfig
from uapd.pipeline import STATE_DARK, STATE_NO_SPACE, STATE_NOFEED, STATE_OK, Region


def frame_state(
    regions: Sequence[Region], shape: tuple[int, int], config: RegionConfig
) -> tuple[str, dict[str, Any]]:
    """Return the state of a frame of ``shape`` (height, width) and the values behind it."""
    frame_area = shape[0] * shape[1]
    for region in regions:
        if region.label == "td":
            x1, y1, x2, y2 = region.box
            ratio = (x2 - x1) * (y2 - y1) / frame_area
            if ratio > config.darkness_area_threshold:
                return STATE_DARK, {"darkness_area_ratio": ratio}
    labels = {region.label for region in regions}
    if "nofeed" in labels:
        return STATE_NOFEED, {}
    if "space" not in labels:
        return STATE_NO_SPACE, {}
    return STATE_OK, {}
