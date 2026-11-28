"""Two region models used as one: each supplies the classes it is good at.

The first model gives every class except those named for the second. The usual
pairing is an outline model (``yolo-seg``) first, for the large shapes - space,
Earth, the station - and a box model (``rcnn-boxes``) second, for lens flares,
the sun, panels and the two whole-frame states, total darkness and no feed.
Both run on the same frame, so their regions belong to the same moment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uapd.pipeline import Region
from uapd.regions import RegionModel

# What the second model supplies unless told otherwise.
SECOND_CLASSES = ("lf", "td", "sun", "nofeed", "panel")


class CombinedRegionModel:
    def __init__(
        self,
        first: RegionModel,
        second: RegionModel,
        second_classes: tuple[str, ...] = SECOND_CLASSES,
    ):
        self.first = first
        self.second = second
        self.second_classes = tuple(second_classes)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "combined",
            "first": self.first.describe(),
            "second": self.second.describe(),
            "second_classes": list(self.second_classes),
        }

    def detect(self, frame: np.ndarray) -> list[Region]:
        regions = [r for r in self.first.detect(frame) if r.label not in self.second_classes]
        regions += [r for r in self.second.detect(frame) if r.label in self.second_classes]
        regions.sort(key=lambda r: r.score, reverse=True)
        return regions

    def close(self) -> None:
        self.first.close()
        self.second.close()
