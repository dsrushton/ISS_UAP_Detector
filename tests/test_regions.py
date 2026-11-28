from __future__ import annotations

import numpy as np
import pytest

from uapd.config import RegionConfig
from uapd.framestate import frame_state
from uapd.pipeline import STATE_DARK, STATE_NO_SPACE, STATE_NOFEED, STATE_OK, PipelineError, Region
from uapd.regions.rcnn import CLASS_NAMES, RcnnRegionModel, select_regions

SHAPE = (720, 939)


def test_select_regions_applies_class_thresholds():
    boxes = [[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30], [0, 0, 40, 40]]
    labels = [CLASS_NAMES.index(n) for n in ("space", "space", "lf", "td")]
    scores = [0.79, 0.95, 0.98, 0.30]
    regions = select_regions(boxes, labels, scores, RegionConfig())
    # space needs 0.80, lf needs 0.99, td needs 0.25
    assert [(r.label, r.score) for r in regions] == [("space", 0.95), ("td", 0.30)]
    assert regions[0].box == (0.0, 0.0, 20.0, 20.0)


def test_select_regions_keeps_the_best_max_boxes():
    labels = [CLASS_NAMES.index("panel")] * 12
    scores = [0.80 + i / 100 for i in range(12)]
    regions = select_regions([[0, 0, 1, 1]] * 12, labels, scores, RegionConfig())
    assert len(regions) == 10
    assert regions[0].score == pytest.approx(0.91)


def test_missing_weights_is_reported(tmp_path):
    with pytest.raises(PipelineError, match="model weights not found"):
        RcnnRegionModel(tmp_path / "missing.pth", RegionConfig())


def test_unreadable_weights_are_reported(tmp_path):
    pytest.importorskip("torchvision")
    weights = tmp_path / "bad.pth"
    weights.write_bytes(b"not a checkpoint")
    with pytest.raises(PipelineError, match="could not load model weights"):
        RcnnRegionModel(weights, RegionConfig(), device="cpu")


def _state(*regions: Region):
    return frame_state(regions, SHAPE, RegionConfig())


def test_frame_states():
    space = Region("space", (0, 0, 939, 720), 0.9)
    assert _state(space) == (STATE_OK, {})
    assert _state() == (STATE_NO_SPACE, {})
    assert _state(Region("earth", (0, 0, 939, 720), 0.9)) == (STATE_NO_SPACE, {})
    assert _state(space, Region("nofeed", (0, 0, 10, 10), 0.5)) == (STATE_NOFEED, {})


def test_darkness_needs_a_large_td_region():
    space = Region("space", (0, 0, 939, 720), 0.9)
    small = Region("td", (0, 0, 300, 300), 0.5)
    large = Region("td", (0, 0, 939, 360), 0.5)
    assert _state(space, small)[0] == STATE_OK
    state, notes = _state(space, large)
    assert state == STATE_DARK
    assert notes == {"darkness_area_ratio": 0.5}


def test_two_models_combined_each_supply_their_classes():
    from uapd.pipeline import Region
    from uapd.regions.combined import CombinedRegionModel

    class Fake:
        def __init__(self, name, regions):
            self.name, self.regions, self.closed = name, regions, False

        def describe(self):
            return {"kind": self.name}

        def detect(self, frame):
            return list(self.regions)

        def close(self):
            self.closed = True

    shape = ((0.0, 0.0), (9.0, 0.0), (0.0, 9.0))
    outlines = Fake(
        "outlines",
        [Region("space", (0, 0, 9, 9), 0.9, shape), Region("lf", (1, 1, 2, 2), 0.95)],
    )
    boxes = Fake(
        "boxes",
        [
            Region("lf", (3, 3, 5, 5), 0.99),
            Region("space", (0, 0, 9, 9), 1.0),
            Region("td", (0, 0, 9, 9), 0.5),
        ],
    )
    model = CombinedRegionModel(outlines, boxes)
    found = model.detect(np.zeros((10, 10, 3), np.uint8))
    assert [(r.label, r.score, bool(r.outline)) for r in found] == [
        ("lf", 0.99, False),  # the box model's flare, not the outline model's
        ("space", 0.9, True),  # the outline model's space, not the box model's
        ("td", 0.5, False),
    ]
    assert model.describe()["first"] == {"kind": "outlines"}
    model.close()
    assert outlines.closed and boxes.closed
