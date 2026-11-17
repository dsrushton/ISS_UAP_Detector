from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from uapd.config import Config
from uapd.eval import run_eval
from uapd.pipeline import PipelineError, Region
from uapd.pipelines.core import CorePipeline

CONFIG = Config.from_dict({"regions": {"cycle_frames": 5}})
CROP_LEFT = CONFIG.frame.crop_left
CROPPED_WIDTH = 1280 - CROP_LEFT - CONFIG.frame.crop_right


class FakeRegionModel:
    """Returns a fixed region list and counts how often it is asked."""

    def __init__(self, regions: list[Region] | None = None):
        self.regions = regions
        self.calls = 0
        self.closed = False

    def describe(self) -> dict[str, Any]:
        return {"kind": "fake"}

    def detect(self, frame: np.ndarray) -> list[Region]:
        self.calls += 1
        if self.regions is not None:
            return self.regions
        height, width = frame.shape[:2]
        return [Region("space", (0.0, 0.0, float(width), float(height)), 0.99)]

    def close(self) -> None:
        self.closed = True


class Frames:
    def __init__(self, frames):
        self._frames = frames

    def describe(self):
        return {"kind": "array", "frames": len(self._frames)}

    def frames(self):
        yield from enumerate(self._frames)


def feed_frame(*squares: tuple[int, int, int, int]) -> np.ndarray:
    """A black 1280x720 feed frame with filled squares (x, y, size, brightness)."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    for x, y, size, value in squares:
        frame[y : y + size, x : x + size] = value
    return frame


def test_detects_from_the_first_frame_in_cropped_coordinates():
    model = FakeRegionModel()
    pipeline = CorePipeline(CONFIG, model)
    report = run_eval(pipeline, Frames([feed_frame((640, 300, 12, 150))] * 12), frame_mode="all")

    assert report["summary"]["states"] == {"ok": 12}
    assert report["summary"]["detections"] == 12
    assert report["frames"][0]["detections"][0]["box"] == [640 - CROP_LEFT, 300, 12, 12]
    # Regions are refreshed on frames 0, 5 and 10 and reused in between.
    assert model.calls == 3
    assert [f["notes"]["regions_age"] for f in report["frames"]] == [0, 1, 2, 3, 4] * 2 + [0, 1]


def test_rejected_candidates_reach_the_report():
    pipeline = CorePipeline(CONFIG, FakeRegionModel())
    frames = Frames([feed_frame((640, 300, 12, 150), (700, 400, 3, 150), (800, 500, 12, 20))] * 2)
    report = run_eval(pipeline, frames)

    assert report["summary"]["detections"] == 2
    assert report["summary"]["rejected"] == 4
    assert report["summary"]["rejected_by_gate"] == {"contrast": 2, "min_dimension": 2}
    rejected = report["frames"][0]["rejected"]
    assert [r["gates"] for r in rejected] == [["min_dimension", "min_area"], ["contrast"]]
    assert rejected[0]["metrics"]["width"] == 3
    assert rejected[1]["metrics"]["contrast"] == 20.0


def test_candidates_mode_lists_frames_with_only_rejections():
    pipeline = CorePipeline(CONFIG, FakeRegionModel())
    frames = Frames([feed_frame(), feed_frame((700, 400, 3, 150))])
    assert run_eval(pipeline, frames)["frames"] == []
    listed = run_eval(pipeline, frames, frame_mode="candidates")["frames"]
    assert [f["index"] for f in listed] == [1]


def test_frames_without_space_are_not_searched():
    earth = [Region("earth", (0.0, 0.0, 939.0, 720.0), 0.9)]
    pipeline = CorePipeline(CONFIG, FakeRegionModel(earth))
    report = run_eval(pipeline, Frames([feed_frame((640, 300, 12, 150))] * 3))
    assert report["summary"]["states"] == {"no_space": 3}
    assert report["summary"]["detections"] == 0
    assert report["summary"]["rejected"] == 0


def test_lens_flares_are_counted_and_overlaps_rejected():
    flare = (400.0, 250.0, 520.0, 350.0)  # covers the object at cropped x=475
    regions = [Region("space", (0.0, 0.0, 939.0, 720.0), 0.99)] + [
        Region("lf", flare, 0.995) for _ in range(3)
    ]
    pipeline = CorePipeline(CONFIG, FakeRegionModel(regions))
    frame = feed_frame((640, 300, 12, 150), (900, 500, 12, 150))
    report = run_eval(pipeline, Frames([frame]), frame_mode="all")

    record = report["frames"][0]
    assert record["notes"]["lens_flares"] == 3
    # Three flares do not stop the search; the object clear of them is still found.
    assert [d["box"][:2] for d in record["detections"]] == [[900 - CROP_LEFT, 500]]
    assert [r["gates"] for r in record["rejected"]] == [["overlap_lf"]]


def test_avoid_boxes_are_applied_and_described():
    pipeline = CorePipeline(CONFIG, FakeRegionModel(), avoid_boxes=[(400.0, 250.0, 520.0, 350.0)])
    report = run_eval(pipeline, Frames([feed_frame((640, 300, 12, 150))]))
    assert report["summary"]["rejected_by_gate"] == {"avoid_box": 1}
    assert report["pipeline"]["avoid_boxes"] == [[400.0, 250.0, 520.0, 350.0]]


def test_describe_records_config_and_model():
    info = CorePipeline(CONFIG, FakeRegionModel()).describe()
    assert info["config_fingerprint"] == CONFIG.fingerprint()
    assert info["config"]["regions"]["cycle_frames"] == 5
    assert info["region_model"] == {"kind": "fake"}


def test_runs_are_repeatable():
    frames = Frames([feed_frame((640, 300, 12, 150), (700, 400, 3, 150))] * 7)
    reports = []
    for _ in range(2):
        report = run_eval(CorePipeline(CONFIG, FakeRegionModel()), frames, frame_mode="all")
        report.pop("run")
        reports.append(report)
    assert reports[0] == reports[1]


def test_narrow_frames_are_rejected():
    pipeline = CorePipeline(CONFIG, FakeRegionModel())
    with pytest.raises(PipelineError, match="configured crop"):
        pipeline.process(np.zeros((720, 300, 3), np.uint8), 0)


def test_close_closes_the_region_model():
    model = FakeRegionModel()
    CorePipeline(CONFIG, model).close()
    assert model.closed


def test_core_finds_what_v1_finds_plus_small_objects(tmp_path, monkeypatch):
    """The M1 check in miniature: same frames through V1 and the core pipeline."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import sys

    from uapd.pipelines.legacy_v1 import DEFAULT_LEGACY_DIR, LegacyV1Pipeline

    if str(DEFAULT_LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(DEFAULT_LEGACY_DIR))
    import SOD_Detections

    def fake_rcnn(self, frame, is_gpu_tensor=False):
        height, width = frame.shape[:2]
        return {
            "boxes": {"space": [[0.0, 0.0, float(width), float(height)]]},
            "scores": {"space": [0.99]},
        }

    monkeypatch.setattr(SOD_Detections.SpaceObjectDetector, "initialize_model", lambda self: True)
    monkeypatch.setattr(SOD_Detections.SpaceObjectDetector, "_run_rcnn_detection", fake_rcnn)
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"stub")

    frame = np.zeros((720, 1280, 3), np.uint8)
    cv2.circle(frame, (500, 300), 10, (150, 150, 150), -1)  # large: both find it
    frame[400:405, 800:805] = 150  # 5x5 px: exactly the configured minimum
    frames = Frames([frame] * 8)

    v1 = LegacyV1Pipeline(weights=weights, rcnn_cycle=5)
    try:
        v1_report = run_eval(v1, frames, frame_mode="all")
    finally:
        v1.close()
    core_report = run_eval(CorePipeline(CONFIG, FakeRegionModel()), frames, frame_mode="all")

    def centres(report, index):
        return sorted(
            (d["box"][0] + d["box"][2] / 2, d["box"][1] + d["box"][3] / 2)
            for d in report["frames"][index]["detections"]
        )

    def same(found, expected):
        return len(found) == len(expected) and all(
            abs(f[0] - e[0]) <= 2 and abs(f[1] - e[1]) <= 2
            for f, e in zip(found, expected, strict=True)
        )

    large, small = (500 - CROP_LEFT, 300), (802.5 - CROP_LEFT, 402.5)
    assert same(centres(v1_report, 7), [large])
    assert same(centres(core_report, 7), [large, small])
    # V1 has no regions for its first five frames; the core pipeline starts at once.
    assert centres(v1_report, 0) == []
    assert same(centres(core_report, 0), [large, small])
