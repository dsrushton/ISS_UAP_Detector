"""The detector the live app gives the legacy main loop, with the region model faked."""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from uapd.config import Config  # noqa: E402
from uapd.pipelines.core import CorePipeline  # noqa: E402
from uapd.runtime.live import LEGACY_DIR, CoreDetector  # noqa: E402

from .test_core_pipeline import FakeRegionModel  # noqa: E402

CONFIG = Config()
LEFT, RIGHT = CONFIG.frame.crop_left, CONFIG.frame.crop_right


@pytest.fixture
def detector():
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    import SOD_Detections

    made = CoreDetector(
        lambda: CorePipeline(CONFIG, FakeRegionModel()), SOD_Detections.DetectionResults
    )
    assert made.initialize_model()
    yield made
    made.cleanup()


def feed(*squares):
    frame = np.zeros((720, 1280, 3), np.uint8)
    for x, y, size, value in squares:
        frame[y : y + size, x : x + size] = value
    return frame


def test_only_confirmed_tracks_reach_the_legacy_loop(detector, capsys):
    frame = feed((640, 300, 12, 150))
    first = detector.process_frame(frame, crop_left=LEFT, crop_right=RIGHT)
    second = detector.process_frame(frame, crop_left=LEFT, crop_right=RIGHT)
    third = detector.process_frame(frame, crop_left=LEFT, crop_right=RIGHT)

    # Pending: outlined in the debug view, but not an anomaly, so nothing is recorded.
    assert first.anomalies == [] and second.anomalies == []
    assert len(first.contours) == 1
    assert third.anomalies == [(640 - LEFT, 300, 12, 12)]
    assert third.metadata["anomaly_metrics"][0]["contrast"] == 150.0
    assert "Track 1 confirmed" in capsys.readouterr().out


def test_results_have_what_the_display_needs(detector):
    out = detector.process_frame(feed(), crop_left=LEFT, crop_right=RIGHT)
    width = 1280 - LEFT - RIGHT
    assert out.rcnn_boxes == {"space": [[0.0, 0.0, float(width), 720.0]]}
    assert out.rcnn_scores == {"space": [0.99]}
    assert out.darkness_detected is False
    assert out.space_mask.shape == (720, width)
    assert len(out.space_contours) == 1
    assert out.metadata["all_space_boxes"] == out.rcnn_boxes["space"]
    assert out.frame_number == 0


def test_a_flash_is_reported_as_a_transient(detector, capsys):
    flash = feed((640, 300, 12, 150))
    empty = feed()
    results = [
        detector.process_frame(f, crop_left=LEFT, crop_right=RIGHT)
        for f in (flash, flash, empty, empty, empty)
    ]
    assert all(r.anomalies == [] for r in results)
    assert detector.transients == 1
    assert (
        "Transient ignored: 12x12 px at (475, 300), seen on 2 frame(s)" in capsys.readouterr().out
    )


def test_test_frames_arrive_cropped_and_refresh_regions(detector):
    cropped = feed((640, 300, 12, 150))[:, LEFT : 1280 - RIGHT]
    detector.process_frame(feed(), crop_left=LEFT, crop_right=RIGHT)
    calls = detector.pipeline.region_model.calls
    out = detector.process_frame(cropped, is_test_frame=True)
    assert detector.pipeline.region_model.calls == calls + 1
    assert out.metadata["test_frame"] is True
    assert len(out.contours) == 1


def test_avoid_boxes_from_the_display_are_applied(detector):
    frame = feed((640, 300, 12, 150))
    for _ in range(3):
        out = detector.process_frame(
            frame, avoid_boxes=[(400, 250, 520, 350)], crop_left=LEFT, crop_right=RIGHT
        )
    assert out.anomalies == []
    assert detector.last_result.rejected[0].gates == ("avoid_box",)
