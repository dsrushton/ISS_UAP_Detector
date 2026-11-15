"""Drives the real V1 detection code through the adapter, with the RCNN stubbed.

The stub stands in for the model only: region output is fixed to one space box
covering the whole cropped frame. Everything downstream (V1's threading, crop,
masks, contour gates) is the unmodified legacy code.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from uapd.eval import run_eval  # noqa: E402
from uapd.pipeline import PipelineError  # noqa: E402
from uapd.pipelines.legacy_v1 import DEFAULT_LEGACY_DIR, LegacyV1Pipeline  # noqa: E402

RCNN_CYCLE = 5
BLOB_CENTRE = (640, 300)  # in the uncropped 1280x720 frame


class _Frames:
    def __init__(self, frames):
        self._frames = frames

    def describe(self):
        return {"kind": "array", "frames": len(self._frames)}

    def frames(self):
        yield from enumerate(self._frames)


def _feed_frame(blob: bool) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), np.uint8)
    if blob:
        cv2.circle(frame, BLOB_CENTRE, 10, (150, 150, 150), -1)
    return frame


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
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
    pipe = LegacyV1Pipeline(weights=weights, rcnn_cycle=RCNN_CYCLE)
    yield pipe
    pipe.close()


def test_missing_weights_is_reported(tmp_path):
    with pytest.raises(PipelineError, match="model weights not found"):
        LegacyV1Pipeline(weights=tmp_path / "missing.pth")


def test_v1_finds_a_bright_object_once_regions_exist(pipeline):
    report = run_eval(pipeline, _Frames([_feed_frame(blob=True)] * 12), frame_mode="all")

    frames = report["frames"]
    # V1 queues its first RCNN run on frame count 5 (index 4); results apply from index 5.
    assert [f["state"] for f in frames[:5]] == ["no_space"] * 5
    assert [f["state"] for f in frames[5:]] == ["ok"] * 7
    assert all(not f["detections"] for f in frames[:5])
    assert all(len(f["detections"]) == 1 for f in frames[5:])

    x, y, w, h = frames[5]["detections"][0]["box"]
    crop_left = report["pipeline"]["constants"]["CROP_LEFT"]
    centre_x = x + w / 2 + crop_left
    centre_y = y + h / 2
    assert abs(centre_x - BLOB_CENTRE[0]) <= 3
    assert abs(centre_y - BLOB_CENTRE[1]) <= 3
    assert 15 <= w <= 25 and 15 <= h <= 25

    metrics = frames[5]["detections"][0]["metrics"]
    assert metrics["contrast"] > 100
    assert report["summary"]["detections"] == 7


def test_v1_reports_nothing_on_empty_space(pipeline):
    report = run_eval(pipeline, _Frames([_feed_frame(blob=False)] * 8))
    assert report["summary"]["detections"] == 0
    assert report["summary"]["states"] == {"no_space": 5, "ok": 3}


def test_runs_are_repeatable(pipeline):
    frames = _Frames([_feed_frame(blob=True)] * 12)
    first = run_eval(pipeline, frames, frame_mode="all")
    # Frame counting continues inside V1, so compare detections, not RCNN timing.
    second = run_eval(pipeline, frames, frame_mode="all")
    boxes = (
        lambda report: [  # noqa: E731
            d["box"] for f in report["frames"][5:] for d in f["detections"]
        ]
    )
    assert boxes(first) == boxes(second)


def test_describe_records_constants_and_weights(pipeline):
    info = pipeline.describe()
    assert info["constants"]["RCNN_DETECTION_CYCLE"] == RCNN_CYCLE
    assert info["constants"]["MIN_CONTRAST"] == 23
    assert len(info["weights_sha256"]) == 64


def test_narrow_frames_are_rejected(pipeline):
    with pytest.raises(PipelineError, match="expects the 1280 px feed"):
        pipeline.process(np.zeros((720, 300, 3), np.uint8), 0)
