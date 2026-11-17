from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uapd.config import Config, TrackingConfig
from uapd.eval import run_eval
from uapd.pipeline import Detection
from uapd.pipelines.core import CorePipeline
from uapd.tracking import Tracker

from .test_core_pipeline import FakeRegionModel, Frames

CROP_LEFT = Config().frame.crop_left


def det(x: int, y: int, size: int = 10, contrast: float = 100.0) -> Detection:
    return Detection(box=(x, y, size, size), metrics={"contrast": contrast})


def test_a_track_is_confirmed_on_its_third_frame():
    tracker = Tracker(TrackingConfig())
    for index in (0, 1):
        update = tracker.update([det(100 + 3 * index, 100)], index)
        assert [d.track for d in update.pending] == [1]
        assert update.detections == [] and update.confirmed == []

    third = tracker.update([det(106, 100)], 2)
    assert [d.track for d in third.detections] == [1]
    assert third.pending == [] and third.confirmed == [1]

    fourth = tracker.update([det(109, 100)], 3)
    assert [d.track for d in fourth.detections] == [1]
    assert fourth.confirmed == []


def test_a_flash_seen_on_two_frames_ends_as_a_transient():
    tracker = Tracker(TrackingConfig())
    tracker.update([det(100, 100, contrast=80.0)], 0)
    tracker.update([det(100, 100, contrast=90.0)], 1)  # the same camera frame, sent again
    assert tracker.update([], 2).ended == []
    assert tracker.update([], 3).ended == []  # two missed frames are tolerated
    (ended,) = tracker.update([], 4).ended
    assert not ended.confirmed
    assert (ended.id, ended.first_index, ended.last_index, ended.frames_seen) == (1, 0, 1, 2)
    assert ended.max_contrast == 90.0


def test_missed_frames_do_not_break_a_track():
    tracker = Tracker(TrackingConfig())
    tracker.update([det(100, 100)], 0)
    tracker.update([det(100, 100)], 1)
    tracker.update([], 2)
    tracker.update([], 3)
    assert tracker.update([det(104, 100)], 4).confirmed == [1]


def test_matching_uses_distance_between_boxes():
    tracker = Tracker(TrackingConfig(match_distance=20.0))
    tracker.update([det(100, 100)], 0)
    near = tracker.update([det(128, 100)], 1)  # 18 px gap
    assert [d.track for d in near.pending] == [1]
    far = tracker.update([det(165, 100)], 2)  # 27 px gap from the last box
    assert [d.track for d in far.pending] == [2]


def test_each_detection_goes_to_its_closest_track():
    tracker = Tracker(TrackingConfig())
    tracker.update([det(100, 100), det(125, 100)], 0)
    update = tracker.update([det(127, 100), det(102, 100)], 1)
    assert [(d.box[0], d.track) for d in update.pending] == [(127, 2), (102, 1)]


def test_confirm_frames_of_one_turns_the_gate_off():
    tracker = Tracker(TrackingConfig(confirm_frames=1))
    update = tracker.update([det(100, 100)], 0)
    assert [d.track for d in update.detections] == [1]
    assert update.confirmed == [1]


def test_finish_ends_open_tracks():
    tracker = Tracker(TrackingConfig())
    for index in range(3):
        tracker.update([det(100, 100)], index)
    tracker.update([det(100, 100), det(300, 300)], 3)
    assert [(t.id, t.confirmed) for t in tracker.finish()] == [(1, True), (2, False)]
    assert tracker.finish() == []


def test_bad_config_is_rejected():
    with pytest.raises(ValueError):
        Tracker(TrackingConfig(confirm_frames=0))


# --- through the pipeline, on a doubled feed like the live one ---------------------------


def doubled_feed(camera_frames: int, flash_at: int, dot_from: int) -> Frames:
    """Stream frames for a camera that sends each frame twice.

    A flash is visible on one camera frame; a small dot appears later and drifts.
    """
    rng = np.random.default_rng(0)
    frames = []
    for n in range(camera_frames):
        frame = rng.integers(0, 6, (720, 1280, 3), dtype=np.uint8)  # sensor noise
        if n == flash_at:
            frame[200:208, 500:508] = 180
        if n >= dot_from:
            x = 700 + 3 * (n - dot_from)
            frame[400:406, x : x + 6] = 120
        frames += [frame, frame.copy()]
    return Frames(frames)


def test_flash_is_dropped_and_persistent_object_is_kept():
    feed = doubled_feed(camera_frames=12, flash_at=2, dot_from=5)
    pipeline = CorePipeline(Config(), FakeRegionModel())
    report = run_eval(pipeline, feed, frame_mode="all")
    frames = report["frames"]

    summary = report["summary"]
    assert summary["tracks_confirmed"] == 1
    assert summary["transients"] == 1

    # The flash (stream frames 4 and 5) is only ever pending.
    assert [len(f.get("pending", [])) for f in frames[4:6]] == [1, 1]
    assert all(not f["detections"] for f in frames[:12])
    (flash,) = [t for f in frames for t in f.get("ended_tracks", [])]
    assert flash["confirmed"] is False
    assert list(flash["first_box"]) == [500 - CROP_LEFT, 200, 8, 8]
    assert (flash["first_index"], flash["last_index"], flash["frames_seen"]) == (4, 5, 2)

    # The dot appears on stream frame 10 and is confirmed on the third frame it is seen.
    assert [len(f.get("pending", [])) for f in frames[10:12]] == [1, 1]
    assert frames[12]["confirmed_tracks"] == [2]
    assert all(len(f["detections"]) == 1 for f in frames[12:])
    assert {d["track"] for f in frames[12:] for d in f["detections"]} == {2}
    assert summary["detections"] == 12
    assert summary["pending"] == 4


def test_without_the_gate_the_flash_is_a_detection():
    feed = doubled_feed(camera_frames=6, flash_at=2, dot_from=99)
    config = dataclasses.replace(Config(), tracking=TrackingConfig(confirm_frames=1))
    report = run_eval(CorePipeline(config, FakeRegionModel()), feed)
    assert report["summary"]["detections"] == 2
    assert report["summary"]["transients"] == 0
