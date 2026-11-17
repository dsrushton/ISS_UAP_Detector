from __future__ import annotations

import json

from uapd.events import EventLog
from uapd.pipeline import STATE_OK, Detection, FrameResult, Region, Rejected, TrackSummary

SPACE = [Region("space", (0.0, 0.0, 939.0, 720.0), 0.99)]


def frame(index, *, detections=(), pending=(), rejected=(), ended=(), regions=SPACE):
    return FrameResult(
        index=index,
        state=STATE_OK,
        regions=list(regions),
        detections=list(detections),
        pending=list(pending),
        rejected=list(rejected),
        ended_tracks=list(ended),
    )


def det(track):
    return Detection(box=(10, 20, 6, 6), metrics={"contrast": 90.0}, track=track)


def test_event_holds_the_lead_in_and_is_written_when_recording_stops(tmp_path):
    log = EventLog(tmp_path, {"source": "test"}, buffer_frames=5)
    reject = Rejected(box=(1, 2, 3, 3), gates=("min_dimension",), metrics={"width": 3})

    log.observe(frame(0, rejected=[reject]), False, None)  # too old to be in the lead-in
    for i in range(1, 8):
        log.observe(frame(i), False, None)
    log.observe(frame(8, pending=[det(1)]), False, None)
    log.observe(frame(9, pending=[det(1)]), False, None)
    log.observe(frame(10, detections=[det(1)]), True, "00007")  # recording starts
    log.observe(frame(11, detections=[det(1)], rejected=[reject]), True, "00007")
    log.observe(frame(12), True, "00007")
    assert log.written == []
    log.observe(frame(13), False, None)  # recording stopped

    (path,) = log.written
    assert path == tmp_path / "00007.json"
    event = json.loads(path.read_text())
    assert event["event_id"] == "00007"
    assert event["source"] == "test"
    assert (event["first_frame"], event["recording_started_frame"], event["last_frame"]) == (
        8,
        10,
        12,
    )
    assert [f["frame"] for f in event["frames"]] == [8, 9, 10, 11]
    assert event["frames"][0]["pending"][0] == {
        "track": 1,
        "box": [10, 20, 6, 6],
        "metrics": {"contrast": 90.0},
    }
    assert event["frames"][3]["rejected"][0]["gates"] == ["min_dimension"]
    assert event["summary"] == {
        "frames_with_detections": 2,
        "detections": 2,
        "tracks": [1],
        "transients": 0,
        "rejected": 1,
        "rejected_by_gate": {"min_dimension": 1},
    }
    assert event["regions"] == [
        {
            "from_frame": 10,
            "regions": [{"label": "space", "box": [0.0, 0.0, 939.0, 720.0], "score": 0.99}],
        }
    ]


def test_region_changes_and_transients_are_recorded(tmp_path):
    log = EventLog(tmp_path, {}, buffer_frames=5)
    flash = TrackSummary(
        id=2,
        confirmed=False,
        first_index=1,
        last_index=2,
        frames_seen=2,
        first_box=(5, 5, 8, 8),
        last_box=(5, 5, 8, 8),
        max_contrast=120.0,
    )
    log.observe(frame(0, detections=[det(1)]), True, "00000")
    earth = SPACE + [Region("earth", (0.0, 400.0, 939.0, 720.0), 0.9)]
    log.observe(frame(1, detections=[det(1)], ended=[flash], regions=earth), True, "00000")
    log.close()

    event = json.loads(log.written[0].read_text())
    assert [r["from_frame"] for r in event["regions"]] == [0, 1]
    assert event["frames"][1]["ended_tracks"][0]["confirmed"] is False
    assert event["summary"]["transients"] == 1


def test_back_to_back_recordings_are_separate_events(tmp_path):
    log = EventLog(tmp_path, {}, buffer_frames=5)
    log.observe(frame(0, detections=[det(1)]), True, "00000")
    log.observe(frame(1, detections=[det(2)]), True, "00001")
    log.close()
    assert [p.name for p in log.written] == ["00000.json", "00001.json"]


def test_nothing_is_written_without_a_recording(tmp_path):
    log = EventLog(tmp_path, {}, buffer_frames=5)
    log.observe(frame(0, pending=[det(1)]), False, None)
    log.close()
    assert log.written == []
    assert list(tmp_path.iterdir()) == []
