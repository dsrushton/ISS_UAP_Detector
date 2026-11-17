from __future__ import annotations

import json

import pytest

from uapd.eval import REPORT_SCHEMA_VERSION, run_eval, write_report


def test_summary_counts_every_frame(fake_pipeline, five_frames):
    report = run_eval(fake_pipeline, five_frames)
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["pipeline"] == {"name": "fake"}
    assert report["summary"] == {
        "frames": 5,
        "frames_with_detections": 2,
        "detections": 2,
        "pending": 0,
        "tracks_confirmed": 0,
        "transients": 0,
        "rejected": 0,
        "rejected_by_gate": {},
        "faint": 0,
        "states": {"ok": 5},
        "frames_with_region": {"space": 5},
    }


def test_default_lists_only_frames_with_detections(fake_pipeline, five_frames):
    report = run_eval(fake_pipeline, five_frames)
    assert [f["index"] for f in report["frames"]] == [1, 3]
    assert report["frames"][0]["detections"][0]["box"] == [1, 2, 3, 4]


def test_all_mode_lists_every_frame(fake_pipeline, five_frames):
    report = run_eval(fake_pipeline, five_frames, frame_mode="all")
    assert [f["index"] for f in report["frames"]] == [0, 1, 2, 3, 4]


def test_max_frames_stops_early(fake_pipeline, five_frames):
    report = run_eval(fake_pipeline, five_frames, max_frames=2)
    assert report["summary"]["frames"] == 2
    assert report["summary"]["detections"] == 1


def test_bad_arguments(fake_pipeline, five_frames):
    with pytest.raises(ValueError):
        run_eval(fake_pipeline, five_frames, frame_mode="some")
    with pytest.raises(ValueError):
        run_eval(fake_pipeline, five_frames, max_frames=0)


def test_report_round_trips_through_json(fake_pipeline, five_frames, tmp_path):
    report = run_eval(fake_pipeline, five_frames)
    out = tmp_path / "nested" / "report.json"
    write_report(report, out)
    loaded = json.loads(out.read_text())
    assert loaded["summary"] == report["summary"]
    assert loaded["frames"][0]["detections"][0]["metrics"]["contrast"] == 42.5


def test_repeat_runs_agree_outside_the_run_block(fake_pipeline, five_frames):
    first = run_eval(fake_pipeline, five_frames, frame_mode="all")
    second = run_eval(fake_pipeline, five_frames, frame_mode="all")
    first.pop("run")
    second.pop("run")
    assert first == second
