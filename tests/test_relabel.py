from __future__ import annotations

import json

import numpy as np

from tests.test_labels import make_xml_folder
from uapd.app.captures import VERDICTS, list_captures, write_verdict
from uapd.labels import KIND_POLYGON, Annotation, Sample, load_folder
from uapd.labels.review import Review, read_review, write_review
from uapd.labels.store import write_folder
from uapd.train.config import TrainConfig
from uapd.train.data import plan_data


def test_review_marks_round_trip_and_bad_files_read_as_empty(tmp_path):
    assert read_review(tmp_path) == Review()
    write_review(tmp_path, Review("reject", {"b.jpg", "a.jpg"}, "blurry"))
    assert read_review(tmp_path) == Review("reject", {"a.jpg", "b.jpg"}, "blurry")
    assert json.loads((tmp_path / "review.json").read_text())["bad_frames"] == ["a.jpg", "b.jpg"]
    (tmp_path / "review.json").write_text("not json")
    assert read_review(tmp_path) == Review()


def test_training_leaves_out_rejected_folders_and_flagged_frames(tmp_path):
    kept = make_xml_folder(tmp_path / "kept", frames=4)
    make_xml_folder(tmp_path / "dropped", frames=3)
    write_review(kept, Review("ok", {"00001.jpg"}))
    write_review(tmp_path / "dropped", Review("reject"))
    plan = plan_data([tmp_path], TrainConfig(val_fraction=0.0))
    assert sorted(item.image.name for item in plan.train) == ["00000.jpg", "00002.jpg", "00003.jpg"]
    assert plan.report["bad_frames"] == 1
    assert [f.rsplit("/", 1)[-1] for f in plan.report["rejected_folders"]] == ["dropped"]
    warnings = " ".join(plan.report["warnings"])
    assert "review says reject: dropped" in warnings and "1 frames left out" in warnings


def test_captures_are_listed_with_their_data_and_verdicts(tmp_path):
    (tmp_path / "00001.avi").write_bytes(b"")
    (tmp_path / "00002.avi").write_bytes(b"")
    (tmp_path / "00002.json").write_text(
        json.dumps(
            {
                "started": "2026-11-27T15:59:38.583+00:00",
                "summary": {"detections": 118, "tracks": [5], "transients": 0},
            }
        )
    )
    first, second = list_captures(tmp_path)
    assert (first.event_id, first.data_file, first.verdict) == ("00001", None, None)
    assert first.describe() == "no data file"
    assert second.describe() == "2026-11-27 15:59:38, 118 detections, 1 track(s)"

    write_verdict(second, "flare")
    stored = json.loads((tmp_path / "00002.verdict.json").read_text())
    assert (stored["verdict"], stored["event_id"], stored["source"]) == ("flare", "00002", "human")
    assert [c.verdict for c in list_captures(tmp_path)] == [None, "flare"]
    # The data file is the detector's and is left alone; a verdict file is not a capture.
    assert "verdict" not in json.loads((tmp_path / "00002.json").read_text())
    write_verdict(second, None)
    assert not (tmp_path / "00002.verdict.json").exists()
    assert len(VERDICTS) == 11
    assert list_captures(tmp_path / "missing") == []


def test_viewer_pictures(tmp_path):
    from uapd.app.relabel import fit, label_folders, render_sample, to_ppm

    folder = make_xml_folder(tmp_path / "set" / "frames_1", frames=1)
    (tmp_path / "set" / "_review").mkdir()
    image = next(folder.glob("*.jpg"))
    triangle = ((10.0, 10.0), (100.0, 10.0), (10.0, 50.0))
    for old in (folder / "2ndTrainingXMLs").iterdir():
        old.unlink()
    write_folder(
        folder,
        [
            Sample(
                image,
                (
                    Annotation("space", (10, 10, 100, 50), KIND_POLYGON, triangle),
                    Annotation("lf", (120, 80, 150, 110)),
                ),
                group="g",
            )
        ],
    )
    assert label_folders(tmp_path / "set") == [folder]
    sample = load_folder(folder).samples[0]
    plain = render_sample(sample, outlines=False, boxes=False)
    outlined = render_sample(sample, outlines=True, boxes=False)
    boxed = render_sample(sample, outlines=False, boxes=True)
    assert plain.shape == (120, 160, 3)
    # The outline tints its own area red and leaves the box's area alone, and the
    # other way round.
    assert int(outlined[20, 20, 2]) > int(plain[20, 20, 2]) + 50
    assert np.array_equal(outlined[90:100, 125:145], plain[90:100, 125:145])
    assert not np.array_equal(boxed[80:111, 120:151], plain[80:111, 120:151])
    assert np.array_equal(boxed[15:40, 15:30], plain[15:40, 15:30])

    small = fit(np.zeros((720, 939, 3), np.uint8), 400, 400)
    assert small.shape[:2] == (306, 400)
    ppm = to_ppm(np.zeros((2, 3, 3), np.uint8))
    assert ppm.startswith(b"P6 3 2 255\n") and len(ppm) == len(b"P6 3 2 255\n") + 18
