from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from uapd.labels import Annotation, LabelError, Sample, find_label_folders, load_folder
from uapd.labels.store import write_folder
from uapd.regions.manifest import (
    INPUT_RGB,
    KIND_RCNN,
    LEGACY_MANIFEST,
    ManifestError,
    ModelManifest,
    read_manifest,
    write_manifest,
)


def make_csv_folder(folder, frames=5, size=(160, 120), rows=None):
    """Frames with labels in the original notebook's format."""
    folder.mkdir(parents=True, exist_ok=True)
    width, height = size
    for i in range(frames):
        image = np.zeros((height, width, 3), np.uint8)
        image[60:] = (200, 90, 20)
        cv2.imwrite(str(folder / f"frame_{i}.jpg"), image)
        with open(folder / f"bboxes_{i}.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x", "y", "width", "height", "class"])
            default = [[0, 0, width, 60, "Space"], [0, 60, width, height - 60, "Earth_broad"]]
            writer.writerows(default if rows is None else rows)
    return folder


def test_notebook_csv_folder_is_read(tmp_path):
    folder = make_csv_folder(tmp_path / "frames_1", frames=3)
    cv2.imwrite(str(folder / "frame_9.jpg"), np.zeros((4, 4, 3), np.uint8))  # no CSV
    labels = load_folder(folder)
    assert labels.format == "csv-frames"
    assert [s.image.name for s in labels.samples] == ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg"]
    assert labels.unlabelled == 1
    first = labels.samples[0].annotations
    assert [(a.label, a.box) for a in first] == [
        ("Space", (0.0, 0.0, 160.0, 60.0)),
        ("Earth_broad", (0.0, 60.0, 160.0, 120.0)),
    ]


def test_csv_rows_that_cannot_be_used_are_reported(tmp_path):
    folder = make_csv_folder(
        tmp_path / "f", frames=1, rows=[[0, 0, "wide", 5, "Space"], [1, 1, 5, 5, ""]]
    )
    labels = load_folder(folder)
    assert labels.samples[0].annotations == ()
    assert len(labels.problems) == 2


def test_native_labels_round_trip(tmp_path):
    image = tmp_path / "a.jpg"
    cv2.imwrite(str(image), np.zeros((8, 8, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "b.jpg"), np.zeros((8, 8, 3), np.uint8))
    box = Annotation("space", (0.0, 0.0, 8.0, 4.0), source="human")
    outline = Annotation(
        "iss", (1.0, 1.0, 5.0, 5.0), kind="polygon", points=((1.0, 1.0), (5.0, 1.0), (5.0, 5.0))
    )
    write_folder(tmp_path, [Sample(image, (box, outline), group=str(tmp_path))])
    labels = load_folder(tmp_path)
    assert labels.format == "labels-json"
    assert labels.samples[0].annotations == (box, outline)
    assert labels.unlabelled == 1


def test_label_folders_are_found_below_a_root(tmp_path):
    a = make_csv_folder(tmp_path / "sample_1" / "frames_1", frames=1)
    b = make_csv_folder(tmp_path / "sample_2" / "frames_5", frames=1)
    (tmp_path / "empty").mkdir()
    assert find_label_folders(tmp_path) == [a, b]
    assert find_label_folders(a) == [a]
    with pytest.raises(LabelError, match="no labels found"):
        load_folder(tmp_path / "empty")


def make_xml_folder(folder, frames=5, size=(160, 120), subfolder="2ndTrainingXMLs", objects=None):
    """Frames with labelImg XML labels, laid out as the January 2025 training data."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / subfolder).mkdir(exist_ok=True)
    width, height = size
    default = [("space", 0, 0, width, 60), ("earth", 0, 60, width, height)]
    for i in range(frames):
        image = np.zeros((height, width, 3), np.uint8)
        image[60:] = (200, 90, 20)
        cv2.imwrite(str(folder / f"{i:05d}.jpg"), image)
        items = "".join(
            f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
            f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
            for n, a, b, c, d in (default if objects is None else objects)
        )
        (folder / subfolder / f"{i:05d}.xml").write_text(f"<annotation>{items}</annotation>")
    return folder


def test_labelimg_xml_is_read_and_preferred_over_old_csvs(tmp_path):
    folder = make_xml_folder(tmp_path / "frames_1", frames=3, objects=[("lf", 5, 6, 50, 60)])
    (folder / "2ndTrainingXMLs" / "00002.xml").unlink()  # one frame left unlabelled
    with open(folder / "bboxes_0.csv", "w") as fh:  # the older label set is still there
        fh.write("frame,object_id,x,y,width,height,class\n0,1,0,0,9,9,Space\n")
    labels = load_folder(folder)
    assert labels.format == "voc-xml (2ndTrainingXMLs)"
    assert [s.image.name for s in labels.samples] == ["00000.jpg", "00001.jpg"]
    assert labels.unlabelled == 1
    assert labels.samples[0].annotations == (Annotation("lf", (5.0, 6.0, 50.0, 60.0)),)
    # Only the frame folder counts as labelled, not the XML folder inside it.
    assert find_label_folders(tmp_path) == [folder]


def test_xml_beside_the_frames_and_broken_objects(tmp_path):
    folder = make_xml_folder(tmp_path / "f", frames=1, subfolder=".")
    (folder / "00000.xml").write_text(
        "<annotation><object><name>iss</name></object>"
        "<object><name>sun</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax>"
        "<ymax>4</ymax></bndbox></object></annotation>"
    )
    labels = load_folder(folder)
    assert labels.format == "voc-xml"
    assert [a.label for a in labels.samples[0].annotations] == ["sun"]
    assert labels.problems == ["00000.xml object 1: no usable box"]


def test_manifest_round_trip_and_legacy_default(tmp_path):
    weights = tmp_path / "best.pth"
    assert read_manifest(weights) is None
    assert LEGACY_MANIFEST.input == "legacy-v1"
    manifest = ModelManifest(
        KIND_RCNN,
        ("background", "space"),
        INPUT_RGB,
        model_args={"min_size": 64},
        record={"trained": {"epoch": 3}},
    )
    path = write_manifest(weights, manifest)
    assert path == tmp_path / "best.json"
    assert read_manifest(weights) == manifest


def test_manifest_with_unknown_input_is_refused(tmp_path):
    (tmp_path / "m.json").write_text(
        json.dumps({"kind": KIND_RCNN, "classes": ["background"], "input": "bgr"})
    )
    with pytest.raises(ManifestError, match="unknown input"):
        read_manifest(tmp_path / "m.pth")
