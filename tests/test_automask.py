from __future__ import annotations

import numpy as np

from tests.test_labels import make_xml_folder
from uapd.labels import KIND_BOX, KIND_POLYGON, load_folder
from uapd.labels.automask import MaskSettings, build_masks, mask_to_polygons, summary


def test_mask_becomes_outlines_inside_its_box_and_specks_are_dropped():
    mask = np.zeros((100, 200), bool)
    mask[20:60, 30:90] = True  # the shape
    mask[80:82, 150:152] = True  # a speck
    mask[20:60, 120:190] = True  # outside the box: ignored
    polygons, fill = mask_to_polygons(mask, (10, 10, 100, 70), MaskSettings())
    assert len(polygons) == 1
    xs, ys = [p[0] for p in polygons[0]], [p[1] for p in polygons[0]]
    assert 28 <= min(xs) <= 32 and 87 <= max(xs) <= 91
    assert 18 <= min(ys) <= 22 and 57 <= max(ys) <= 61
    assert 0.35 < fill < 0.5
    assert mask_to_polygons(np.zeros((100, 200), bool), (10, 10, 100, 70), MaskSettings()) == (
        [],
        0.0,
    )


def test_build_masks_outlines_mask_classes_and_copies_the_rest(tmp_path):
    source = make_xml_folder(tmp_path / "in" / "clip" / "frames_1", frames=3)
    before = load_folder(source)
    labels = {a.label for s in before.samples for a in s.annotations}
    assert "space" in labels

    def half_of_each_box(image, boxes):
        masks = np.zeros((len(boxes), *image.shape[:2]), bool)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            masks[i, int(y1) : int(y2), int(x1) : int((x1 + x2) / 2)] = True
        return masks

    lines: list[str] = []
    results = build_masks(
        [source],
        tmp_path / "out",
        half_of_each_box,
        MaskSettings(mask_classes=("space",)),
        log=lines.append,
    )
    target = tmp_path / "out" / "frames_1"
    assert results[0].out == target and summary(results)["images"] == 3
    after = load_folder(target)
    assert after.format == "labels-json" and len(after.samples) == 3
    for sample in after.samples:
        assert sample.image.is_file()
        for a in sample.annotations:
            assert a.kind == (KIND_POLYGON if a.label == "space" else KIND_BOX)
            if a.kind == KIND_POLYGON:
                assert a.source == "model:sam2" and len(a.points) >= 4
    assert (tmp_path / "out" / "_review" / "frames_1.jpg").is_file()
    assert "median share of box filled: space" in lines[0]

    # A second pass leaves finished folders alone.
    again = build_masks([source], tmp_path / "out", half_of_each_box, log=lines.append)
    assert again[0].skipped


def test_an_empty_mask_keeps_the_box(tmp_path):
    source = make_xml_folder(tmp_path / "in" / "frames_1", frames=1)

    def nothing(image, boxes):
        return np.zeros((len(boxes), *image.shape[:2]), bool)

    results = build_masks([source], tmp_path / "out", nothing, log=lambda _: None)
    assert results[0].outlined == 0 and results[0].kept_boxes > 0
    sample = load_folder(results[0].out).samples[0]
    assert all(a.kind == KIND_BOX for a in sample.annotations)


def test_a_hole_is_cut_into_the_outline():
    import cv2

    mask = np.zeros((120, 160), bool)
    mask[10:110, 10:150] = True
    mask[40:80, 50:110] = False  # a pocket
    polygons, fill = mask_to_polygons(mask, (0, 0, 160, 120), MaskSettings())
    assert len(polygons) == 1
    drawn = np.zeros((120, 160), np.uint8)
    cv2.fillPoly(drawn, [np.array(polygons[0], np.int32)], 255)
    assert drawn[20, 20] == 255 and drawn[60, 80] == 0 and drawn[60, 130] == 255
    assert abs(fill - (100 * 140 - 40 * 60) / (120 * 160)) < 0.03


def test_elimination_makes_the_station_what_is_left_of_its_box(tmp_path):
    from uapd.labels import Annotation, Sample
    from uapd.labels.automask import eliminate_folder, fill_annotations
    from uapd.labels.store import write_folder

    source = make_xml_folder(
        tmp_path / "in" / "frames_1",
        frames=2,
        objects=[("iss", 0, 0, 160, 120), ("space", 0, 0, 160, 60)],
    )
    out = tmp_path / "out" / "frames_1"
    out.mkdir(parents=True)
    samples = []
    for sample in load_folder(source).samples:
        copy = out / sample.image.name
        copy.write_bytes(sample.image.read_bytes())
        space = ((0.0, 0.0), (160.0, 0.0), (160.0, 60.0), (0.0, 60.0))
        patch = ((10.0, 70.0), (30.0, 70.0), (30.0, 90.0))  # what the mask model found
        samples.append(
            Sample(
                copy,
                (
                    Annotation("space", (0, 0, 160, 60), KIND_POLYGON, space),
                    Annotation("iss", (10, 70, 30, 90), KIND_POLYGON, patch),
                ),
                group="g",
            )
        )
    write_folder(out, samples)

    count, sheet = eliminate_folder(out, source, review_frames=1)
    assert count == 2 and len(sheet) == 1
    after = load_folder(out).samples[0]
    station = [a for a in after.annotations if a.label == "iss"]
    assert len(station) == 1 and station[0].source == "elimination"
    mask = fill_annotations((120, 160), after.annotations, ["iss"])
    assert mask[100, 80] == 255 and mask[30, 80] == 0  # below space, not in it
    assert (out / "labels.before-elimination.json").is_file()
    # A second run needs no source: the boxes were kept.
    assert eliminate_folder(out)[0] == 2


def test_click_edits_add_remove_and_undo_and_the_station_follows(tmp_path):
    from uapd.labels import Annotation, Sample
    from uapd.labels.automask import eliminate_folder, fill_annotations
    from uapd.labels.edit import add_outline, can_undo, remove_outline, undo
    from uapd.labels.store import write_folder

    source = make_xml_folder(
        tmp_path / "in" / "frames_1", frames=3, objects=[("iss", 0, 0, 160, 120)]
    )
    out = tmp_path / "out" / "frames_1"
    out.mkdir(parents=True)
    samples = []
    for sample in load_folder(source).samples:
        copy = out / sample.image.name
        copy.write_bytes(sample.image.read_bytes())
        samples.append(Sample(copy, (Annotation("lf", (5, 5, 20, 20)),), group="g"))
    write_folder(out, samples)
    eliminate_folder(out, source)

    def station(index=0):
        return fill_annotations((120, 160), load_folder(out).samples[index].annotations, ["iss"])

    assert station()[30, 120] == 255  # everything in the box is station to begin with

    def top_right(image, point):
        mask = np.zeros(image.shape[:2], bool)
        mask[0:60, 80:160] = True
        return mask

    # One frame: Earth is added there and comes out of the station.
    assert add_outline(out, "earth", (120, 30), top_right, frames=["00000.jpg"]) == 1
    first = load_folder(out).samples[0]
    earth = [a for a in first.annotations if a.label == "earth"]
    assert len(earth) == 1 and earth[0].source == "human-click"
    assert station(0)[30, 120] == 0 and station(0)[100, 30] == 255
    assert station(1)[30, 120] == 255  # the other frames are untouched
    assert any(a.label == "lf" for a in first.annotations)

    # The whole folder, then one step back.
    assert add_outline(out, "earth", (120, 30), top_right) == 3
    assert station(2)[30, 120] == 0
    assert can_undo(out) and undo(out) and not can_undo(out)
    assert station(2)[30, 120] == 255 and station(0)[30, 120] == 0

    # Removing gives the area back to the station; the station itself cannot be removed.
    assert remove_outline(out, (100, 100)) == 0
    assert remove_outline(out, (120, 30), frames=["00000.jpg"]) == 1
    assert station(0)[30, 120] == 255
