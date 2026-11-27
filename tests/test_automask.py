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
