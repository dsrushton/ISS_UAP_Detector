from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uapd.candidates import build_search_area, find_candidates
from uapd.config import CandidateConfig

SHAPE = (200, 300)  # height, width
WHOLE_FRAME = [(0.0, 0.0, 300.0, 200.0)]


def _frame(*squares: tuple[int, int, int, int, int]) -> np.ndarray:
    """A black frame with filled rectangles (x, y, w, h, brightness)."""
    frame = np.zeros((*SHAPE, 3), np.uint8)
    for x, y, w, h, value in squares:
        frame[y : y + h, x : x + w] = value
    return frame


def _find(frame, config=None, space=WHOLE_FRAME, **kwargs):
    area = build_search_area(SHAPE, space)
    return find_candidates(frame, area, config or CandidateConfig(), **kwargs)


def test_measurements_are_full_resolution_and_exact():
    found = _find(_frame((100, 80, 12, 8, 150)))
    (candidate,) = found.candidates
    assert candidate.accepted
    assert candidate.box == (100, 80, 12, 8)
    assert candidate.metrics == {
        "width": 12,
        "height": 8,
        "area": 96,
        "bbox_area": 96,
        "aspect_ratio": 1.5,
        "obj_brightness": 150.0,
        "peak_brightness": 150,
        "bg_brightness": 0.0,
        "contrast": 150.0,
        "border_distance": 80.0,
        "distance_lf": None,
        "distance_iss": None,
        "distance_panel": None,
    }


def test_smallest_configured_object_is_accepted():
    # 5x5 px is the configured minimum (min_dimension 5, min_area 25). V1 rejects it.
    (candidate,) = _find(_frame((100, 80, 5, 5, 150))).candidates
    assert candidate.accepted
    assert candidate.metrics["area"] == 25


def test_too_small_object_is_recorded_with_its_gates():
    (candidate,) = _find(_frame((100, 80, 4, 4, 150))).candidates
    assert candidate.failed == ("min_dimension", "min_area")
    assert candidate.metrics["width"] == 4
    assert candidate.metrics["obj_brightness"] == 150.0


def test_single_pixel_is_a_candidate():
    (candidate,) = _find(_frame((100, 80, 1, 1, 150))).candidates
    assert candidate.box == (100, 80, 1, 1)
    assert candidate.metrics["area"] == 1
    assert "min_dimension" in candidate.failed


def test_background_is_measured_around_the_object():
    # A faint glow (below the bright threshold) surrounds the object.
    frame = _frame((96, 76, 18, 18, 9), (100, 80, 10, 10, 30))
    (candidate,) = _find(frame).candidates
    assert candidate.metrics["bg_brightness"] == 9.0
    assert candidate.metrics["contrast"] == 21.0
    assert candidate.failed == ("contrast",)


@pytest.mark.parametrize(
    ("square", "gate"),
    [
        ((100, 80, 120, 10, 150), "max_size"),
        ((100, 80, 10, 10, 230), "max_brightness"),
    ],
)
def test_gates(square, gate):
    (candidate,) = _find(_frame(square)).candidates
    assert candidate.failed == (gate,)


def test_aspect_ratio_gate():
    config = dataclasses.replace(CandidateConfig(), max_aspect_ratio=3.0)
    (candidate,) = _find(_frame((100, 80, 40, 8, 150)), config).candidates
    assert candidate.failed == ("aspect_ratio",)


def test_border_distance_is_measured_to_the_space_edge():
    space = [(50.0, 50.0, 250.0, 150.0)]
    near = _find(_frame((52, 90, 10, 10, 150)), space=space).candidates[0]
    assert near.metrics["border_distance"] == 2.0
    assert near.failed == ("border",)

    clear = _find(_frame((53, 90, 10, 10, 150)), space=space).candidates[0]
    assert clear.metrics["border_distance"] == 3.0
    assert clear.accepted


def test_frame_edge_counts_as_a_border():
    (candidate,) = _find(_frame((1, 90, 10, 10, 150))).candidates
    assert candidate.metrics["border_distance"] == 1.0
    assert "border" in candidate.failed


def test_objects_outside_space_are_not_candidates():
    space = [(0.0, 0.0, 140.0, 199.0)]
    assert _find(_frame((200, 80, 10, 10, 150)), space=space).candidates == []


def test_bright_area_reaching_the_edge_of_space_is_not_enclosed():
    # Like the Earth limb: bright from inside the space box to its boundary.
    space = [(0.0, 0.0, 140.0, 199.0)]
    assert _find(_frame((100, 0, 200, 200, 150)), space=space).candidates == []


def test_region_distances_and_overlap_gates():
    frame = _frame((100, 80, 10, 10, 150))
    regions = {
        "lf": [(105.0, 85.0, 130.0, 100.0)],
        "iss": [(105.0, 85.0, 130.0, 100.0)],
        "panel": [(120.0, 80.0, 140.0, 90.0)],
    }
    (candidate,) = _find(frame, region_boxes=regions).candidates
    assert candidate.metrics["distance_lf"] == 0.0
    assert candidate.metrics["distance_iss"] == 0.0
    assert candidate.metrics["distance_panel"] == 10.0
    # filter_iss is off by default, so the ISS overlap is measured but not a gate.
    assert candidate.failed == ("overlap_lf",)


def test_avoid_boxes():
    frame = _frame((100, 80, 10, 10, 150))
    hit = _find(frame, avoid_boxes=[(90.0, 70.0, 101.0, 81.0)]).candidates[0]
    assert hit.failed == ("avoid_box",)
    # Sharing an edge is not an overlap.
    miss = _find(frame, avoid_boxes=[(90.0, 70.0, 100.0, 80.0)]).candidates[0]
    assert miss.accepted


MEASURE_FAINT = dataclasses.replace(CandidateConfig(), measure_faint=True)


def test_every_failed_gate_is_listed():
    (candidate,) = _find(_frame((1, 90, 3, 3, 20)), MEASURE_FAINT).candidates
    assert candidate.failed == ("min_dimension", "min_area", "contrast", "border")


def test_faint_candidates_are_counted_not_measured():
    # Peak brightness 20 is below min_contrast (23), so the contrast gate must fail.
    frame = _frame((100, 80, 10, 10, 20), (150, 80, 1, 1, 12), (200, 80, 10, 10, 150))
    found = _find(frame)
    assert found.found == 3
    assert found.faint == 2
    assert [c.box for c in found.candidates] == [(200, 80, 10, 10)]
    assert not found.truncated

    measured = _find(frame, MEASURE_FAINT)
    assert measured.faint == 0
    assert [c.failed for c in measured.candidates] == [
        ("contrast",),
        ("min_dimension", "min_area", "contrast"),
        (),
    ]


def test_bright_enough_candidates_are_never_set_aside():
    # Peak 23 could still pass, so it is measured even though it then fails.
    found = _find(_frame((96, 76, 18, 18, 9), (100, 80, 10, 10, 23)))
    assert found.faint == 0
    assert found.candidates[0].failed == ("contrast",)


def test_candidate_limit_keeps_the_largest_and_reports_the_total():
    squares = [(20 + 30 * i, 80, 6 + i, 6 + i, 150) for i in range(6)]
    config = dataclasses.replace(CandidateConfig(), max_candidates=4)
    found = _find(_frame(*squares), config)
    assert found.found == 6
    assert found.truncated
    assert [c.metrics["width"] for c in found.candidates] == [8, 9, 10, 11]


def test_order_is_stable_top_to_bottom():
    found = _find(_frame((200, 150, 8, 8, 150), (50, 20, 8, 8, 150), (20, 150, 8, 8, 150)))
    assert [c.box[:2] for c in found.candidates] == [(50, 20), (20, 150), (200, 150)]
    assert not found.truncated


def test_no_space_no_candidates():
    assert _find(_frame((100, 80, 10, 10, 150)), space=[]).candidates == []


def test_frame_must_match_the_search_area():
    area = build_search_area(SHAPE, WHOLE_FRAME)
    with pytest.raises(ValueError, match="search area"):
        find_candidates(np.zeros((100, 100, 3), np.uint8), area, CandidateConfig())
