"""Edits a person makes to an outlined label folder, one click at a time.

* ``add_outline`` - outline whatever is at a point (with a mask model) and label it.
* ``remove_outline`` - take away the outline under a point.
* ``undo`` - put the labels back as they were before the last edit.

Each edit can cover one frame or every frame of the folder: a clip's camera
barely moves, so the same point usually lands on the same thing in all of them.
After an edit the class found by elimination (the station; see
``uapd.labels.automask.eliminate_folder``) is worked out again for the frames
that changed, so adding a missing piece of Earth also takes it out of the station.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from uapd.labels import KIND_POLYGON, Annotation, LabelError, Sample, load_folder
from uapd.labels.automask import (
    SOURCE_ELIMINATION,
    SOURCE_HUMAN,
    MaskSettings,
    carve,
    eliminate_sample,
    fill_annotations,
    mask_to_polygons,
    read_elimination,
)
from uapd.labels.store import LABELS_FILE, write_folder

UNDO_FILE = "labels.undo.json"
CLICK_SOURCE = f"{SOURCE_HUMAN}-click"
CLICK_SETTINGS = MaskSettings(min_piece=0.0005)
ELIMINATION_SETTINGS = MaskSettings(min_piece=0.002)

PointMasker = Callable[[np.ndarray, Sequence[float]], np.ndarray]
Progress = Callable[[int, int], None]


def _inside(points: Sequence[tuple[float, float]], point: Sequence[float]) -> bool:
    import cv2

    contour = np.array(points, np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def _area(annotation: Annotation) -> float:
    x1, y1, x2, y2 = annotation.box
    return (x2 - x1) * (y2 - y1)


def _apply(
    folder: Path,
    frames: Sequence[str] | None,
    change: Callable[[Sample, np.ndarray], list[Annotation] | None],
    progress: Progress | None,
) -> int:
    """Run ``change`` on the chosen frames and save. Returns how many frames changed."""
    import cv2

    folder = Path(folder)
    labels = load_folder(folder)
    if labels.format != "labels-json":
        raise LabelError(f"{folder} has no {LABELS_FILE}; only outlined folders can be edited")
    chosen = None if frames is None else set(frames)
    todo = [s for s in labels.samples if chosen is None or s.image.name in chosen]
    stored = read_elimination(folder)
    changed = 0
    result: dict[str, Sample] = {}
    for number, sample in enumerate(todo, start=1):
        image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
        if image is None:
            raise LabelError(f"could not read image {sample.image}")
        annotations = change(sample, image)
        if annotations is not None:
            if stored is not None:
                annotations = eliminate_sample(
                    annotations,
                    stored["boxes"].get(sample.image.name, []),
                    image.shape[:2],
                    ELIMINATION_SETTINGS,
                    stored["target"],
                )
            result[sample.image.name] = Sample(sample.image, tuple(annotations), sample.group)
            changed += 1
        if progress is not None:
            progress(number, len(todo))
    if changed:
        shutil.copyfile(folder / LABELS_FILE, folder / UNDO_FILE)
        write_folder(folder, [result.get(s.image.name, s) for s in labels.samples])
    return changed


def add_outline(
    folder: Path,
    label: str,
    point: Sequence[float],
    masker: PointMasker,
    frames: Sequence[str] | None = None,
    progress: Progress | None = None,
) -> int:
    """Outline what is at ``point`` in each of ``frames`` (None = all) and label it."""

    def change(sample: Sample, image: np.ndarray) -> list[Annotation] | None:
        import cv2

        height, width = image.shape[:2]
        polygons, _ = mask_to_polygons(masker(image, point), (0, 0, width, height), CLICK_SETTINGS)
        if not polygons:
            return None
        # What a person outlines comes out of the model's space and Earth outlines.
        shape = (height, width)
        clicked = np.zeros(shape, np.uint8)
        for points in polygons:
            cv2.fillPoly(clicked, [np.array(points, np.int32)], 255)
        current: list[Annotation] = list(sample.annotations)
        for other in ("space", "earth"):
            if other == label.lower():
                continue
            theirs = [
                a
                for a in current
                if a.label.lower() == other
                and a.kind == KIND_POLYGON
                and not a.source.startswith(SOURCE_HUMAN)
            ]
            before = fill_annotations(shape, theirs, [other]) > 0
            if (before & (clicked > 0)).any():
                source = theirs[0].source
                current = carve(
                    current, shape, other, before & (clicked == 0), ELIMINATION_SETTINGS, source
                )
        added = []
        for points in polygons:
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            added.append(
                Annotation(
                    label,
                    (min(xs), min(ys), max(xs), max(ys)),
                    KIND_POLYGON,
                    points,
                    source=CLICK_SOURCE,
                )
            )
        return [*current, *added]

    return _apply(folder, frames, change, progress)


def outline_at(sample: Sample, point: Sequence[float]) -> Annotation | None:
    """The smallest outline under ``point`` that a person can remove, if any."""
    under = [
        a
        for a in sample.annotations
        if a.kind == KIND_POLYGON and a.source != SOURCE_ELIMINATION and _inside(a.points, point)
    ]
    return min(under, key=_area) if under else None


def remove_outline(
    folder: Path,
    point: Sequence[float],
    label: str | None = None,
    frames: Sequence[str] | None = None,
    progress: Progress | None = None,
) -> int:
    """Take away the smallest outline under ``point`` (of ``label``, if given)."""

    def change(sample: Sample, image: np.ndarray) -> list[Annotation] | None:
        candidates = replace(
            sample,
            annotations=tuple(
                a for a in sample.annotations if label is None or a.label.lower() == label.lower()
            ),
        )
        found = outline_at(candidates, point)
        if found is None:
            return None
        return [a for a in sample.annotations if a is not found]

    return _apply(folder, frames, change, progress)


def can_undo(folder: Path) -> bool:
    return (Path(folder) / UNDO_FILE).is_file()


def undo(folder: Path) -> bool:
    """Put the labels back as they were before the last edit. False if there is none."""
    folder = Path(folder)
    if not can_undo(folder):
        return False
    shutil.move(folder / UNDO_FILE, folder / LABELS_FILE)
    return True
