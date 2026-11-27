"""Detection scores: average precision per class and their mean (mAP).

Plain numpy, and classes are compared by name, so models with different class
lists can be scored on the same labelled images.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageBoxes:
    """Boxes of one image, predicted or labelled. ``scores`` is empty for labels."""

    boxes: np.ndarray  # N x 4, x1 y1 x2 y2
    names: tuple[str, ...]
    scores: tuple[float, ...] = ()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(right_bottom - left_top, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def average_precision(
    predictions: Sequence[ImageBoxes], truths: Sequence[ImageBoxes], name: str, iou: float
) -> float | None:
    """Area under the precision-recall curve for one class; None if it is never labelled."""
    scored: list[tuple[float, bool]] = []
    total = 0
    for predicted, truth in zip(predictions, truths, strict=True):
        mask = np.array([n == name for n in truth.names], dtype=bool)
        truth_boxes = np.asarray(truth.boxes, np.float64).reshape(-1, 4)[mask]
        total += len(truth_boxes)
        chosen = [i for i, n in enumerate(predicted.names) if n == name]
        chosen.sort(key=lambda i: predicted.scores[i], reverse=True)
        if not chosen:
            continue
        overlaps = iou_matrix(np.asarray(predicted.boxes).reshape(-1, 4)[chosen], truth_boxes)
        taken: set[int] = set()
        for row, index in enumerate(chosen):
            best = int(np.argmax(overlaps[row])) if len(truth_boxes) else -1
            hit = best >= 0 and overlaps[row, best] >= iou and best not in taken
            if hit:
                taken.add(best)
            scored.append((predicted.scores[index], hit))
    if total == 0:
        return None
    if not scored:
        return 0.0
    scored.sort(key=lambda pair: pair[0], reverse=True)
    hits = np.cumsum([hit for _, hit in scored])
    precision = hits / np.arange(1, len(scored) + 1)
    recall = hits / total
    # Precision at each recall level is the best precision at that recall or beyond.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision))


def score(
    predictions: Sequence[ImageBoxes],
    truths: Sequence[ImageBoxes],
    classes: Sequence[str],
    iou: float = 0.5,
) -> dict[str, Any]:
    """mAP over the classes that have labelled boxes, with each class's AP."""
    per_class = {name: average_precision(predictions, truths, name, iou) for name in classes}
    scored = [value for value in per_class.values() if value is not None]
    return {
        "map": round(float(np.mean(scored)), 4) if scored else None,
        "iou_threshold": iou,
        "ap": {name: None if v is None else round(v, 4) for name, v in per_class.items()},
        "images": len(truths),
    }
