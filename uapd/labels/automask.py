"""Turn box labels into outline labels with a promptable mask model (SAM 2).

Each labelled box of a mask class (space, earth, iss by default) is given to the
mask model as a prompt; the shape it finds inside the box becomes one or more
``polygon`` annotations. Boxes of other classes are copied as they are. The
result is a new folder per input folder, holding a copy of the images and a
``labels.json``, so the original labels are never touched.

The mask model is a labelling aid here, not part of the detector. Its output
needs a look before training: ``build_masks`` writes a contact sheet per folder
and reports how much of each box its outline fills.

Polygons are outer outlines only. A space region that surrounds part of the
station comes out as one outline that includes it; whoever uses the masks takes
the other classes away from space.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from uapd.labels import (
    KIND_BOX,
    KIND_POLYGON,
    Annotation,
    LabelError,
    Sample,
    load_folder,
)
from uapd.labels.store import LABELS_FILE, write_folder

DEFAULT_MASK_CLASSES = ("space", "earth", "iss")
DEFAULT_MODEL = "sam2.1_b.pt"
REVIEW_FOLDER = "_review"

# A mask model for one image: (BGR image, boxes as x1 y1 x2 y2) -> one boolean
# mask per box, each the size of the image.
Masker = Callable[[np.ndarray, Sequence[Sequence[float]]], np.ndarray]


@dataclass(frozen=True)
class MaskSettings:
    mask_classes: tuple[str, ...] = DEFAULT_MASK_CLASSES
    # Specks and pinholes up to this size (pixels across) are smoothed away.
    smooth: int = 5
    # Pieces smaller than this share of the box are dropped.
    min_piece: float = 0.01
    # An outline that fills less than this share of its box is not trusted; the
    # box is kept as a box instead.
    min_fill: float = 0.05
    # Outline points closer to a straight line than this many pixels are removed.
    simplify: float = 1.5


@dataclass
class FolderResult:
    source: Path
    out: Path
    images: int = 0
    outlined: int = 0  # boxes replaced by outlines
    kept_boxes: int = 0  # boxes of mask classes the model could not outline
    # Share of the box covered by its outline, per class.
    fill: dict[str, list[float]] = field(default_factory=dict)
    skipped: bool = False


def mask_to_polygons(
    mask: np.ndarray, box: Sequence[float], settings: MaskSettings
) -> tuple[list[tuple[tuple[float, float], ...]], float]:
    """Outer outlines of ``mask`` inside ``box``, and the share of the box they cover."""
    import cv2

    height, width = mask.shape[:2]
    x1, y1 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
    x2, y2 = min(int(np.ceil(box[2])), width), min(int(np.ceil(box[3])), height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return [], 0.0
    inside = np.zeros((height, width), np.uint8)
    inside[y1:y2, x1:x2] = mask[y1:y2, x1:x2].astype(np.uint8) * 255
    if settings.smooth > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (settings.smooth, settings.smooth))
        inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE, kernel)
        inside = cv2.morphologyEx(inside, cv2.MORPH_OPEN, kernel)

    box_area = float((x2 - x1) * (y2 - y1))
    contours, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons, covered = [], 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < settings.min_piece * box_area:
            break
        approx = cv2.approxPolyDP(contour, settings.simplify, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        polygons.append(tuple((float(x), float(y)) for x, y in approx))
        covered += area
    return polygons, covered / box_area


def outline_sample(
    sample: Sample, image: np.ndarray, masker: Masker, settings: MaskSettings, source: str
) -> tuple[list[Annotation], list[tuple[str, float]], int]:
    """New annotations for one image: outlines where the model found a shape.

    Returns the annotations, (class, fill) for each outlined box, and how many
    boxes of mask classes were kept as boxes.
    """
    wanted = {name.lower() for name in settings.mask_classes}
    prompts = [
        index
        for index, a in enumerate(sample.annotations)
        if a.kind == KIND_BOX and a.label.lower() in wanted
    ]
    masks = masker(image, [sample.annotations[i].box for i in prompts]) if prompts else []
    outlines = dict(zip(prompts, masks, strict=True))

    out: list[Annotation] = []
    fills: list[tuple[str, float]] = []
    kept = 0
    for index, annotation in enumerate(sample.annotations):
        if index not in outlines:
            out.append(annotation)
            continue
        polygons, fill = mask_to_polygons(outlines[index], annotation.box, settings)
        if not polygons or fill < settings.min_fill:
            out.append(annotation)
            kept += 1
            continue
        fills.append((annotation.label.lower(), fill))
        for points in polygons:
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            out.append(
                replace(
                    annotation,
                    kind=KIND_POLYGON,
                    box=(min(xs), min(ys), max(xs), max(ys)),
                    points=points,
                    source=source,
                )
            )
    return out, fills, kept


def sam_masker(model: str = DEFAULT_MODEL, device: str | None = None) -> Masker:
    """SAM 2 through the ultralytics package, prompted with boxes."""
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise LabelError(
            f"outlining boxes needs the ultralytics package: {exc} "
            "(install with: pip install -e '.[yolo]')"
        ) from exc
    sam = SAM(model)

    def run(image: np.ndarray, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        result = sam(image, bboxes=[list(b) for b in boxes], device=device, verbose=False)[0]
        if result.masks is None:
            return np.zeros((len(boxes), *image.shape[:2]), bool)
        masks = result.masks.data.cpu().numpy().astype(bool)
        if len(masks) != len(boxes):
            raise LabelError(f"the mask model returned {len(masks)} masks for {len(boxes)} boxes")
        return masks

    return run


def build_masks(
    folders: Sequence[Path],
    out_root: Path,
    masker: Masker,
    settings: MaskSettings | None = None,
    source: str = "model:sam2",
    force: bool = False,
    log: Callable[[str], None] = print,
    review_frames: int = 12,
    limit: int | None = None,
) -> list[FolderResult]:
    """Write an outlined copy of each label folder below ``out_root``.

    Folders keep their path relative to the folder all of them share. A folder
    whose copy already has labels is skipped unless ``force`` is set. ``limit``
    takes only the first images of each folder, for a trial.
    """
    import cv2

    settings = settings or MaskSettings()
    folders = [Path(f).resolve() for f in folders]
    common = Path(os.path.commonpath(folders)) if len(folders) > 1 else folders[0].parent
    results = []
    for folder in folders:
        target = Path(out_root) / folder.relative_to(common)
        result = FolderResult(folder, target)
        results.append(result)
        if (target / LABELS_FILE).is_file() and not force:
            result.skipped = True
            log(f"{target}: already done, skipped")
            continue
        labels = load_folder(folder)
        if limit is not None:
            labels.samples = labels.samples[:limit]
        target.mkdir(parents=True, exist_ok=True)
        samples, sheet = [], []
        step = max(len(labels.samples) // max(review_frames, 1), 1)
        for number, sample in enumerate(labels.samples):
            image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
            if image is None:
                raise LabelError(f"could not read image {sample.image}")
            annotations, fills, kept = outline_sample(sample, image, masker, settings, source)
            copy = target / sample.image.name
            shutil.copyfile(sample.image, copy)
            samples.append(Sample(copy, tuple(annotations), group=str(target)))
            result.images += 1
            result.outlined += len(fills)
            result.kept_boxes += kept
            for name, fill in fills:
                result.fill.setdefault(name, []).append(fill)
            if review_frames and number % step == 0 and len(sheet) < review_frames:
                sheet.append(draw_annotations(image, annotations))
        write_folder(target, samples)
        if sheet:
            review = Path(out_root) / REVIEW_FOLDER
            review.mkdir(parents=True, exist_ok=True)
            name = "__".join(folder.relative_to(common).parts) + ".jpg"
            cv2.imwrite(str(review / name), contact_sheet(sheet))
        log(format_result(result))
    return results


def format_result(result: FolderResult) -> str:
    fills = ", ".join(
        f"{name} {np.median(values):.0%}" for name, values in sorted(result.fill.items())
    )
    return (
        f"{result.out}: {result.images} images, {result.outlined} boxes outlined, "
        f"{result.kept_boxes} left as boxes; median share of box filled: {fills or '-'}"
    )


COLOURS: dict[str, tuple[int, int, int]] = {
    "space": (0, 0, 255),
    "earth": (255, 120, 0),
    "iss": (0, 255, 0),
    "lf": (255, 0, 255),
    "sun": (0, 255, 255),
    "panel": (0, 160, 255),
}


def draw_annotations(image: np.ndarray, annotations: Sequence[Annotation]) -> np.ndarray:
    """The image with outlines filled in and boxes drawn, for a person to check."""
    import cv2

    out = image.copy()
    tint = image.copy()
    for a in annotations:
        if a.kind == KIND_POLYGON:
            colour = COLOURS.get(a.label.lower(), (200, 200, 200))
            cv2.fillPoly(tint, [np.array(a.points, np.int32)], colour)
    out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)
    for a in annotations:
        colour = COLOURS.get(a.label.lower(), (200, 200, 200))
        if a.kind == KIND_POLYGON:
            cv2.polylines(out, [np.array(a.points, np.int32)], True, colour, 2)
        else:
            x1, y1, x2, y2 = (int(v) for v in a.box)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)
            cv2.putText(out, a.label, (x1 + 4, y1 + 16), 0, 0.5, colour, 1)
    return out


def contact_sheet(images: Sequence[np.ndarray], columns: int = 4, width: int = 480) -> np.ndarray:
    import cv2

    tiles = []
    for image in images:
        height = round(image.shape[0] * width / image.shape[1])
        tiles.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    tile_height = max(t.shape[0] for t in tiles)
    rows = []
    for start in range(0, len(tiles), columns):
        row = [
            cv2.copyMakeBorder(t, 0, tile_height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            for t in tiles[start : start + columns]
        ]
        while len(row) < columns:
            row.append(np.zeros((tile_height, width, 3), np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def summary(results: Sequence[FolderResult]) -> dict[str, Any]:
    done = [r for r in results if not r.skipped]
    return {
        "folders": len(done),
        "skipped": len(results) - len(done),
        "images": sum(r.images for r in done),
        "outlined": sum(r.outlined for r in done),
        "kept_boxes": sum(r.kept_boxes for r in done),
    }
