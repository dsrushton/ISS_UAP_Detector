"""Turn box labels into outline labels with a promptable mask model (SAM 2).

Each labelled box of a mask class (space, earth, iss by default) is given to the
mask model as a prompt; the shape it finds inside the box becomes one or more
``polygon`` annotations. Boxes of other classes are copied as they are. The
result is a new folder per input folder, holding a copy of the images and a
``labels.json``, so the original labels are never touched.

The mask model is a labelling aid here, not part of the detector. Its output
needs a look before training: ``build_masks`` writes a contact sheet per folder
and reports how much of each box its outline fills.

The station is better found by elimination than by the mask model, which picks
out one module of it: ``eliminate`` makes the station everything inside its box
that is not space or Earth.

An outline with a hole in it (the station around a pocket of space) is stored as
one polygon that runs along the outer edge, in to the hole, round it and back out
the same way. Filling such a polygon leaves the hole empty.
"""

from __future__ import annotations

import json
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
    smallest = settings.min_piece * box_area
    contours, hierarchy = cv2.findContours(inside, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], 0.0
    # hierarchy rows are (next, previous, first child, parent); holes have a parent.
    parents = hierarchy[0][:, 3]
    outer = [i for i in range(len(contours)) if parents[i] < 0]
    polygons, covered = [], 0.0
    for index in sorted(outer, key=lambda i: cv2.contourArea(contours[i]), reverse=True):
        area = cv2.contourArea(contours[index])
        if area < smallest:
            break
        points = cv2.approxPolyDP(contours[index], settings.simplify, True).reshape(-1, 2)
        if len(points) < 3:
            continue
        for hole in (i for i in range(len(contours)) if parents[i] == index):
            hole_area = cv2.contourArea(contours[hole])
            if hole_area < smallest:
                continue
            inner = cv2.approxPolyDP(contours[hole], settings.simplify, True).reshape(-1, 2)
            if len(inner) >= 3:
                points = _cut_in(points, inner)
                area -= hole_area
        polygons.append(tuple((float(x), float(y)) for x, y in points))
        covered += area
    return polygons, covered / box_area


def _cut_in(outer: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """One polygon for ``outer`` with ``hole`` cut out of it, joined where they are closest."""
    gaps = ((outer[:, None, :].astype(np.int64) - hole[None, :, :]) ** 2).sum(axis=2)
    a, b = np.unravel_index(int(gaps.argmin()), gaps.shape)
    return np.concatenate([outer[: a + 1], hole[b:], hole[: b + 1], outer[a:]])


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


ELIMINATION_FILE = "elimination.json"
BACKUP_FILE = "labels.before-elimination.json"
SOURCE_ELIMINATION = "elimination"
SOURCE_HUMAN = "human"


def fill_annotations(
    shape: tuple[int, int], annotations: Sequence[Annotation], labels: Sequence[str]
) -> np.ndarray:
    """A mask (255 inside) of the outlines of ``labels`` among ``annotations``."""
    import cv2

    wanted = {name.lower() for name in labels}
    mask = np.zeros(shape, np.uint8)
    for a in annotations:
        if a.kind == KIND_POLYGON and a.label.lower() in wanted and len(a.points) >= 3:
            cv2.fillPoly(mask, [np.array(a.points, np.int32)], 255)
    return mask


def eliminate_sample(
    annotations: Sequence[Annotation],
    boxes: Sequence[Sequence[float]],
    shape: tuple[int, int],
    settings: MaskSettings,
    target: str = "iss",
    minus: Sequence[str] = ("space", "earth"),
) -> list[Annotation]:
    """``annotations`` with ``target`` redone as: its boxes, less the ``minus`` outlines."""
    height, width = shape
    area = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in boxes:
        area[max(int(y1), 0) : int(np.ceil(y2)), max(int(x1), 0) : int(np.ceil(x2))] = 255
    area[fill_annotations(shape, annotations, minus) > 0] = 0
    polygons, _ = mask_to_polygons(area > 0, (0, 0, width, height), settings)
    # Outlines a person added for the target are kept beside what is left over.
    out = [
        a
        for a in annotations
        if a.label.lower() != target.lower() or a.source.startswith(SOURCE_HUMAN)
    ]
    for points in polygons:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        out.append(
            Annotation(
                target,
                (min(xs), min(ys), max(xs), max(ys)),
                KIND_POLYGON,
                points,
                source=SOURCE_ELIMINATION,
            )
        )
    return out


def read_elimination(folder: Path) -> dict[str, Any] | None:
    try:
        with open(Path(folder) / ELIMINATION_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def eliminate_folder(
    folder: Path,
    source: Path | None = None,
    settings: MaskSettings | None = None,
    target: str = "iss",
    minus: Sequence[str] = ("space", "earth"),
    review_frames: int = 0,
) -> tuple[int, list[np.ndarray]]:
    """Redo ``target`` in an outlined folder by elimination. Returns images changed.

    The boxes of ``target`` come from ``source`` (the folder the outlines were
    made from) the first time, and are kept in ``elimination.json`` so that later
    runs, after outlines were added or removed, need only the folder itself. The
    labels as they were before the first run are kept in a backup file.
    """
    import cv2

    folder = Path(folder)
    settings = settings or MaskSettings(min_piece=0.002)
    stored = read_elimination(folder)
    if stored is None or stored.get("target") != target:
        if source is None:
            raise LabelError(
                f"{folder}: the boxes of {target!r} are needed; give the source folder"
            )
        boxes = {
            sample.image.name: [
                list(a.box) for a in sample.annotations if a.label.lower() == target
            ]
            for sample in load_folder(source).samples
        }
        stored = {"schema_version": 1, "target": target, "source": str(source), "boxes": boxes}
        with open(folder / ELIMINATION_FILE, "w", encoding="utf-8") as fh:
            json.dump(stored, fh)
    labels = load_folder(folder)
    if labels.format != "labels-json":
        raise LabelError(f"{folder} is not an outlined folder (no {LABELS_FILE})")
    if not (folder / BACKUP_FILE).exists():
        shutil.copyfile(folder / LABELS_FILE, folder / BACKUP_FILE)

    shape: tuple[int, int] | None = None
    samples, sheet = [], []
    step = max(len(labels.samples) // max(review_frames, 1), 1)
    for number, sample in enumerate(labels.samples):
        wanted = bool(review_frames) and number % step == 0 and len(sheet) < review_frames
        image = None
        if shape is None or wanted:
            image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
            if image is None:
                raise LabelError(f"could not read image {sample.image}")
            shape = image.shape[:2]  # the frames of one folder share a size
        boxes = stored["boxes"].get(sample.image.name, [])
        annotations = eliminate_sample(sample.annotations, boxes, shape, settings, target, minus)
        samples.append(Sample(sample.image, tuple(annotations), sample.group))
        if wanted and image is not None:
            sheet.append(draw_annotations(image, annotations))
    write_folder(folder, samples)
    return len(samples), sheet


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


def common_root(folders: Sequence[Path]) -> Path:
    resolved = [Path(f).resolve() for f in folders]
    return Path(os.path.commonpath(resolved)) if len(resolved) > 1 else resolved[0].parent


def output_folders(folders: Sequence[Path], out_root: Path) -> list[tuple[Path, Path]]:
    """Each source folder with its outlined copy: the path below what the sources share."""
    common = common_root(folders)
    out_root = Path(out_root).resolve()
    return [(Path(f).resolve(), out_root / Path(f).resolve().relative_to(common)) for f in folders]


SOURCE_DARK = "dark-in-box"
# Space is dark. Brightest colour channel below this counts as dark.
DARK_BELOW = 40
# A mask-model "space" outline with more than this share of bright pixels is an
# outline of hardware: in a box of sky with a module in it, the module is the
# object the model picks.
HARDWARE_SHARE = 0.25


def dark_pixels(image: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Boolean mask of the dark parts of a BGR image, without specks."""
    import cv2

    dark = (image.max(axis=2) < DARK_BELOW).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth, smooth))
    dark = cv2.morphologyEx(cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    return dark > 0


def carve(
    annotations: Sequence[Annotation],
    shape: tuple[int, int],
    label: str,
    keep: np.ndarray,
    settings: MaskSettings,
    source: str,
) -> list[Annotation]:
    """``annotations`` with the model-made outlines of ``label`` replaced by ``keep``."""
    height, width = shape
    out = [
        a
        for a in annotations
        if not (
            a.label.lower() == label
            and a.kind == KIND_POLYGON
            and not a.source.startswith(SOURCE_HUMAN)
        )
    ]
    polygons, _ = mask_to_polygons(keep, (0, 0, width, height), settings)
    for points in polygons:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        out.append(
            Annotation(
                label, (min(xs), min(ys), max(xs), max(ys)), KIND_POLYGON, points, source=source
            )
        )
    return out


def fix_space_sample(
    annotations: Sequence[Annotation],
    image: np.ndarray,
    space_boxes: Sequence[Sequence[float]],
    settings: MaskSettings,
) -> list[Annotation] | None:
    """Redo the mask model's space outlines so that space is dark sky.

    An outline that is mostly dark is kept, trimmed to its dark pixels. One that is
    mostly bright is hardware: space is then the dark part of the labelled space
    boxes it lies in, without that hardware. Earth outlines are taken out. Returns
    None when the image has no space outline made by the mask model.
    """
    import cv2

    shape = image.shape[:2]
    dark = dark_pixels(image)
    good = np.zeros(shape, bool)
    hardware = np.zeros(shape, bool)
    found = False
    for a in annotations:
        if (
            a.label.lower() != "space"
            or a.kind != KIND_POLYGON
            or not a.source.startswith("model:")
        ):
            continue
        found = True
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.array(a.points, np.int32)], 255)
        mask = mask > 0
        if not mask.any():
            continue
        if (~dark[mask]).mean() > HARDWARE_SHARE:
            hardware |= mask
        else:
            good |= mask
    if not found:
        return None
    space = good & dark
    for x1, y1, x2, y2 in space_boxes:
        box = np.zeros(shape, bool)
        box[max(int(y1), 0) : int(np.ceil(y2)), max(int(x1), 0) : int(np.ceil(x2))] = True
        if (hardware & box).any():
            space |= box & dark & ~hardware
    space &= fill_annotations(shape, annotations, ["earth"]) == 0
    return carve(annotations, shape, "space", space, settings, SOURCE_DARK)


def fix_space_folder(
    folder: Path, source: Path, review_frames: int = 0
) -> tuple[int, list[np.ndarray]]:
    """Redo the space outlines of an outlined folder (see ``fix_space_sample``), then
    the class found by elimination. ``source`` is the folder the outlines were made
    from, for its space boxes. Returns the number of images changed."""
    import cv2

    folder = Path(folder)
    labels = load_folder(folder)
    if labels.format != "labels-json":
        raise LabelError(f"{folder} is not an outlined folder (no {LABELS_FILE})")
    boxes = {
        sample.image.name: [a.box for a in sample.annotations if a.label.lower() == "space"]
        for sample in load_folder(source).samples
    }
    stored = read_elimination(folder)
    settings = MaskSettings(min_piece=0.002)
    samples, sheet, changed = [], [], 0
    step = max(len(labels.samples) // max(review_frames, 1), 1)
    for number, sample in enumerate(labels.samples):
        image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
        if image is None:
            raise LabelError(f"could not read image {sample.image}")
        annotations = fix_space_sample(
            sample.annotations, image, boxes.get(sample.image.name, []), settings
        )
        if annotations is None:
            annotations = list(sample.annotations)
        else:
            changed += 1
            if stored is not None:
                annotations = eliminate_sample(
                    annotations,
                    stored["boxes"].get(sample.image.name, []),
                    image.shape[:2],
                    settings,
                    stored["target"],
                )
        samples.append(Sample(sample.image, tuple(annotations), sample.group))
        if review_frames and number % step == 0 and len(sheet) < review_frames:
            sheet.append(draw_annotations(image, annotations))
    if changed:
        shutil.copyfile(folder / LABELS_FILE, folder / "labels.before-space-fix.json")
        write_folder(folder, samples)
    return changed, sheet


def sam_point_masker(model: str = DEFAULT_MODEL, device: str | None = None):
    """SAM 2 prompted with one point: (BGR image, (x, y)) -> boolean mask of what is there."""
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise LabelError(
            f"outlining by click needs the ultralytics package: {exc} "
            "(install with: pip install -e '.[yolo]')"
        ) from exc
    sam = SAM(model)

    def run(image: np.ndarray, point: Sequence[float]) -> np.ndarray:
        result = sam(
            image,
            points=[[float(point[0]), float(point[1])]],
            labels=[1],
            device=device,
            verbose=False,
        )[0]
        if result.masks is None or len(result.masks.data) == 0:
            return np.zeros(image.shape[:2], bool)
        return result.masks.data[0].cpu().numpy().astype(bool)

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
    results = []
    common = common_root(folders)
    for folder, target in output_folders(folders, Path(out_root)):
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
