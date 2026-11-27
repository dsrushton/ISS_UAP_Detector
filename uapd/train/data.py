"""Turn label folders into a training plan: class ids, a split and a report.

Nothing here needs PyTorch, so ``uapd train --check`` and the app can show what
a run would train on without loading it.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from uapd.labels import (
    KIND_POLYGON,
    FolderLabels,
    LabelError,
    Sample,
    find_label_folders,
    load_folder,
)
from uapd.train.config import TrainConfig

Box = tuple[float, float, float, float]


class DataError(ValueError):
    """Raised when the data cannot be trained on."""


@dataclass(frozen=True)
class Item:
    """One image with its boxes as class ids (1 = first class; 0 is background)."""

    image: Path
    boxes: tuple[Box, ...]
    labels: tuple[int, ...]
    # Outline of each box as (x, y) points, or () where the label is only a box.
    # Empty when no label of the image has an outline.
    outlines: tuple[tuple[tuple[float, float], ...], ...] = ()


@dataclass
class DataPlan:
    classes: tuple[str, ...]
    train: list[Item] = field(default_factory=list)
    val: list[Item] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def resolve_folders(paths: list[str | Path]) -> list[Path]:
    """Label folders for the given paths; a path without labels is searched below."""
    folders: list[Path] = []
    for path in paths:
        try:
            found = find_label_folders(path)
        except LabelError as exc:
            raise DataError(str(exc)) from exc
        if not found:
            raise DataError(f"no labelled frames in or below {path}")
        folders.extend(f for f in found if f not in folders)
    if not folders:
        raise DataError("no data folders given")
    return folders


def plan_data(paths: list[str | Path], config: TrainConfig) -> DataPlan:
    folders = resolve_folders(paths)
    class_ids = {name: index for index, name in enumerate(config.classes, start=1)}
    lookup = {name.lower(): name for name in config.classes}
    lookup.update(
        {alias.lower(): target for alias, target in config.aliases.items() if target in class_ids}
    )

    # A bare name holds out every folder of that name; a path holds out that folder.
    val_names = {f for f in config.val_folders if Path(f).name == f}
    val_paths = {Path(f).resolve() for f in config.val_folders if Path(f).name != f}
    plan = DataPlan(classes=config.classes)
    boxes = {"train": Counter(), "val": Counter()}
    unknown: Counter[str] = Counter()
    bad_boxes = 0
    folder_reports = []
    digest = hashlib.sha256()

    for folder in folders:
        try:
            labels: FolderLabels = load_folder(folder)
        except LabelError as exc:
            raise DataError(str(exc)) from exc
        items = []
        for sample in labels.samples:
            item, dropped, bad = _item(sample, lookup, class_ids)
            unknown.update(dropped)
            bad_boxes += bad
            items.append(item)
            digest.update(f"{sample.image.name}:{item.boxes}:{item.labels}\n".encode())
            if item.outlines:
                digest.update(f"{item.outlines}\n".encode())

        if config.val_folders:
            held_out = folder.name in val_names or folder.resolve() in val_paths
            n_val = len(items) if held_out else 0
        else:
            n_val = math.floor(len(items) * config.val_fraction)
        train, val = items[: len(items) - n_val], items[len(items) - n_val :]
        plan.train.extend(train)
        plan.val.extend(val)
        for part, part_items in (("train", train), ("val", val)):
            boxes[part].update(config.classes[i - 1] for it in part_items for i in it.labels)
        folder_reports.append(
            {
                "folder": str(folder),
                "format": labels.format,
                "train_images": len(train),
                "val_images": len(val),
                "unlabelled_images": labels.unlabelled,
                "image_size": _image_size(items[0].image) if items else None,
                "problems": labels.problems[:20],
                "problem_count": len(labels.problems),
            }
        )

    if unknown and config.unknown_labels == "error":
        names = ", ".join(f"{name} ({count})" for name, count in unknown.most_common())
        raise DataError(f"labels that are not a class or an alias: {names}")

    plan.report = {
        "folders": folder_reports,
        "train_images": len(plan.train),
        "val_images": len(plan.val),
        "classes": {
            name: {"train_boxes": boxes["train"][name], "val_boxes": boxes["val"][name]}
            for name in config.classes
        },
        "dropped_unknown_labels": dict(unknown.most_common()),
        "dropped_bad_boxes": bad_boxes,
        "images_without_boxes": sum(1 for it in (*plan.train, *plan.val) if not it.boxes),
        "labels_sha256": digest.hexdigest(),
        "warnings": [],
    }
    plan.report["warnings"] = _warnings(plan, config)
    return plan


def _item(
    sample: Sample, lookup: dict[str, str], class_ids: dict[str, int]
) -> tuple[Item, list[str], int]:
    boxes, labels, outlines, dropped, bad = [], [], [], [], 0
    for annotation in sample.annotations:
        name = lookup.get(annotation.label.lower())
        if name is None:
            dropped.append(annotation.label)
            continue
        x1, y1, x2, y2 = annotation.box
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        if not (x2 - x1 >= 1.0 and y2 - y1 >= 1.0):
            bad += 1
            continue
        boxes.append((x1, y1, x2, y2))
        labels.append(class_ids[name])
        outlines.append(annotation.points if annotation.kind == KIND_POLYGON else ())
    if not any(outlines):
        outlines = []
    return Item(sample.image, tuple(boxes), tuple(labels), tuple(outlines)), dropped, bad


def _image_size(path: Path) -> list[int] | None:
    import cv2

    image = cv2.imread(str(path))
    return None if image is None else [int(image.shape[1]), int(image.shape[0])]


def _warnings(plan: DataPlan, config: TrainConfig) -> list[str]:
    report = plan.report
    out = []
    if report["dropped_unknown_labels"]:
        names = ", ".join(f"{n} ({c})" for n, c in report["dropped_unknown_labels"].items())
        out.append(
            f"Boxes left out because their label is not a class or an alias: {names}. "
            "Those areas are trained as background."
        )
    if report["dropped_bad_boxes"]:
        out.append(f"{report['dropped_bad_boxes']} boxes left out: under 1 px wide or high.")
    for name, counts in report["classes"].items():
        if counts["train_boxes"] == 0:
            out.append(f"Class {name!r} has no training boxes; the model cannot learn it.")
        elif counts["val_boxes"] == 0 and plan.val:
            out.append(f"Class {name!r} has no validation boxes; it will not be scored.")
    sizes = {tuple(f["image_size"]) for f in report["folders"] if f["image_size"]}
    if len(sizes) > 1:
        listed = ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
        out.append(
            f"Folders have different frame sizes ({listed}). The detector runs on the "
            "cropped frame; check every folder was cropped the same way."
        )
    formats = sorted({f["format"] for f in report["folders"]})
    if len(formats) > 1:
        out.append(
            f"Folders use different label sets ({', '.join(formats)}). The CSV labels are "
            "the older August 2024 set; check the folders that fell back to them."
        )
    unlabelled = sum(f["unlabelled_images"] for f in report["folders"])
    if unlabelled:
        out.append(f"{unlabelled} images have no label file and are not used.")
    problems = sum(f["problem_count"] for f in report["folders"])
    if problems:
        out.append(f"{problems} label rows could not be read; see the folder list.")
    if not plan.val:
        out.append("No validation images: the run cannot be scored; the last epoch is kept.")
    return out


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"{report['train_images']} training images, {report['val_images']} validation images, "
        f"{report['images_without_boxes']} with no boxes",
        "",
        f"{'class':<12}{'train boxes':>12}{'val boxes':>11}",
    ]
    for name, counts in report["classes"].items():
        lines.append(f"{name:<12}{counts['train_boxes']:>12}{counts['val_boxes']:>11}")
    lines.append("")
    for folder in report["folders"]:
        size = "x".join(str(v) for v in folder["image_size"]) if folder["image_size"] else "?"
        lines.append(
            f"{folder['folder']}: {folder['train_images']} train, {folder['val_images']} val, "
            f"{size} px, {folder['format']}"
        )
        lines.extend(f"    {problem}" for problem in folder["problems"])
    for warning in report["warnings"]:
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines)
