"""The native label format: one ``labels.json`` per folder of images.

    {
      "schema_version": 1,
      "images": {
        "frame_000120.jpg": {
          "annotations": [
            {"kind": "box", "label": "space", "box": [0, 0, 939, 410], "source": "human"}
          ]
        }
      }
    }

An image listed with an empty ``annotations`` list is labelled as containing
nothing. An image in the folder that is not listed is unlabelled.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from uapd.labels import (
    IMAGE_SUFFIXES,
    KIND_BOX,
    KIND_MASK,
    KIND_POLYGON,
    Annotation,
    FolderLabels,
    LabelError,
    Sample,
)

FORMAT = "labels-json"
LABELS_FILE = "labels.json"
SCHEMA_VERSION = 1
KINDS = (KIND_BOX, KIND_POLYGON, KIND_MASK)


def read_folder(directory: Path) -> FolderLabels:
    path = directory / LABELS_FILE
    try:
        with open(path, encoding="utf-8") as fh:
            images = json.load(fh)["images"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LabelError(f"could not read {path}: {exc!r}") from exc

    out = FolderLabels(format=FORMAT)
    for name in sorted(images):
        image = directory / name
        if not image.is_file():
            out.problems.append(f"{name}: image file is missing")
            continue
        annotations = []
        for index, item in enumerate(images[name].get("annotations", [])):
            try:
                annotations.append(_annotation(item))
            except (KeyError, TypeError, ValueError) as exc:
                out.problems.append(f"{name} annotation {index}: {exc!r}")
        out.samples.append(Sample(image, tuple(annotations), group=str(directory)))
    on_disk = sum(1 for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    out.unlabelled = max(on_disk - len(out.samples), 0)
    return out


def write_folder(directory: Path, samples: Iterable[Sample]) -> Path:
    """Write ``labels.json`` for images that are in ``directory``."""
    images: dict[str, Any] = {}
    for sample in samples:
        if sample.image.parent.resolve() != Path(directory).resolve():
            raise LabelError(f"{sample.image} is not in {directory}")
        images[sample.image.name] = {"annotations": [_record(a) for a in sample.annotations]}
    path = Path(directory) / LABELS_FILE
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"schema_version": SCHEMA_VERSION, "images": images}, fh, indent=1)
        fh.write("\n")
    return path


def _annotation(item: dict[str, Any]) -> Annotation:
    kind = item.get("kind", KIND_BOX)
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    x1, y1, x2, y2 = (float(v) for v in item["box"])
    return Annotation(
        label=str(item["label"]),
        box=(x1, y1, x2, y2),
        kind=kind,
        points=tuple((float(x), float(y)) for x, y in item.get("points", ())),
        mask_file=item.get("mask_file"),
        source=str(item.get("source", "imported")),
    )


def _record(annotation: Annotation) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": annotation.kind,
        "label": annotation.label,
        "box": list(annotation.box),
        "source": annotation.source,
    }
    if annotation.points:
        record["points"] = [list(p) for p in annotation.points]
    if annotation.mask_file:
        record["mask_file"] = annotation.mask_file
    return record
