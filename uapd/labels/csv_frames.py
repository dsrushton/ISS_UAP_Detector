"""The label format of the original training notebook.

A folder of ``<name>_<n>.jpg`` frames, each with ``bboxes_<n>.csv`` beside it.
The CSV has a header and the columns ``x``, ``y``, ``width``, ``height`` and
``class`` (top-left corner and size, in pixels).

A frame with a CSV that has no rows is labelled as containing nothing. A frame
with no CSV is unlabelled and is left out.
"""

from __future__ import annotations

import csv
from pathlib import Path

from uapd.labels import IMAGE_SUFFIXES, Annotation, FolderLabels, Sample

FORMAT = "csv-frames"
COLUMNS = ("x", "y", "width", "height", "class")


def _csv_for(image: Path) -> Path:
    return image.with_name(f"bboxes_{image.stem.split('_')[-1]}.csv")


def has_labels(directory: Path) -> bool:
    return any(directory.glob("bboxes_*.csv"))


def read_folder(directory: Path) -> FolderLabels:
    out = FolderLabels(format=FORMAT)
    images = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    for image in images:
        path = _csv_for(image)
        if not path.is_file():
            out.unlabelled += 1
            continue
        annotations = []
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in COLUMNS if c not in (reader.fieldnames or ())]
            if missing and reader.fieldnames:
                out.problems.append(f"{path.name}: no column {', '.join(missing)}")
                continue
            for number, row in enumerate(reader, start=2):
                try:
                    x, y = float(row["x"]), float(row["y"])
                    w, h = float(row["width"]), float(row["height"])
                except (TypeError, ValueError):
                    out.problems.append(f"{path.name} line {number}: not a number")
                    continue
                label = (row["class"] or "").strip()
                if not label:
                    out.problems.append(f"{path.name} line {number}: no class")
                    continue
                annotations.append(Annotation(label=label, box=(x, y, x + w, y + h)))
        out.samples.append(Sample(image, tuple(annotations), group=str(directory)))
    return out
