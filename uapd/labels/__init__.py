"""Labels: what a person (or an import) says is in an image.

This is the contract between labelling and training. Labelling tools write
labels; trainers read ``Sample`` objects and never look at files themselves, so
a new label format or a new model type touches one side only.

An annotation has a ``kind``. Box trainers use boxes. Polygons and masks are
reserved for segmentation models: every annotation carries a bounding box, so a
box trainer can still use them.

Folder formats (``load_folder`` picks by what the folder contains):

* ``labels.json`` - the native format, written by ``write_folder`` and, later, by
  the relabel tool. See ``uapd.labels.store``.
* ``<name>.xml`` per frame, from labelImg - what ``best_map_model.pth`` was
  trained on (January 2025). See ``uapd.labels.voc_xml``.
* ``bboxes_<n>.csv`` beside ``<name>_<n>.jpg`` - the older label set of the
  August 2024 notebook. Used only when a folder has neither of the above.
  See ``uapd.labels.csv_frames``.

All coordinates are pixels of the image file. Label names are stored as written;
mapping them to a model's classes is the trainer's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

KIND_BOX = "box"
KIND_POLYGON = "polygon"
KIND_MASK = "mask"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


class LabelError(ValueError):
    """Raised when a label folder cannot be read."""


@dataclass(frozen=True)
class Annotation:
    label: str
    # x1, y1, x2, y2
    box: tuple[float, float, float, float]
    kind: str = KIND_BOX
    # Polygon outline as (x, y) points, for kind "polygon".
    points: tuple[tuple[float, float], ...] = ()
    # Mask image beside the labelled image, for kind "mask".
    mask_file: str | None = None
    # Who made it: "human", "imported" or "model:<weights hash>".
    source: str = "imported"


@dataclass(frozen=True)
class Sample:
    image: Path
    annotations: tuple[Annotation, ...]
    # Frames that belong together (one clip). A split keeps nearby frames of a
    # group on the same side.
    group: str


@dataclass
class FolderLabels:
    samples: list[Sample] = field(default_factory=list)
    format: str = ""
    # Images in the folder that have no labels at all. They are not negatives.
    unlabelled: int = 0
    # Rows that could not be used, as "file: reason".
    problems: list[str] = field(default_factory=list)


def load_folder(directory: str | Path) -> FolderLabels:
    """The labelled images of one folder, sorted by file name."""
    from uapd.labels import csv_frames, store, voc_xml

    directory = Path(directory)
    if not directory.is_dir():
        raise LabelError(f"not a folder: {directory}")
    if (directory / store.LABELS_FILE).is_file():
        return store.read_folder(directory)
    if voc_xml.has_labels(directory):
        return voc_xml.read_folder(directory)
    if csv_frames.has_labels(directory):
        return csv_frames.read_folder(directory)
    raise LabelError(f"no labels found in {directory}")


def find_label_folders(root: str | Path) -> list[Path]:
    """``root`` if it holds labels, otherwise every folder below it that does."""
    from uapd.labels import csv_frames, store, voc_xml

    root = Path(root)
    if not root.is_dir():
        raise LabelError(f"not a folder: {root}")

    def labelled(directory: Path) -> bool:
        return (
            (directory / store.LABELS_FILE).is_file()
            or voc_xml.has_labels(directory)
            or csv_frames.has_labels(directory)
        )

    if labelled(root):
        return [root]
    return sorted(d for d in root.rglob("*") if d.is_dir() and labelled(d))
