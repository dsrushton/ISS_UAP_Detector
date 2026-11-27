"""labelImg (Pascal VOC) labels: one ``<image name>.xml`` per frame.

This is the format ``best_map_model.pth`` was trained on. The XML files are
looked for in a ``2ndTrainingXMLs`` subfolder of the frame folder (where the
January 2025 training script read them), then beside the frames.

A frame with an XML that has no objects is labelled as containing nothing. A
frame with no XML is unlabelled and is left out.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from uapd.labels import IMAGE_SUFFIXES, Annotation, FolderLabels, Sample

FORMAT = "voc-xml"
# Where the XML files may be, relative to the frame folder; the first that has any wins.
XML_FOLDERS = ("2ndTrainingXMLs", ".")


def _images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def xml_folder(directory: Path) -> Path | None:
    """The folder holding this frame folder's XML files, if it has any."""
    images = None
    for name in XML_FOLDERS:
        folder = directory / name
        if not folder.is_dir() or not any(folder.glob("*.xml")):
            continue
        images = _images(directory) if images is None else images
        if any((folder / f"{image.stem}.xml").is_file() for image in images):
            return folder
    return None


def has_labels(directory: Path) -> bool:
    return xml_folder(directory) is not None


def read_folder(directory: Path) -> FolderLabels:
    folder = xml_folder(directory)
    where = "" if folder == directory / "." else f" ({folder.name})"
    out = FolderLabels(format=FORMAT + where)
    for image in _images(directory):
        path = folder / f"{image.stem}.xml"
        if not path.is_file():
            out.unlabelled += 1
            continue
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as exc:
            out.problems.append(f"{path.name}: {exc}")
            continue
        annotations = []
        for number, item in enumerate(root.iter("object"), start=1):
            try:
                label = (item.findtext("name") or "").strip()
                box = item.find("bndbox")
                corners = tuple(float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            except (AttributeError, TypeError, ValueError):
                out.problems.append(f"{path.name} object {number}: no usable box")
                continue
            if not label:
                out.problems.append(f"{path.name} object {number}: no name")
                continue
            annotations.append(Annotation(label=label, box=corners))
        out.samples.append(Sample(image, tuple(annotations), group=str(directory)))
    return out
