"""Model manifests: the JSON file beside a weights file that says what the weights are.

``best.pth`` has its manifest in ``best.json``. The manifest names the network,
the classes in output order and the input the weights were trained on, so the
detector prepares frames the way the trainer did. Weights with no manifest are
the V1 weights (``best_map_model.pth``) and get ``LEGACY_MANIFEST``.

No PyTorch here: the app and the tests read manifests without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1

KIND_RCNN = "faster-rcnn-resnet50-fpn"

# How a cropped BGR frame becomes the model's input.
# "rgb": channels reversed to RGB, values 0-1. The torchvision model normalises
#   inside itself. This is what training uses.
# "rgb-normalized": RGB, ImageNet-normalised before the model, which then
#   normalises again. This is what best_map_model.pth was trained on (the January
#   2025 script normalised in its augmentation step as well).
# "legacy-v1": what the V1 app feeds best_map_model.pth: the same double
#   normalisation, but with the channels left in BGR.
INPUT_RGB = "rgb"
INPUT_RGB_NORMALIZED = "rgb-normalized"
INPUT_LEGACY_V1 = "legacy-v1"
INPUT_MODES = (INPUT_RGB, INPUT_RGB_NORMALIZED, INPUT_LEGACY_V1)

LEGACY_CLASSES = ("background", "space", "earth", "iss", "lf", "td", "sun", "nofeed", "panel")


class ManifestError(ValueError):
    """Raised when a manifest file cannot be used."""


@dataclass(frozen=True)
class ModelManifest:
    kind: str
    # Output order of the network. Index 0 is the background class for box models.
    classes: tuple[str, ...]
    input: str
    # What the labels are: "boxes" now; "masks" when a segmentation model arrives.
    task: str = "boxes"
    # Arguments the network was built with that change its behaviour (input size).
    model_args: dict[str, Any] = field(default_factory=dict)
    # Everything else: training record, metrics, dataset summary, versions.
    record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "kind": self.kind,
            "task": self.task,
            "classes": list(self.classes),
            "input": self.input,
            "model_args": self.model_args,
            **self.record,
        }


LEGACY_MANIFEST = ModelManifest(kind=KIND_RCNN, classes=LEGACY_CLASSES, input=INPUT_LEGACY_V1)

_OWN_KEYS = {"schema_version", "kind", "task", "classes", "input", "model_args"}


def manifest_path(weights: str | Path) -> Path:
    return Path(weights).with_suffix(".json")


def read_manifest(weights: str | Path) -> ModelManifest | None:
    """The manifest beside ``weights``, or None when there is none."""
    path = manifest_path(weights)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        manifest = ModelManifest(
            kind=str(data["kind"]),
            classes=tuple(str(name) for name in data["classes"]),
            input=str(data["input"]),
            task=str(data.get("task", "boxes")),
            model_args=dict(data.get("model_args", {})),
            record={k: v for k, v in data.items() if k not in _OWN_KEYS},
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManifestError(f"could not read model manifest {path}: {exc!r}") from exc
    if manifest.input not in INPUT_MODES:
        raise ManifestError(f"{path}: unknown input {manifest.input!r}")
    if not manifest.classes:
        raise ManifestError(f"{path}: no classes")
    return manifest


def write_manifest(weights: str | Path, manifest: ModelManifest) -> Path:
    path = manifest_path(weights)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest.to_dict(), fh, indent=2)
        fh.write("\n")
    return path
