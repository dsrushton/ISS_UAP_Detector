"""The Faster R-CNN region model.

Runs the V1 weights (``best_map_model.pth``) with V1's input preparation, so the
regions are the ones V1 sees, and runs weights trained by ``uapd train`` the way
their manifest says they were trained (see ``uapd.regions.manifest``).
PyTorch is imported when the model is built, not when this module is.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from uapd.config import RegionConfig
from uapd.pipeline import PipelineError, Region
from uapd.regions.manifest import (
    INPUT_LEGACY_V1,
    INPUT_MODES,
    INPUT_RGB,
    KIND_RCNN,
    LEGACY_CLASSES,
    LEGACY_MANIFEST,
    ManifestError,
    read_manifest,
)

CLASS_NAMES = LEGACY_CLASSES
DEFAULT_THRESHOLD = 0.5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RcnnRegionModel:
    def __init__(self, weights: str | Path, config: RegionConfig, device: str | None = None):
        self.weights = Path(weights)
        self.config = config
        if not self.weights.is_file():
            raise PipelineError(
                f"model weights not found: {self.weights} "
                "(download link is in README.md; pass the path with --weights)"
            )
        try:
            import torch
        except ImportError as exc:
            raise PipelineError(
                f"the region model needs PyTorch and torchvision: {exc} "
                "(install with: pip install -e '.[torch]')"
            ) from exc

        try:
            manifest = read_manifest(self.weights)
        except ManifestError as exc:
            raise PipelineError(str(exc)) from exc
        self.has_manifest = manifest is not None
        self.manifest = manifest or LEGACY_MANIFEST
        if self.manifest.kind != KIND_RCNN:
            raise PipelineError(
                f"{self.weights} is a {self.manifest.kind!r} model, not {KIND_RCNN!r}"
            )
        self.classes = self.manifest.classes
        self.input = self.manifest.input if config.input == "auto" else config.input
        if self.input not in INPUT_MODES:
            raise PipelineError(
                f"regions.input must be auto, {' or '.join(INPUT_MODES)}; got {config.input!r}"
            )

        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = self._load_model()
        self._sha256 = sha256_file(self.weights)

    def _load_model(self) -> Any:
        torch = self._torch
        model = build_model(len(self.classes), **self.manifest.model_args)
        try:
            state = torch.load(self.weights, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
        except Exception as exc:
            raise PipelineError(f"could not load model weights from {self.weights}: {exc}") from exc
        return model.to(self.device).eval()

    def describe(self) -> dict[str, Any]:
        return {
            "kind": KIND_RCNN,
            "weights": str(self.weights),
            "weights_sha256": self._sha256,
            "device": str(self.device),
            "classes": list(self.classes),
            "input": self.input,
            "manifest": self.has_manifest,
        }

    def detect(self, frame: np.ndarray) -> list[Region]:
        torch = self._torch
        image = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
        image = model_input(image.permute(2, 0, 1).float() / 255.0, self.input)
        with torch.no_grad():
            prediction = self._model([image])[0]
        return select_regions(
            prediction["boxes"].cpu().tolist(),
            prediction["labels"].cpu().tolist(),
            prediction["scores"].cpu().tolist(),
            self.config,
            self.classes,
        )

    def close(self) -> None:
        self._model = None


def model_input(image: Any, mode: str) -> Any:
    """Turn a BGR, 0-1, channels-first tensor into what the network takes.

    Training and detection both go through here, so they cannot disagree about
    channel order or normalisation.
    """
    if mode == INPUT_RGB:
        return image.flip(0)
    if mode == INPUT_LEGACY_V1:
        # V1's preparation, kept exactly: channels stay in BGR order and the image
        # is normalised here as well as inside the torchvision model.
        mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
        return (image - mean) / std
    raise ValueError(f"unknown model input {mode!r}")


def build_model(
    num_classes: int = len(LEGACY_CLASSES), trainable_layers: int = 3, **model_args: Any
) -> Any:
    """The V1 network, without downloading pretrained backbone weights.

    ``model_args`` go to torchvision's ``FasterRCNN`` (``min_size``, ``max_size``).
    """
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.ops.misc import FrozenBatchNorm2d

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        norm_layer=FrozenBatchNorm2d,
        trainable_layers=trainable_layers,
    )
    return FasterRCNN(backbone, num_classes=num_classes, **model_args)


def select_regions(
    boxes: list[list[float]],
    labels: list[int],
    scores: list[float],
    config: RegionConfig,
    classes: tuple[str, ...] = LEGACY_CLASSES,
) -> list[Region]:
    """Keep predictions above their class threshold, best first, at most ``max_boxes``."""
    regions = []
    for box, label, score in zip(boxes, labels, scores, strict=True):
        name = classes[label]
        if score > config.thresholds.get(name, DEFAULT_THRESHOLD):
            regions.append(Region(label=name, box=tuple(float(v) for v in box), score=float(score)))
    regions.sort(key=lambda r: r.score, reverse=True)
    return regions[: config.max_boxes]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
