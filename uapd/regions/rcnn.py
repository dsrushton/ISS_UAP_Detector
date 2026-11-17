"""The V1 Faster R-CNN region model (``best_map_model.pth``).

Same network, weights and input preparation as V1, so the regions are the ones
V1 sees. PyTorch is imported when the model is built, not when this module is.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from uapd.config import RegionConfig
from uapd.pipeline import PipelineError, Region

CLASS_NAMES = ("background", "space", "earth", "iss", "lf", "td", "sun", "nofeed", "panel")
DEFAULT_THRESHOLD = 0.5


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

        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = self._load_model()
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        self._sha256 = _sha256(self.weights)

    def _load_model(self) -> Any:
        torch = self._torch
        model = build_model()
        try:
            state = torch.load(self.weights, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
        except Exception as exc:
            raise PipelineError(f"could not load model weights from {self.weights}: {exc}") from exc
        return model.to(self.device).eval()

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "faster-rcnn-resnet50-fpn",
            "weights": str(self.weights),
            "weights_sha256": self._sha256,
            "device": str(self.device),
            "classes": list(CLASS_NAMES),
        }

    def detect(self, frame: np.ndarray) -> list[Region]:
        torch = self._torch
        # V1's input preparation, kept exactly: channels stay in BGR order and the
        # image is normalised here as well as inside the torchvision model. The
        # weights are only known to work with this input.
        image = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
        image = image.permute(2, 0, 1).float() / 255.0
        image = (image - self._mean) / self._std
        with torch.no_grad():
            prediction = self._model([image])[0]
        return select_regions(
            prediction["boxes"].cpu().tolist(),
            prediction["labels"].cpu().tolist(),
            prediction["scores"].cpu().tolist(),
            self.config,
        )

    def close(self) -> None:
        self._model = None


def build_model() -> Any:
    """The V1 network, without downloading pretrained backbone weights."""
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.ops.misc import FrozenBatchNorm2d

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, norm_layer=FrozenBatchNorm2d, trainable_layers=3
    )
    return FasterRCNN(backbone, num_classes=len(CLASS_NAMES))


def select_regions(
    boxes: list[list[float]], labels: list[int], scores: list[float], config: RegionConfig
) -> list[Region]:
    """Keep predictions above their class threshold, best first, at most ``max_boxes``."""
    regions = []
    for box, label, score in zip(boxes, labels, scores, strict=True):
        name = CLASS_NAMES[label]
        if score > config.thresholds.get(name, DEFAULT_THRESHOLD):
            regions.append(Region(label=name, box=tuple(float(v) for v in box), score=float(score)))
    regions.sort(key=lambda r: r.score, reverse=True)
    return regions[: config.max_boxes]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
