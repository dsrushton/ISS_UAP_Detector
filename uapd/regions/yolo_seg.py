"""The YOLO segmentation region model (ultralytics).

Gives each region its outline as well as its box, so the search area follows the
real edge of space instead of a rectangle around it. Trained by
``uapd.train.yolo_seg``; the manifest beside the weights names the classes.

Scores from this model are not on the same scale as the Faster R-CNN's, so the
per-class thresholds of ``RegionConfig`` (tuned for that model) do not apply.
One confidence is used for every class: ``confidence`` in the manifest's
``model_args``, or ``DEFAULT_CONFIDENCE``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from uapd.config import RegionConfig
from uapd.labels.automask import MaskSettings, mask_to_polygons
from uapd.pipeline import PipelineError, Region
from uapd.regions.manifest import KIND_YOLO_SEG, ManifestError, read_manifest
from uapd.regions.rcnn import sha256_file

DEFAULT_CONFIDENCE = 0.4
DEFAULT_IMAGE_SIZE = 960
# How a predicted mask becomes outlines: small specks dropped, light smoothing.
OUTLINE = MaskSettings(smooth=3, min_piece=0.0005, simplify=1.0)


class YoloSegRegionModel:
    def __init__(self, weights: str | Path, config: RegionConfig, device: str | None = None):
        self.weights = Path(weights)
        self.config = config
        if not self.weights.is_file():
            raise PipelineError(f"model weights not found: {self.weights}")
        try:
            manifest = read_manifest(self.weights)
        except ManifestError as exc:
            raise PipelineError(str(exc)) from exc
        if manifest is None or manifest.kind != KIND_YOLO_SEG:
            raise PipelineError(f"{self.weights} has no {KIND_YOLO_SEG!r} manifest")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise PipelineError(
                f"this region model needs the ultralytics package: {exc} "
                "(install with: pip install -e '.[yolo]')"
            ) from exc
        self.manifest = manifest
        # The manifest lists background first, as the box models do; YOLO has no
        # background class, so its class 0 is the manifest's class 1.
        self.classes = manifest.classes
        self.image_size = int(manifest.model_args.get("image_size", DEFAULT_IMAGE_SIZE))
        self.confidence = float(manifest.model_args.get("confidence", DEFAULT_CONFIDENCE))
        self.device = device
        try:
            self._model = YOLO(str(self.weights), task="segment")
        except Exception as exc:
            raise PipelineError(f"could not load model weights from {self.weights}: {exc}") from exc
        self._sha256 = sha256_file(self.weights)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": KIND_YOLO_SEG,
            "weights": str(self.weights),
            "weights_sha256": self._sha256,
            "device": str(self.device or "auto"),
            "classes": list(self.classes),
            "input": self.manifest.input,
            "manifest": True,
            "confidence": self.confidence,
        }

    def detect(self, frame: np.ndarray) -> list[Region]:
        result = self._model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            max_det=self.config.max_boxes,
            device=self.device,
            retina_masks=True,  # masks at the frame's own size
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().tolist()
        labels = result.boxes.cls.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        masks = result.masks.data.cpu().numpy() > 0.5 if result.masks is not None else None
        height, width = frame.shape[:2]
        regions = []
        for index, (box, label, score) in enumerate(zip(boxes, labels, scores, strict=True)):
            name = self.classes[int(label) + 1]
            pieces = []
            if masks is not None and masks[index].shape == (height, width):
                # Each separate piece of the mask is a region of its own, and a hole
                # in a piece stays a hole (see uapd.labels.automask.mask_to_polygons).
                pieces, _ = mask_to_polygons(masks[index], (0, 0, width, height), OUTLINE)
            if not pieces:
                regions.append(Region(name, tuple(float(v) for v in box), float(score)))
                continue
            for points in pieces:
                xs, ys = [p[0] for p in points], [p[1] for p in points]
                regions.append(
                    Region(name, (min(xs), min(ys), max(xs), max(ys)), float(score), points)
                )
        regions.sort(key=lambda r: r.score, reverse=True)
        return regions

    def close(self) -> None:
        self._model = None
