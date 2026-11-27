"""Region models: anything that labels the scene regions of a cropped frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from uapd.config import RegionConfig
from uapd.pipeline import PipelineError, Region


class RegionModel(Protocol):
    def describe(self) -> dict[str, Any]:
        """Model identity: kind, weights file and hash, device."""
        ...

    def detect(self, frame: np.ndarray) -> list[Region]:
        """Regions of one cropped BGR frame, already filtered by score."""
        ...

    def close(self) -> None: ...


def load_region_model(weights: str | Path, config: RegionConfig) -> RegionModel:
    """The region model for a weights file, chosen by the kind in its manifest."""
    from uapd.regions.manifest import KIND_RCNN, LEGACY_MANIFEST, ManifestError, read_manifest

    try:
        manifest = read_manifest(weights) or LEGACY_MANIFEST
    except ManifestError as exc:
        raise PipelineError(str(exc)) from exc
    if manifest.kind == KIND_RCNN:
        from uapd.regions.rcnn import RcnnRegionModel

        return RcnnRegionModel(weights, config)
    raise PipelineError(f"{weights}: no region model for kind {manifest.kind!r}")
