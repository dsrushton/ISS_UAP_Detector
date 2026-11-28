"""Training settings: typed, immutable, loaded from an optional TOML file."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from uapd.config import ConfigError, _coerce
from uapd.regions.manifest import LEGACY_CLASSES

# Label names in the August 2024 CSV label set, mapped to the names the pipeline
# uses. Matching ignores case. The January 2025 XML labels need no mapping.
NOTEBOOK_ALIASES = {
    "Earth_broad": "earth",
    "panels": "panel",
    "no_feed": "nofeed",
    "total_darkness": "td",
    "t_d": "td",
}


@dataclass(frozen=True)
class TrainConfig:
    # Which trainer runs; see uapd.train.TRAINERS.
    kind: str = "rcnn-boxes"
    # Classes the model learns, in output order after the background class. The
    # pipeline looks regions up by these names, so keep them unless it changes too.
    classes: tuple[str, ...] = LEGACY_CLASSES[1:]
    aliases: dict[str, str] = field(default_factory=lambda: dict(NOTEBOOK_ALIASES))
    # Labels that are neither a class nor an alias: "drop" leaves those boxes out
    # and counts them; "error" refuses to train.
    unknown_labels: str = "drop"

    # Validation: the last part of each folder's frames, in file-name order.
    # Neighbouring video frames are near copies, so a random split would put the
    # same picture on both sides and make every score look better than it is.
    val_fraction: float = 0.2
    # Folders listed here are held out whole instead; the others are all training.
    # A bare name ("frames_5") matches every folder of that name; a path matches
    # that one folder.
    val_folders: tuple[str, ...] = ()
    # Stop when the validation score has not risen by more than min_improvement
    # for this many epochs in a row (0 = always run every epoch). best.pth is the
    # best epoch either way.
    patience: int = 0
    min_improvement: float = 0.001

    # What training starts from: "coco" (torchvision's pretrained detector,
    # downloaded once), "none", or the path of a weights file to continue from.
    init: str = "coco"
    # yolo-seg only: the pretrained model training starts from (init = "coco"),
    # and the longer side frames are resized to.
    yolo_model: str = "yolo11s-seg.pt"
    image_size: int = 960
    trainable_backbone_layers: int = 3
    min_size: int = 800
    max_size: int = 1333

    # Defaults below are the recipe that produced best_map_model.pth (January 2025):
    # Adam at 0.0001, batch 8, mixed precision, gradients clipped at 1.0, random
    # horizontal and vertical flips. Its folder name says 12 epochs.
    epochs: int = 12
    batch_size: int = 8
    optimizer: str = "adam"  # or "sgd"
    learning_rate: float = 0.0001
    momentum: float = 0.9
    weight_decay: float = 0.0005
    # Steps over which the learning rate rises to its full value.
    warmup_steps: int = 200
    # The learning rate drops to a tenth after this share of the run (1.0 = never).
    lr_drop_at: float = 0.8
    # Largest gradient norm allowed per step (0 = no limit).
    grad_clip: float = 1.0
    # Random flips of training frames. Validation frames are never flipped.
    hflip: bool = True
    vflip: bool = True
    # Random changes of hue, saturation and brightness (yolo-seg only). Off, as in
    # the January 2025 recipe: colour is a main cue between Earth, space and station.
    color_jitter: bool = False
    # Also keep the weights of every epoch (epoch-01, epoch-02, ...) beside best
    # and last, so any of them can be tried in the app. Off: a run folder then
    # holds two weights files, not dozens.
    keep_epochs: bool = False
    amp: bool = True
    workers: int = 0
    seed: int = 0

    # A prediction matches a labelled box of its class at this overlap or more.
    iou_threshold: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - fields)
        if unknown:
            raise ConfigError(f"unknown training setting(s): {', '.join(unknown)}")
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            default = getattr(defaults, name)
            if isinstance(default, tuple):
                if not isinstance(value, (list, tuple)) or not all(
                    isinstance(v, str) for v in value
                ):
                    raise ConfigError(f"{name}: expected a list of names, got {value!r}")
                kwargs[name] = tuple(value)
            elif isinstance(default, dict):
                if not isinstance(value, dict) or not all(
                    isinstance(v, str) for v in value.values()
                ):
                    raise ConfigError(f"{name}: expected a table of names, got {value!r}")
                kwargs[name] = dict(value)
            else:
                kwargs[name] = _coerce(value, type(default), name)
        return cls(**kwargs).validated()

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    def replace(self, **changes: Any) -> TrainConfig:
        return dataclasses.replace(self, **changes).validated()

    def validated(self) -> TrainConfig:
        def check(ok: bool, message: str) -> None:
            if not ok:
                raise ConfigError(message)

        check(bool(self.classes), "classes: at least one class is needed")
        check(len(set(self.classes)) == len(self.classes), "classes: names must be unique")
        check("background" not in self.classes, "classes: background is added automatically")
        check(self.unknown_labels in ("drop", "error"), "unknown_labels: drop or error")
        check(0.0 <= self.val_fraction < 1.0, "val_fraction: must be 0 or more and below 1")
        check(self.optimizer in ("sgd", "adam"), "optimizer: sgd or adam")
        check(self.epochs >= 1, "epochs: at least 1")
        check(self.patience >= 0, "patience: 0 or more")
        check(self.image_size >= 32, "image_size: at least 32")
        check(self.min_improvement >= 0, "min_improvement: 0 or more")
        check(self.batch_size >= 1, "batch_size: at least 1")
        check(self.learning_rate > 0, "learning_rate: must be above 0")
        check(0.0 < self.lr_drop_at <= 1.0, "lr_drop_at: above 0, at most 1")
        check(0 <= self.trainable_backbone_layers <= 5, "trainable_backbone_layers: 0 to 5")
        return self

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]
