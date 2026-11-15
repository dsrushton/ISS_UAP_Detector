"""Typed, immutable configuration for the V2 pipeline.

A Config is built once (from defaults, optionally overridden by a TOML profile)
and passed explicitly to whatever needs it. Nothing reads configuration at import
time, so a changed profile always reaches every component.

All pixel values are in full-resolution pixels of the cropped frame.
Defaults are the values V1 runs with on ``dev`` (legacy/SOD_Constants.py).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a profile contains an unknown key or a value of the wrong type."""


@dataclass(frozen=True)
class SourceConfig:
    # The ISS live feed moves to a new video id from time to time; override in a profile.
    live_url: str = "https://www.youtube.com/watch?v=M3HKLzjvKPc"


@dataclass(frozen=True)
class FrameConfig:
    # The ISS feed is 1280 wide with side bars; V1 crops it to 939 px.
    crop_left: int = 165
    crop_right: int = 176


@dataclass(frozen=True)
class RegionConfig:
    cycle_frames: int = 120
    max_boxes: int = 10
    darkness_area_threshold: float = 0.25
    max_lens_flares: int = 3
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "space": 0.80,
            "earth": 0.75,
            "iss": 0.75,
            "lf": 0.99,
            "td": 0.25,
            "sun": 0.75,
            "nofeed": 0.25,
            "panel": 0.75,
        }
    )


@dataclass(frozen=True)
class CandidateConfig:
    min_dimension: int = 5
    max_width: int = 100
    max_height: int = 100
    min_area: int = 25
    max_aspect_ratio: float = 20.0
    min_object_brightness: int = 10
    max_object_brightness: int = 200
    max_bg_brightness: int = 23
    min_contrast: int = 23
    dark_region_threshold: int = 10
    morph_kernel_size: int = 1
    border_margin: int = 3
    filter_iss: bool = False
    filter_panel: bool = True
    filter_lf: bool = True


@dataclass(frozen=True)
class CaptureConfig:
    buffer_seconds: float = 3.0
    post_detection_seconds: float = 2.1
    save_interval_seconds: float = 2.0


@dataclass(frozen=True)
class Config:
    source: SourceConfig = field(default_factory=SourceConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build a Config from a nested dict; missing keys keep their defaults."""
        return _build(cls, data, path="")

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load a TOML profile."""
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Short stable id of the values, recorded with everything this config produced."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'profile'}: expected a table, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        where = f" in [{path}]" if path else ""
        raise ConfigError(f"unknown key(s){where}: {', '.join(unknown)}")

    defaults = cls()
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        key = f"{path}.{name}" if path else name
        default = getattr(defaults, name)
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value, key)
        elif isinstance(default, dict):
            kwargs[name] = _merge_float_table(default, value, key)
        else:
            kwargs[name] = _coerce(value, type(default), key)
    return cls(**kwargs)


def _merge_float_table(default: dict[str, float], value: Any, key: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ConfigError(f"{key}: expected a table, got {type(value).__name__}")
    unknown = sorted(set(value) - set(default))
    if unknown:
        raise ConfigError(f"unknown key(s) in [{key}]: {', '.join(unknown)}")
    merged = dict(default)
    for name, item in value.items():
        merged[name] = _coerce(item, float, f"{key}.{name}")
    return merged


def _coerce(value: Any, expected: type, key: str) -> Any:
    # bool is a subclass of int, so it has to be checked first in both directions.
    if expected is bool:
        if isinstance(value, bool):
            return value
    elif isinstance(value, bool):
        pass
    elif expected is float and isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, expected):
        return value
    raise ConfigError(f"{key}: expected {expected.__name__}, got {value!r}")
