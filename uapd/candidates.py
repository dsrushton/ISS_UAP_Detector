"""Candidate generator: bright objects enclosed by dark space.

Ported from V1's ``detect_anomalies`` / ``analyze_object`` with these differences:

* Everything is measured at full resolution on the cropped frame. V1 measured on
  a 0.6x image and compared against full-scale thresholds.
* ``area`` is the number of pixels the object covers. V1 used ``cv2.contourArea``,
  which is 16 for a 5x5 px object, and required 10 contour points.
* The background is a real ring around the object. V1 built its ring with the
  morphology kernel (size 1), which is empty, so its background was always 0.
* The border distance is measured to the actual edge of the space mask, not to
  the bounding rectangle of the space region.
* Nothing is dropped silently. Every measured candidate is returned with its full
  measurement vector and every gate it failed; an empty ``failed`` means it was
  accepted. Candidates too faint to pass the contrast gate whatever their
  background are counted (``CandidateSet.faint``) rather than measured, unless
  ``measure_faint`` is set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from uapd.config import CandidateConfig

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

# Region classes whose distance to each candidate is measured.
CONTEXT_LABELS = ("lf", "iss", "panel")

# Gate names, in the order they are checked.
GATE_MIN_DIMENSION = "min_dimension"
GATE_MAX_SIZE = "max_size"
GATE_MIN_AREA = "min_area"
GATE_ASPECT_RATIO = "aspect_ratio"
GATE_MIN_BRIGHTNESS = "min_brightness"
GATE_MAX_BRIGHTNESS = "max_brightness"
GATE_BG_BRIGHTNESS = "bg_brightness"
GATE_CONTRAST = "contrast"
GATE_BORDER = "border"
GATE_AVOID_BOX = "avoid_box"


def overlap_gate(label: str) -> str:
    return f"overlap_{label}"


@dataclass(frozen=True)
class SearchArea:
    """Where to look. Rebuilt only when the space regions change."""

    mask: np.ndarray  # uint8, 255 inside space
    # For each space pixel: how many space pixels lie between it and the edge of
    # the space mask (the frame edge counts as an edge).
    border_distance: np.ndarray
    bounds: tuple[int, int, int, int]  # x, y, w, h of the mask's bounding box


@dataclass(frozen=True)
class Candidate:
    box: tuple[int, int, int, int]  # x, y, w, h in the cropped frame
    metrics: dict[str, Any]
    failed: tuple[str, ...]  # gates failed, in check order; empty when accepted

    @property
    def accepted(self) -> bool:
        return not self.failed


@dataclass(frozen=True)
class CandidateSet:
    candidates: list[Candidate]
    found: int  # objects found, before any were set aside
    faint: int = 0  # certain contrast rejects that were counted, not measured

    @property
    def truncated(self) -> bool:
        """True when the max_candidates limit left some objects unmeasured."""
        return self.found > len(self.candidates) + self.faint


def build_search_area(shape: tuple[int, int], space_boxes: Sequence[Box]) -> SearchArea:
    """Combine the space boxes into one mask for a frame of ``shape`` (height, width)."""
    height, width = shape
    mask = np.zeros((height, width), np.uint8)
    for x1, y1, x2, y2 in space_boxes:
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

    # Pad with non-space so the frame edge counts as a border.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    border_distance = np.maximum(distance[1:-1, 1:-1] - 1.0, 0.0)
    return SearchArea(mask=mask, border_distance=border_distance, bounds=cv2.boundingRect(mask))


def find_candidates(
    frame: np.ndarray,
    area: SearchArea,
    config: CandidateConfig,
    region_boxes: Mapping[str, Sequence[Box]] | None = None,
    avoid_boxes: Sequence[Box] = (),
) -> CandidateSet:
    """Find and measure every bright object enclosed by dark space in ``frame``.

    ``frame`` is the cropped BGR (or single-channel) frame that ``area`` was built
    for. ``region_boxes`` maps region labels to boxes; distances to the
    ``CONTEXT_LABELS`` classes are measured for every candidate.
    """
    if frame.shape[:2] != area.mask.shape:
        raise ValueError(
            f"frame is {frame.shape[1]}x{frame.shape[0]} but the search area is "
            f"{area.mask.shape[1]}x{area.mask.shape[0]}"
        )
    rx, ry, rw, rh = area.bounds
    if rw == 0 or rh == 0:
        return CandidateSet(candidates=[], found=0)
    region_boxes = region_boxes or {}

    roi = frame[ry : ry + rh, rx : rx + rw]
    value = _max_channel(roi)
    space = area.mask[ry : ry + rh, rx : rx + rw]

    search = _search_mask(value, space, config)
    contours, _ = cv2.findContours(search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = len(contours)
    faint = 0
    if not config.measure_faint:
        # Contrast is object mean minus background, so it cannot exceed the brightest
        # pixel in the object's box. Below min_contrast the contrast gate must fail.
        kept = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if value[y : y + h, x : x + w].max() < config.min_contrast:
                faint += 1
            else:
                kept.append(contour)
        contours = kept
    if len(contours) > config.max_candidates:
        contours = sorted(contours, key=_bbox_area, reverse=True)[: config.max_candidates]

    border = area.border_distance[ry : ry + rh, rx : rx + rw]
    candidates = [
        _measure(contour, value, border, (rx, ry), config, region_boxes, avoid_boxes)
        for contour in contours
    ]
    # Stable order, independent of how OpenCV walked the mask.
    candidates.sort(key=lambda c: (c.box[1], c.box[0], c.box[3], c.box[2]))
    return CandidateSet(candidates=candidates, found=found, faint=faint)


def _max_channel(image: np.ndarray) -> np.ndarray:
    """Brightest of the colour channels per pixel (numpy's max over the axis is ~20x slower)."""
    if image.ndim == 2:
        return image
    channels = cv2.split(image)
    value = channels[0]
    for channel in channels[1:]:
        value = cv2.max(value, channel)
    return value


def _search_mask(value: np.ndarray, space: np.ndarray, config: CandidateConfig) -> np.ndarray:
    """Bright pixels that lie inside an outline of dark space (V1's hybrid mask)."""
    kernel = np.ones((config.morph_kernel_size, config.morph_kernel_size), np.uint8)

    _, dark = cv2.threshold(value, config.dark_region_threshold, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.bitwise_and(dark, space)

    # Filling the outer outlines of the dark areas turns anything they enclose into
    # search area. Bright areas that reach the edge of the dark are not enclosed.
    outlines, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    enclosed = np.zeros_like(space)
    cv2.drawContours(enclosed, outlines, -1, 255, -1)

    _, bright = cv2.threshold(value, config.min_object_brightness, 255, cv2.THRESH_BINARY)
    bright = cv2.dilate(bright, kernel)
    return cv2.bitwise_and(bright, enclosed)


def _bbox_area(contour: np.ndarray) -> int:
    _, _, w, h = cv2.boundingRect(contour)
    return w * h


def _measure(
    contour: np.ndarray,
    value: np.ndarray,
    border: np.ndarray,
    offset: tuple[int, int],
    config: CandidateConfig,
    region_boxes: Mapping[str, Sequence[Box]],
    avoid_boxes: Sequence[Box],
) -> Candidate:
    x, y, w, h = cv2.boundingRect(contour)

    # Work in a small window around the object: its box plus the background ring.
    pad = max(config.background_ring, 0)
    wx1, wy1 = max(x - pad, 0), max(y - pad, 0)
    wx2, wy2 = min(x + w + pad, value.shape[1]), min(y + h + pad, value.shape[0])
    window = value[wy1:wy2, wx1:wx2]

    obj = np.zeros(window.shape, np.uint8)
    cv2.drawContours(obj, [contour], -1, 255, -1, offset=(-wx1, -wy1))
    ring_kernel = np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)
    ring = cv2.bitwise_xor(cv2.dilate(obj, ring_kernel), obj)

    area = int(cv2.countNonZero(obj))
    obj_brightness = float(cv2.mean(window, mask=obj)[0])
    peak_brightness = int(window[obj > 0].max())
    if cv2.countNonZero(ring):
        bg_brightness = float(cv2.mean(window, mask=ring)[0])
        contrast = obj_brightness - bg_brightness
    else:
        bg_brightness = None
        contrast = None
    border_distance = float(border[y : y + h, x : x + w].min())

    gx, gy = x + offset[0], y + offset[1]
    box_xyxy = (gx, gy, gx + w, gy + h)
    aspect_ratio = w / h

    metrics: dict[str, Any] = {
        "width": w,
        "height": h,
        "area": area,
        "bbox_area": w * h,
        "aspect_ratio": aspect_ratio,
        "obj_brightness": obj_brightness,
        "peak_brightness": peak_brightness,
        "bg_brightness": bg_brightness,
        "contrast": contrast,
        "border_distance": border_distance,
    }

    failed: list[str] = []
    if w < config.min_dimension or h < config.min_dimension:
        failed.append(GATE_MIN_DIMENSION)
    if w > config.max_width or h > config.max_height:
        failed.append(GATE_MAX_SIZE)
    if area < config.min_area:
        failed.append(GATE_MIN_AREA)
    if aspect_ratio > config.max_aspect_ratio or aspect_ratio < 1 / config.max_aspect_ratio:
        failed.append(GATE_ASPECT_RATIO)
    if obj_brightness <= config.min_object_brightness:
        failed.append(GATE_MIN_BRIGHTNESS)
    if obj_brightness >= config.max_object_brightness:
        failed.append(GATE_MAX_BRIGHTNESS)
    if bg_brightness is not None and bg_brightness > config.max_bg_brightness:
        failed.append(GATE_BG_BRIGHTNESS)
    if contrast is not None and contrast < config.min_contrast:
        failed.append(GATE_CONTRAST)
    if border_distance < config.border_margin:
        failed.append(GATE_BORDER)

    filtered = {"lf": config.filter_lf, "iss": config.filter_iss, "panel": config.filter_panel}
    for label in CONTEXT_LABELS:
        distance, overlaps = _nearest(box_xyxy, region_boxes.get(label, ()))
        metrics[f"distance_{label}"] = distance
        if filtered[label] and overlaps:
            failed.append(overlap_gate(label))
    if _nearest(box_xyxy, avoid_boxes)[1]:
        failed.append(GATE_AVOID_BOX)

    return Candidate(box=(gx, gy, w, h), metrics=metrics, failed=tuple(failed))


def _nearest(box: Box, others: Sequence[Box]) -> tuple[float | None, bool]:
    """Gap in pixels to the nearest of ``others`` (None when empty), and whether any overlaps.

    Boxes that only share an edge have a gap of 0 but do not overlap.
    """
    best: float | None = None
    overlaps = False
    for other in others:
        gap = math.hypot(
            max(other[0] - box[2], box[0] - other[2], 0.0),
            max(other[1] - box[3], box[1] - other[3], 0.0),
        )
        if best is None or gap < best:
            best = gap
        if box[0] < other[2] and box[2] > other[0] and box[1] < other[3] and box[3] > other[1]:
            overlaps = True
    return best, overlaps
