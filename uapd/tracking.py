"""Temporal layer: link per-frame detections into tracks and gate them on persistence.

A detection starts a pending track. Detections on later frames that lie near a
track continue it. Once a track has been seen on ``confirm_frames`` frames it is
confirmed; until then its detections are only pending. A track that ends
unconfirmed is a transient: a brief flash such as a cosmic ray hit on the sensor.

Frames are counted as they arrive. The live feed sends each camera frame twice,
but the copies are not identical (a line along the top edge of the picture
changes on every frame), so the tracker does not try to tell them apart; the
default ``confirm_frames`` allows for the doubling instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from uapd.config import TrackingConfig
from uapd.pipeline import Detection, TrackSummary


@dataclass
class _Track:
    id: int
    first_index: int
    first_box: tuple[int, int, int, int]
    last_index: int
    last_box: tuple[int, int, int, int]
    frames_seen: int = 1
    missed: int = 0
    confirmed: bool = False
    max_contrast: float | None = None

    def summary(self) -> TrackSummary:
        return TrackSummary(
            id=self.id,
            confirmed=self.confirmed,
            first_index=self.first_index,
            last_index=self.last_index,
            frames_seen=self.frames_seen,
            first_box=self.first_box,
            last_box=self.last_box,
            max_contrast=self.max_contrast,
        )


@dataclass
class TrackUpdate:
    detections: list[Detection] = field(default_factory=list)  # on confirmed tracks
    pending: list[Detection] = field(default_factory=list)  # on tracks not yet confirmed
    confirmed: list[int] = field(default_factory=list)  # track ids confirmed on this frame
    ended: list[TrackSummary] = field(default_factory=list)


class Tracker:
    def __init__(self, config: TrackingConfig):
        if config.confirm_frames < 1:
            raise ValueError("tracking.confirm_frames must be at least 1")
        self.config = config
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: Sequence[Detection], index: int) -> TrackUpdate:
        """Assign this frame's accepted candidates to tracks."""
        update = TrackUpdate()

        # Closest pairs first; each track and each detection is used once.
        pairs = sorted(
            (gap, t, d)
            for t, track in enumerate(self._tracks)
            for d, detection in enumerate(detections)
            if (gap := _gap(track.last_box, detection.box)) <= self.config.match_distance
        )
        track_of: dict[int, _Track] = {}
        matched: set[int] = set()
        for _, t, d in pairs:
            if t in matched or d in track_of:
                continue
            matched.add(t)
            track = self._tracks[t]
            track_of[d] = track
            track.last_index = index
            track.last_box = detections[d].box
            track.frames_seen += 1
            track.missed = 0

        for d, detection in enumerate(detections):
            track = track_of.get(d)
            if track is None:
                track = _Track(
                    id=self._next_id,
                    first_index=index,
                    first_box=detection.box,
                    last_index=index,
                    last_box=detection.box,
                )
                self._next_id += 1
                self._tracks.append(track)
                track_of[d] = track
            contrast = detection.metrics.get("contrast")
            if contrast is not None and (
                track.max_contrast is None or contrast > track.max_contrast
            ):
                track.max_contrast = contrast
            if not track.confirmed and track.frames_seen >= self.config.confirm_frames:
                track.confirmed = True
                update.confirmed.append(track.id)
            tracked = Detection(box=detection.box, metrics=detection.metrics, track=track.id)
            (update.detections if track.confirmed else update.pending).append(tracked)

        seen = {track.id for track in track_of.values()}
        alive = []
        for track in self._tracks:
            if track.id not in seen:
                track.missed += 1
                if track.missed > self.config.max_missed_frames:
                    update.ended.append(track.summary())
                    continue
            alive.append(track)
        self._tracks = alive
        return update

    def finish(self) -> list[TrackSummary]:
        """End every open track, for the end of a clip."""
        ended = [track.summary() for track in self._tracks]
        self._tracks = []
        return ended


def _gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Distance in pixels between two x, y, w, h boxes; 0 when they touch or overlap."""
    gap_x = max(b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]), 0)
    gap_y = max(b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]), 0)
    return math.hypot(gap_x, gap_y)
