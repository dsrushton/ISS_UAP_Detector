"""Event data files: what the detector saw around each recording.

An event is one recording. While the app runs, ``EventLog`` keeps the last few
seconds of frame records; when a recording starts they become the start of the
event, and when it stops the event is written as JSON beside the clip.

Only frames on which something happened are kept: a detection, a pending
detection, a rejected candidate, or a track ending.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

from uapd.eval.headless import _json_default
from uapd.pipeline import Detection, FrameResult, Region, region_record

EVENT_SCHEMA_VERSION = 1


class EventLog:
    def __init__(self, directory: str | Path, header: dict[str, Any], buffer_frames: int):
        """``header`` is stored in every event: versions, config, source."""
        self.directory = Path(directory)
        self.header = header
        self.buffer_frames = max(int(buffer_frames), 0)
        self._recent: deque[tuple[int, dict[str, Any]]] = deque()
        self._regions: list[Region] | None = None
        self._count = 0
        self._event: dict[str, Any] | None = None
        self.written: list[Path] = []

    def observe(self, result: FrameResult, recording: bool, event_id: str | None) -> None:
        """Record one processed frame. ``event_id`` names the recording in progress."""
        count = self._count
        self._count += 1

        if self._event is not None and (not recording or event_id != self._event["event_id"]):
            self._finish()

        regions_changed = result.regions != self._regions
        self._regions = list(result.regions)
        if recording and self._event is None and event_id is not None:
            self._start(event_id, count)  # stores the current regions
        elif self._event is not None and regions_changed:
            self._event["regions"].append(_regions_record(result.regions, count))

        record = _frame_record(result, count)
        if self._event is not None:
            if record is not None:
                self._event["frames"].append(record)
            self._event["last_frame"] = count
        elif record is not None:
            self._recent.append((count, record))
        while self._recent and self._recent[0][0] < count - self.buffer_frames:
            self._recent.popleft()

    def close(self) -> None:
        """Write the event in progress, if any."""
        if self._event is not None:
            self._finish()

    def _start(self, event_id: str, count: int) -> None:
        self._event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": event_id,
            "started": _now(),
            **self.header,
            "coordinates": "cropped frame",
            # Frame numbers count frames processed since the app started.
            "first_frame": self._recent[0][0] if self._recent else count,
            "recording_started_frame": count,
            "last_frame": count,
            "regions": [_regions_record(self._regions or [], count)],
            "frames": [record for _, record in self._recent],
        }
        self._recent.clear()

    def _finish(self) -> None:
        event, self._event = self._event, None
        event["ended"] = _now()
        event["summary"] = _summary(event["frames"])
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{event['event_id']}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(event, fh, indent=1, default=_json_default)
            fh.write("\n")
        self.written.append(path)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")


def _detection(detection: Detection) -> dict[str, Any]:
    return {"track": detection.track, "box": list(detection.box), "metrics": detection.metrics}


def _regions_record(regions: list[Region], count: int) -> dict[str, Any]:
    return {
        "from_frame": count,
        "regions": [region_record(r) for r in regions],
    }


def _frame_record(result: FrameResult, count: int) -> dict[str, Any] | None:
    if not (result.detections or result.pending or result.rejected or result.ended_tracks):
        return None
    record: dict[str, Any] = {"frame": count, "state": result.state}
    if result.detections:
        record["detections"] = [_detection(d) for d in result.detections]
    if result.pending:
        record["pending"] = [_detection(d) for d in result.pending]
    if result.rejected:
        record["rejected"] = [
            {"box": list(r.box), "gates": list(r.gates), "metrics": r.metrics}
            for r in result.rejected
        ]
    if result.ended_tracks:
        record["ended_tracks"] = [dataclasses.asdict(t) for t in result.ended_tracks]
    if result.faint:
        record["faint"] = result.faint
    return record


def _summary(frames: list[dict[str, Any]]) -> dict[str, Any]:
    tracks = {d["track"] for f in frames for d in f.get("detections", [])}
    gates = Counter(r["gates"][0] for f in frames for r in f.get("rejected", []) if r["gates"])
    ended = [t for f in frames for t in f.get("ended_tracks", [])]
    return {
        "frames_with_detections": sum(1 for f in frames if f.get("detections")),
        "detections": sum(len(f.get("detections", [])) for f in frames),
        "tracks": sorted(t for t in tracks if t is not None),
        "transients": sum(1 for t in ended if not t["confirmed"]),
        "rejected": sum(len(f.get("rejected", [])) for f in frames),
        "rejected_by_gate": dict(sorted(gates.items())),
    }
