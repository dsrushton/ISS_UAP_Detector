"""Saved captures and the verdict a person gives each one. No Tk here.

A capture is a clip in ``Detections/AVI`` with the detector's data file beside
it (``00083.avi``, ``00083.json``). A verdict says what the event was and is
written beside them as ``00083.verdict.json``::

    {"schema_version": 1, "event_id": "00083", "verdict": "flare", "note": "",
     "decided": "2026-11-27T20:15:02+00:00", "source": "human"}

Verdicts are the stockpile of prosaic examples for the learned filter
(docs/V2_PLAN.md, sections 5 and 10). They never train the region model.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
VERDICT_SUFFIX = ".verdict.json"
CLIP_SUFFIXES = (".avi", ".mp4")

# The starting taxonomy of docs/V2_PLAN.md, section 5. The order is the order of
# the buttons and of the keys 1 2 3 ... 9 0 -.
VERDICTS = (
    "unknown-interesting",
    "flare",
    "ice",
    "panel-edge",
    "iss-structure",
    "debris",
    "earth-limb",
    "star-or-planet",
    "compression-artifact",
    "hot-pixel",
    "other-prosaic",
)
VERDICT_KEYS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "minus")


@dataclass
class Capture:
    clip: Path
    data_file: Path | None
    verdict: str | None = None
    note: str = ""

    @property
    def event_id(self) -> str:
        return self.clip.stem

    @property
    def verdict_file(self) -> Path:
        return self.clip.with_name(self.clip.stem + VERDICT_SUFFIX)

    def data(self) -> dict[str, Any]:
        """The detector's data file, or {} when there is none or it cannot be read."""
        if self.data_file is None:
            return {}
        try:
            with open(self.data_file, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def describe(self) -> str:
        """One line about the event from its data file."""
        data = self.data()
        if not data:
            return "no data file"
        summary = data.get("summary", {})
        parts = [str(data.get("started", ""))[:19].replace("T", " ")]
        if "detections" in summary:
            parts.append(f"{summary['detections']} detections")
        if summary.get("tracks"):
            parts.append(f"{len(summary['tracks'])} track(s)")
        if summary.get("transients"):
            parts.append(f"{summary['transients']} transient(s)")
        return ", ".join(p for p in parts if p)


def list_captures(directory: str | Path) -> list[Capture]:
    """Clips in ``directory`` in name order, with their data file and verdict."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    captures = []
    for clip in sorted(p for p in directory.iterdir() if p.suffix.lower() in CLIP_SUFFIXES):
        data_file = clip.with_suffix(".json")
        capture = Capture(clip, data_file if data_file.is_file() else None)
        capture.verdict, capture.note = read_verdict(capture)
        captures.append(capture)
    return captures


def read_verdict(capture: Capture) -> tuple[str | None, str]:
    try:
        with open(capture.verdict_file, encoding="utf-8") as fh:
            data = json.load(fh)
        verdict = data.get("verdict")
        return (verdict if isinstance(verdict, str) and verdict else None), str(
            data.get("note", "")
        )
    except (OSError, ValueError, AttributeError):
        return None, ""


def write_verdict(capture: Capture, verdict: str | None, note: str = "") -> None:
    """Record ``verdict`` for the capture; None removes the verdict file."""
    if verdict is None:
        capture.verdict_file.unlink(missing_ok=True)
        capture.verdict, capture.note = None, ""
        return
    with open(capture.verdict_file, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "event_id": capture.event_id,
                "verdict": verdict,
                "note": note,
                "decided": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "source": "human",
            },
            fh,
            indent=1,
        )
        fh.write("\n")
    capture.verdict, capture.note = verdict, note
