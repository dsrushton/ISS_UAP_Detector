"""Review marks: what a person decided about a folder of labelled frames.

Labels made by a model (``uapd masks``) need a look before they are trained on.
The marks are kept in ``review.json`` beside the folder's labels::

    {"schema_version": 1, "verdict": "ok", "bad_frames": ["00012.jpg"], "note": ""}

``verdict`` is "ok", "reject" or null (not looked at yet). A rejected folder and
the frames listed in ``bad_frames`` are left out of training (``uapd.train.data``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REVIEW_FILE = "review.json"
SCHEMA_VERSION = 1
VERDICT_OK = "ok"
VERDICT_REJECT = "reject"
VERDICTS = (VERDICT_OK, VERDICT_REJECT)


@dataclass
class Review:
    verdict: str | None = None
    bad_frames: set[str] = field(default_factory=set)
    note: str = ""

    @property
    def rejected(self) -> bool:
        return self.verdict == VERDICT_REJECT


def read_review(folder: str | Path) -> Review:
    """The marks of ``folder``; an empty review when there are none or they cannot be read."""
    try:
        with open(Path(folder) / REVIEW_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        verdict = data.get("verdict")
        return Review(
            verdict=verdict if verdict in VERDICTS else None,
            bad_frames={str(name) for name in data.get("bad_frames", [])},
            note=str(data.get("note", "")),
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return Review()


def write_review(folder: str | Path, review: Review) -> Path:
    path = Path(folder) / REVIEW_FILE
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "verdict": review.verdict,
                "bad_frames": sorted(review.bad_frames),
                "note": review.note,
            },
            fh,
            indent=1,
        )
        fh.write("\n")
    return path
