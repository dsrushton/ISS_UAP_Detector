"""Run a pipeline over a source with no display and produce a JSON-serialisable report.

The report has two parts with different guarantees:

* ``summary`` and ``frames`` depend only on the footage and the pipeline, so two
  runs of a deterministic pipeline can be compared directly.
* ``run`` holds what varies between runs (timing, date, code commit).
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from uapd import __version__
from uapd.pipeline import FrameResult, Pipeline
from uapd.sources import FrameSource

REPORT_SCHEMA_VERSION = 1
FRAME_MODES = ("detections", "all")


def run_eval(
    pipeline: Pipeline,
    source: FrameSource,
    *,
    max_frames: int | None = None,
    frame_mode: str = "detections",
) -> dict[str, Any]:
    """Process every frame of ``source`` and return the report.

    ``frame_mode`` selects which frames are listed individually: only those with
    detections, or all of them. The summary always covers every processed frame.
    """
    if frame_mode not in FRAME_MODES:
        raise ValueError(f"frame_mode must be one of {FRAME_MODES}, got {frame_mode!r}")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    states: Counter[str] = Counter()
    region_frames: Counter[str] = Counter()
    frames_out: list[dict[str, Any]] = []
    durations: list[float] = []
    processed = 0
    detections_total = 0
    frames_with_detections = 0

    started = time.perf_counter()
    for index, frame in source.frames():
        if max_frames is not None and processed >= max_frames:
            break
        t0 = time.perf_counter()
        result = pipeline.process(frame, index)
        durations.append(time.perf_counter() - t0)

        processed += 1
        states[result.state] += 1
        for label in {region.label for region in result.regions}:
            region_frames[label] += 1
        if result.detections:
            frames_with_detections += 1
            detections_total += len(result.detections)
        if result.detections or frame_mode == "all":
            frames_out.append(_frame_record(result))
    elapsed = time.perf_counter() - started

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "pipeline": pipeline.describe(),
        "source": source.describe(),
        "frame_mode": frame_mode,
        "summary": {
            "frames": processed,
            "frames_with_detections": frames_with_detections,
            "detections": detections_total,
            "states": dict(sorted(states.items())),
            "frames_with_region": dict(sorted(region_frames.items())),
        },
        "frames": frames_out,
        "run": {
            "uapd_version": __version__,
            "commit": _git_commit(),
            "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 3),
            "frame_ms": _timing(durations),
        },
    }


def write_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
        fh.write("\n")


def dumps_report(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=_json_default)


def _frame_record(result: FrameResult) -> dict[str, Any]:
    record: dict[str, Any] = {
        "index": result.index,
        "state": result.state,
        "regions": [
            {"label": r.label, "box": [round(v, 1) for v in r.box], "score": round(r.score, 4)}
            for r in result.regions
        ],
        "detections": [{"box": list(d.box), "metrics": d.metrics} for d in result.detections],
    }
    if result.notes:
        record["notes"] = result.notes
    return record


def _timing(durations: list[float]) -> dict[str, float] | None:
    if not durations:
        return None
    ms = np.asarray(durations) * 1000.0
    return {
        "mean": round(float(ms.mean()), 3),
        "p50": round(float(np.percentile(ms, 50)), 3),
        "p95": round(float(np.percentile(ms, 95)), 3),
        "max": round(float(ms.max()), 3),
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
