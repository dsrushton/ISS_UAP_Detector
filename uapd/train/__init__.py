"""Training: label folders in, a run folder with weights and manifests out.

    uapd/train/config.py   TrainConfig
    uapd/train/data.py     folders -> class ids, split, report (no PyTorch)
    uapd/train/metrics.py  average precision (no PyTorch)
    uapd/train/rcnn.py     the Faster R-CNN box trainer
    uapd/train/__init__.py this file: the run folder, progress reporting, registry

A trainer is a function that takes a ``TrainJob`` and returns a result dict. To
add a model type, write one and register it in ``TRAINERS``; the run folder,
progress reporting, CLI and app need no change. Its weights need a manifest
``kind`` that ``uapd.regions.load_region_model`` knows how to load.

A run folder holds::

    run.json    settings, data report, versions, score after every epoch
    last.pth    weights after the latest epoch, with last.json (the manifest)
    best.pth    weights of the best-scoring epoch, with best.json

A trained model is a challenger. Nothing here makes it the model the detector
uses; that takes the evaluation gate (docs/V2_PLAN.md, section 6).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from uapd import __version__
from uapd.train.config import TrainConfig
from uapd.train.data import DataError, DataPlan, format_report, plan_data

RUN_SCHEMA_VERSION = 1
# Lines with this prefix on stdout are progress events for the app, as JSON.
EVENT_PREFIX = "@uapd "


class TrainError(RuntimeError):
    """Raised when a training run cannot start or continue."""


class Reporter:
    """Progress output: text lines for people, and JSON events when ``machine`` is on."""

    def __init__(self, stream: TextIO | None = None, machine: bool = False):
        self.stream = stream or sys.stdout
        self.machine = machine

    def log(self, text: str) -> None:
        print(text, file=self.stream, flush=True)

    def event(self, kind: str, **data: Any) -> None:
        if self.machine:
            line = json.dumps({"event": kind, **data}, default=str)
            print(EVENT_PREFIX + line, file=self.stream, flush=True)


@dataclass
class TrainJob:
    config: TrainConfig
    plan: DataPlan
    run_dir: Path
    reporter: Reporter
    device: str | None = None
    # Weights to score on the same validation images before training, for comparison.
    baseline: Path | None = None
    run: dict[str, Any] = field(default_factory=dict)

    def save_run(self) -> None:
        with open(self.run_dir / "run.json", "w", encoding="utf-8") as fh:
            json.dump(self.run, fh, indent=2, default=str)
            fh.write("\n")


def _rcnn_boxes(job: TrainJob) -> dict[str, Any]:
    from uapd.train.rcnn import train_rcnn

    return train_rcnn(job)


def _yolo_seg(job: TrainJob) -> dict[str, Any]:
    from uapd.train.yolo_seg import train_yolo_seg

    return train_yolo_seg(job)


TRAINERS: dict[str, Callable[[TrainJob], dict[str, Any]]] = {
    "rcnn-boxes": _rcnn_boxes,
    "yolo-seg": _yolo_seg,
}


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def run_training(
    config: TrainConfig,
    data: list[str | Path],
    out_dir: str | Path,
    reporter: Reporter | None = None,
    device: str | None = None,
    baseline: str | Path | None = None,
) -> dict[str, Any]:
    """Train on the label folders in ``data``; returns the finished run record."""
    from uapd.eval.headless import _git_commit

    reporter = reporter or Reporter()
    if config.kind not in TRAINERS:
        known = ", ".join(sorted(TRAINERS))
        raise TrainError(f"unknown trainer {config.kind!r} (available: {known})")
    if baseline is not None and not Path(baseline).is_file():
        raise TrainError(f"baseline weights not found: {baseline}")

    plan = plan_data(data, config)
    reporter.log(format_report(plan.report))
    if not plan.train:
        raise DataError("no training images")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir) / f"{stamp}-{config.kind}"
    run_dir.mkdir(parents=True, exist_ok=False)
    job = TrainJob(
        config=config,
        plan=plan,
        run_dir=run_dir,
        reporter=reporter,
        device=device,
        baseline=Path(baseline) if baseline else None,
    )
    job.run = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "running",
        "started": now(),
        "uapd_version": __version__,
        "commit": _git_commit(),
        "train_config": config.to_dict(),
        "train_config_fingerprint": config.fingerprint(),
        "data": plan.report,
        "epochs": [],
    }
    job.save_run()
    reporter.log(f"\nRun folder: {run_dir}")

    try:
        result = TRAINERS[config.kind](job)
    except BaseException as exc:
        job.run.update(status="failed", ended=now(), error=repr(exc))
        job.save_run()
        raise
    job.run.update(status="finished", ended=now(), result=result)
    job.save_run()
    reporter.event("done", run_dir=str(run_dir), **result)
    return job.run
