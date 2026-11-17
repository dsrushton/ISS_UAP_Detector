"""Command line entry point: ``uapd``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from uapd.config import Config, ConfigError
from uapd.eval.headless import FRAME_MODES, dumps_report, run_eval, write_report
from uapd.pipeline import PipelineError
from uapd.pipelines import PIPELINES, build_pipeline
from uapd.sources import SourceError, open_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uapd", description="ISS UAP Detector V2")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser(
        "eval", help="run a pipeline over a video file or frame folder, with no display"
    )
    ev.add_argument("source", help="video file or folder of image frames")
    ev.add_argument(
        "--pipeline",
        default="core",
        choices=sorted(PIPELINES),
        help="pipeline to run (default: core, the V2 pipeline; legacy-v1 is the V1 baseline)",
    )
    ev.add_argument("--out", help="write the JSON report here (default: print to stdout)")
    ev.add_argument("--max-frames", type=int, help="stop after this many frames")
    ev.add_argument(
        "--frames",
        dest="frame_mode",
        default="detections",
        choices=FRAME_MODES,
        help="which frames to list individually: those with detections (default), "
        "those with any candidate including rejected ones, or all",
    )
    ev.add_argument("--weights", default="best_map_model.pth", help="path to the RCNN weights file")
    ev.add_argument("--profile", help="TOML profile to apply over the defaults (core only)")
    ev.add_argument(
        "--rcnn-cycle",
        type=int,
        help="override V1's RCNN_DETECTION_CYCLE, frames between runs (legacy-v1 only)",
    )

    run = sub.add_parser(
        "run", help="run the live app: window, recording and restream, with V2 detection"
    )
    run.add_argument(
        "source", nargs="?", help="video file or YouTube URL (default: the live feed in the config)"
    )
    run.add_argument(
        "--weights", default="best_map_model.pth", help="path to the RCNN weights file"
    )
    run.add_argument("--profile", help="TOML profile to apply over the defaults")

    cfg = sub.add_parser("config", help="print the effective V2 configuration as JSON")
    cfg.add_argument("--profile", help="TOML profile to apply over the defaults")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "eval":
            return _cmd_eval(args)
        if args.command == "config":
            return _cmd_config(args)
        if args.command == "run":
            return _cmd_run(args)
    except (PipelineError, SourceError, ConfigError, ValueError) as exc:
        print(f"uapd: error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    source = open_source(args.source)
    options = {}
    if args.pipeline == "legacy-v1":
        if args.profile:
            raise ValueError("--profile applies to the core pipeline; V1 reads SOD_Constants.py")
        options = {"weights": args.weights, "rcnn_cycle": args.rcnn_cycle}
    elif args.pipeline == "core":
        if args.rcnn_cycle is not None:
            raise ValueError(
                "--rcnn-cycle applies to legacy-v1; set regions.cycle_frames in a profile"
            )
        config = Config.load(args.profile) if args.profile else Config()
        options = {"weights": args.weights, "config": config}
    pipeline = build_pipeline(args.pipeline, **options)
    try:
        report = run_eval(pipeline, source, max_frames=args.max_frames, frame_mode=args.frame_mode)
    finally:
        pipeline.close()

    if args.out:
        write_report(report, args.out)
        summary = report["summary"]
        print(
            f"{summary['frames']} frames, {summary['detections']} detections on "
            f"{summary['frames_with_detections']} frames, {summary['transients']} transients, "
            f"{summary['rejected']} rejected candidates -> {args.out}",
            file=sys.stderr,
        )
    else:
        print(dumps_report(report))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from uapd.runtime.live import run_live

    config = Config.load(args.profile) if args.profile else Config()
    try:
        run_live(config, args.weights, args.source)
    except KeyboardInterrupt:
        pass
    # The legacy restream leaves threads running that keep the process alive after
    # everything has been cleaned up, so end the process here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _cmd_config(args: argparse.Namespace) -> int:
    config = Config.load(args.profile) if args.profile else Config()
    print(json.dumps({"fingerprint": config.fingerprint(), **config.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
