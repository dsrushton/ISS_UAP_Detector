"""Command line entry point: ``uapd``."""

from __future__ import annotations

import argparse
import json
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
        "--pipeline", default="legacy-v1", choices=sorted(PIPELINES), help="pipeline to run"
    )
    ev.add_argument("--out", help="write the JSON report here (default: print to stdout)")
    ev.add_argument("--max-frames", type=int, help="stop after this many frames")
    ev.add_argument(
        "--frames",
        dest="frame_mode",
        default="detections",
        choices=FRAME_MODES,
        help="which frames to list individually (default: only frames with detections)",
    )
    legacy = ev.add_argument_group("legacy-v1 options")
    legacy.add_argument(
        "--weights", default="best_map_model.pth", help="path to the RCNN weights file"
    )
    legacy.add_argument(
        "--rcnn-cycle", type=int, help="override V1's RCNN_DETECTION_CYCLE (frames between runs)"
    )

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
    except (PipelineError, SourceError, ConfigError, ValueError) as exc:
        print(f"uapd: error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    source = open_source(args.source)
    options = {}
    if args.pipeline == "legacy-v1":
        options = {"weights": args.weights, "rcnn_cycle": args.rcnn_cycle}
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
            f"{summary['frames_with_detections']} frames -> {args.out}",
            file=sys.stderr,
        )
    else:
        print(dumps_report(report))
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    config = Config.load(args.profile) if args.profile else Config()
    print(json.dumps({"fingerprint": config.fingerprint(), **config.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
