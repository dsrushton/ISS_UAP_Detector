"""Headless evaluation: run a pipeline over footage and report what it found."""

from uapd.eval.headless import REPORT_SCHEMA_VERSION, run_eval, write_report

__all__ = ["REPORT_SCHEMA_VERSION", "run_eval", "write_report"]
