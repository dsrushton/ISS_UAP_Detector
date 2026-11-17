"""Registry of runnable pipelines. Imports are lazy so heavy dependencies load on demand."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from uapd.pipeline import Pipeline, PipelineError


def _legacy_v1(**options: Any) -> Pipeline:
    from uapd.pipelines.legacy_v1 import LegacyV1Pipeline

    return LegacyV1Pipeline(**options)


def _core(**options: Any) -> Pipeline:
    from uapd.pipelines.core import build_core_pipeline

    return build_core_pipeline(**options)


PIPELINES: dict[str, Callable[..., Pipeline]] = {
    "core": _core,
    "legacy-v1": _legacy_v1,
}


def build_pipeline(name: str, **options: Any) -> Pipeline:
    try:
        factory = PIPELINES[name]
    except KeyError:
        known = ", ".join(sorted(PIPELINES))
        raise PipelineError(f"unknown pipeline {name!r} (available: {known})") from None
    return factory(**options)
