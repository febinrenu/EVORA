"""Bridge to the perception package (M2). Uses the real functions when present, else a labelled stub.

Contract (PLAN.md §5.6):
    detect_clock(path: str) -> tuple[float, str]
    ingest(cam, profile, layers, on_progress) -> None
"""
from __future__ import annotations

import importlib
import inspect
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from contracts.models import CameraInfo, IngestJob, Zone

from evora.core.media import ProbeResult

log = logging.getLogger("evora.perception_adapter")

ProgressFn = Callable[[IngestJob], None]


MODULES = (
    "evora.perception", "evora.perception.clock", "evora.perception.pipeline", "evora.perception.embed",
    "evora.perception.events", "evora.reid",
)


def _find(name: str) -> Callable | None:
    for module in MODULES:
        try:
            fn = getattr(importlib.import_module(module), name, None)
        except Exception as exc:  # noqa: BLE001 - a half-installed stack (missing DLL, wrong CUDA) must mean "not available"
            log.debug("could not import %s: %s", module, exc)
            continue
        if callable(fn):
            return fn
    return None


def detect_clock(path: Path, probed: ProbeResult | None = None) -> tuple[float, str]:
    real = _find("detect_clock")
    if real is not None:
        return real(str(path))
    if probed is not None and probed.creation_time:
        try:
            return datetime.fromisoformat(probed.creation_time.replace("Z", "+00:00")).timestamp(), "metadata"
        except ValueError:
            log.warning("unparseable creation_time %r", probed.creation_time)
    return os.path.getmtime(path), "manual"


_warned = False


def ingest(
    cam: CameraInfo, profile: str, layers: set[str], on_progress: ProgressFn,
    tick_s: float = 0.05, stop: threading.Event | None = None, ws: Any | None = None,
) -> None:
    """Run M2's pipeline if installed; otherwise report simulated progress so the platform can be tested."""
    global _warned
    real = _find("ingest")
    if real is not None:
        # M2's pipeline resolves its own workspace unless told which one the app is using
        accepts_ws = ws is not None and "ws" in inspect.signature(real).parameters
        real(cam, profile, layers, on_progress, **({"ws": ws} if accepts_ws else {}))
        return
    if not _warned:
        log.warning("perception.ingest is not installed: using the simulated ingest stub")
        _warned = True
    for layer in sorted(layers):
        for step in range(1, 5):
            if stop is not None and stop.is_set():
                return
            time.sleep(tick_s)
            on_progress(IngestJob(
                id="", camera_id=cam.id, state="running", layer=layer,  # type: ignore[arg-type]
                progress=step / 4, video_s_per_s=None,
            ))


def get_blur_faces() -> Callable[[bytes], bytes] | None:
    """M2's `blur_faces(jpeg) -> jpeg`, or None when the face model is not installed."""
    return _find("blur_faces")


def get_query_embedder() -> Any | None:
    """M2's image-text query embedder (object with `embed_text(str) -> ndarray`), or None when not installed."""
    factory = _find("query_embedder")
    if factory is None:
        return None
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 - a model that fails to load means "retrieval not ready", not a crash
        log.warning("query embedder could not be loaded: %s", exc)
        return None


def _accepts(fn: Callable, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def recompute_events(camera_id: str, zones: list[Zone], db: Any | None = None) -> int | None:
    """M2's retroactive event recompute: the number of events written, or None when it is not available.

    `db` is the app's database; M2's function would otherwise guess the workspace from the environment.
    """
    real = _find("recompute_events")
    if real is None:
        return None
    try:
        extra = {"db": db} if db is not None and _accepts(real, "db") else {}
        return int(real(camera_id, zones, **extra))
    except Exception as exc:  # noqa: BLE001 - a failed recompute leaves the zone saved and reports "pending"
        log.warning("recompute_events failed for %s: %s", camera_id, exc)
        return None


def link_identities(workspace: Any | None = None) -> int | None:
    """M2's cross-camera linking: how many identities now span two or more cameras, or None when unavailable."""
    real = _find("link_global_ids")
    if real is None:
        return None
    try:
        extra = {"workspace": workspace} if workspace is not None and _accepts(real, "workspace") else {}
        return int(real(**extra))
    except Exception as exc:  # noqa: BLE001 - linking is an upgrade over per-camera tracks, never a reason to fail an ingest
        log.warning("link_global_ids failed: %s", exc)
        return None
