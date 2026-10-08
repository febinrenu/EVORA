"""Live analysis: run M2's `live_ingest` on a stream and feed every event to the alert engine."""
from __future__ import annotations

import inspect
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from evora.alerts.engine import AlertEngine
from evora.core import cameras as cams
from evora.core import perception_adapter
from evora.core.bus import Bus
from evora.core.db import Database
from evora.live.recorder import Recorder
from evora.live.restream import LiveError, ReplayManager

log = logging.getLogger("evora.live.runner")

_CAMERA_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
ACTIVE_STREAM_STATES = ("starting", "running", "retrying")
INSTALL_HINT = "Live analysis needs the perception stack: " + perception_adapter.SETUP_HINT.split(", then")[0] + "."


@dataclass
class Analyzer:
    camera_id: str
    url: str
    thread: threading.Thread | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    state: str = "starting"            # starting | running | retrying | stopped | error
    error: str | None = None


class LiveRunner:
    def __init__(
        self, db: Database, ws: Any, bus: Bus, engine: AlertEngine, replay: ReplayManager, *, profile: str = "cpu",
        onprem: Callable[[], bool] = lambda: False, allows: Callable[[str], bool] = lambda host: True,
        find_live_ingest: Callable[[], Callable[..., None] | None] = perception_adapter.get_live_ingest,
        join_timeout_s: float = 5.0, recorder: Recorder | None = None,
    ) -> None:
        self.db, self.ws, self.bus, self.engine, self.replay = db, ws, bus, engine, replay
        self.profile, self._onprem, self._allows = profile, onprem, allows
        self._find, self._join_timeout = find_live_ingest, join_timeout_s
        self.recorder = recorder  # keeps the last minutes of real cameras so their alerts have playable evidence
        self._lock = threading.RLock()
        self._running: dict[str, Analyzer] = {}

    # --- where a camera's live frames come from ---
    def stream_url(self, camera_id: str) -> str:
        """An RTSP camera's own address, or the replay stream of a recorded file. Raises LiveError when it is not live."""
        if not _CAMERA_ID.match(camera_id):
            raise LiveError(422, f"invalid camera id: {camera_id!r}")
        try:
            cam = cams.get_camera(self.db, camera_id)
        except cams.CameraNotFound:
            raise LiveError(404, f"unknown camera: {camera_id}") from None
        if cam.kind == "rtsp":
            url = cam.source_uri
        else:
            stream = next((s for s in self.replay.status()["streams"]
                           if s["camera_id"] == camera_id and s["state"] in ACTIVE_STREAM_STATES), None)
            if stream is None:
                raise LiveError(409, f"{cam.name} is a recorded file: start its replay first (POST /api/live/replay)")
            url = stream["url"]
        host = urlparse(url).hostname or ""
        if self._onprem() and not self._allows(host):
            raise LiveError(403, f"on-prem mode: {cam.name} is not on this machine, so it cannot be streamed")
        return url

    # --- control ---
    def start(self, camera_ids: list[str]) -> dict[str, Any]:
        if not camera_ids:
            raise LiveError(422, "choose at least one camera")
        fn = self._find()
        if fn is None:
            raise LiveError(503, INSTALL_HINT)
        plan = [(cid, self.stream_url(cid)) for cid in dict.fromkeys(camera_ids)]  # validate everything first
        with self._lock:
            for cid, url in plan:
                existing = self._running.get(cid)
                if existing and existing.state in ("starting", "running", "retrying"):
                    continue
                self._record(cid, url)
                analyzer = Analyzer(cid, url)
                analyzer.thread = threading.Thread(
                    target=self._run, args=(analyzer, fn), name=f"live-{cid}", daemon=True,
                )
                self._running[cid] = analyzer
                analyzer.thread.start()
        return self.status()

    def _record(self, camera_id: str, url: str) -> None:
        """Record a real camera while it is analysed. A recorded file needs no recording: the file is its own source."""
        if self.recorder is None:
            return
        try:
            if cams.get_camera(self.db, camera_id).kind == "rtsp":
                self.recorder.start(camera_id, url)
        except (LiveError, cams.CameraNotFound):
            log.warning("could not start recording %s: its alerts will have no clip", camera_id, exc_info=True)

    def stop(self, camera_ids: list[str] | None = None) -> dict[str, Any]:
        if self.recorder is not None:
            self.recorder.stop(camera_ids)
        with self._lock:
            targets = [a for cid, a in self._running.items() if camera_ids is None or cid in camera_ids]
        for a in targets:
            a.stop.set()
        for a in targets:
            if a.thread is not None:
                a.thread.join(self._join_timeout)
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            analyzers = [{"camera_id": a.camera_id, "state": a.state, "error": a.error} for a in self._running.values()]
        return {"analyzers": analyzers, "recordings": self.recorder.status() if self.recorder is not None else []}

    def shutdown(self) -> None:
        self.stop(None)

    # --- the worker ---
    def _announce(self, a: Analyzer, state: str) -> None:
        a.state = state
        self.bus.publish("analysis", {"camera_id": a.camera_id, "state": state, "error": a.error})

    def _on_event(self, event: dict) -> None:
        try:
            self.engine.evaluate(event, historical=False)  # live: a phone may be told
        except Exception:  # noqa: BLE001 - one bad event must never stop the stream
            log.exception("alert evaluation failed for event %s", event.get("id"))

    def _run(self, a: Analyzer, fn: Callable[..., None]) -> None:
        try:
            cam = cams.get_camera(self.db, a.camera_id)
            import time

            live_cam = cam.model_copy(update={"kind": "rtsp", "source_uri": a.url, "t0": time.time(), "t0_source": "live"})

            def on_state(state: str) -> None:
                if state == "running":
                    cams.set_status(self.db, a.camera_id, "live")
                self._announce(a, state)

            params = inspect.signature(fn).parameters
            extra: dict[str, Any] = {}
            if "ws" in params:
                extra["ws"] = self.ws
            if "on_state" in params:
                extra["on_state"] = on_state
            fn(live_cam, self.profile, a.stop, self._on_event, **extra)
            a.error = None
        except Exception as exc:  # noqa: BLE001 - report the reason and keep the rest of the platform running
            log.exception("live analysis of %s failed", a.camera_id)
            a.error = str(exc)[:200] or exc.__class__.__name__
        finally:
            failed = a.error is not None and not a.stop.is_set()
            try:
                cams.set_status(self.db, a.camera_id, "error" if failed else "ready")
            except Exception:  # noqa: BLE001
                log.exception("could not reset the status of %s", a.camera_id)
            self._announce(a, "error" if failed else "stopped")
