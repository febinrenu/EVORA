"""Reading a camera's on-screen clock in the background, so an upload never waits for the vision model.

An upload registers the camera at once with the quick clock (file name, container metadata, else the file time). When
that was only the file time, the slow reading (on-screen timestamp, then a filmed slate) runs here. Indexing of that
camera waits for the reading first, so the pipeline always starts on the final clock and nothing has to be moved.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evora.core import cameras as cams
from evora.core import clock_shift
from evora.core.bus import Bus
from evora.core.db import Database

log = logging.getLogger("evora.clock_reader")

ReadFn = Callable[[Path], tuple[float, str]]
BETTER = ("filename", "metadata", "osd", "slate")  # a reading from any of these replaces the file time


class ClockReader:
    def __init__(self, db: Database, vectors_dir: Path, bus: Bus, read_fn: ReadFn, *, workers: int = 1) -> None:
        self.db, self.vectors_dir, self.bus, self.read_fn = db, vectors_dir, bus, read_fn
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="clock")  # the vision model is serial anyway
        self._pending: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def submit(self, camera_id: str, path: Path) -> None:
        with self._lock:
            if camera_id in self._pending:
                return
            done = threading.Event()
            self._pending[camera_id] = done
        self.bus.publish("clock", {"camera_id": camera_id, "state": "reading"})
        try:
            self._pool.submit(self._read, camera_id, path, done)
        except RuntimeError:  # shutting down
            self._finish(camera_id, done)

    def pending(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._pending

    def wait(self, camera_id: str, timeout: float = 90.0) -> bool:
        """Block until the camera's clock reading is over (True), or the timeout passed (False)."""
        with self._lock:
            done = self._pending.get(camera_id)
        return True if done is None else done.wait(timeout)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            for done in self._pending.values():
                done.set()  # nobody waits on a reading that will never come
            self._pending.clear()

    def _finish(self, camera_id: str, done: threading.Event) -> None:
        with self._lock:
            self._pending.pop(camera_id, None)
        done.set()

    def _read(self, camera_id: str, path: Path, done: threading.Event) -> None:
        state: dict = {"camera_id": camera_id, "state": "done"}
        try:
            t0, source = self.read_fn(path)
            cam = cams.get_camera(self.db, camera_id)
            if source in BETTER and cam.t0_source == "manual" and not self._set_by_hand(camera_id):
                # the camera's indexing is waiting for this reading, so nothing is being written on the old clock
                clock_shift.set_camera_clock(
                    self.db, self.vectors_dir, camera_id, t0, source=source, actor="clock reader", check_busy=False,
                )
                state.update(t0=t0, t0_source=source)
            else:
                state.update(t0=cam.t0, t0_source=cam.t0_source)
        except Exception as exc:  # noqa: BLE001 - a failed reading keeps the file time; it never blocks indexing
            log.exception("clock reading failed for %s", camera_id)
            state.update(state="failed", error=str(exc)[:200])
        finally:
            self._finish(camera_id, done)
            self.bus.publish("clock", state)

    def _set_by_hand(self, camera_id: str) -> bool:
        """An operator typed a clock while the reading ran: theirs wins."""
        from evora.evidence import audit

        return any(e["detail"].get("camera_id") == camera_id and e["actor"] != "clock reader"
                   for e in audit.entries(self.db, "clock_change"))
