"""Ingest job runner: one job per camera, resumable by layer, one failure never stops the others."""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from contracts.models import CameraInfo, IngestJob

from evora.core import cameras as cams
from evora.core import perception_adapter
from evora.core.bus import Bus
from evora.core.db import Database

log = logging.getLogger("evora.jobs")

LAYERS = ("L0", "L1", "L2", "L3")
IngestFn = Callable[[CameraInfo, str, set[str], Callable[[IngestJob], None]], None]


def auto_workers(configured: int) -> int:
    return configured if configured > 0 else max(1, (os.cpu_count() or 4) // 4)


class JobRunner:
    def __init__(
        self, db: Database, bus: Bus, *, profile: str = "cpu", workers: int = 0,
        default_layers: list[str] | None = None, ingest_fn: IngestFn | None = None, stub_tick_s: float = 0.05,
        ws: Any | None = None, deferred_layers: list[str] | None = None,
    ):
        self.db, self.bus, self.profile = db, bus, profile
        self.default_layers = list(default_layers or LAYERS)
        # Layers that run only after every camera has reached the others (captions: slow, and they must never hold up
        # searchable footage). They are submitted by the runner itself once the batch is idle, once per camera.
        self.deferred_layers = [ly for ly in (deferred_layers or []) if ly in LAYERS]
        self._deferred_tried: set[str] = set()
        self._index_work_done = False  # real indexing finished since the last idle hook
        self._ingest = ingest_fn or (
            lambda c, p, ly, cb: perception_adapter.ingest(c, p, ly, cb, tick_s=stub_tick_s, ws=ws)
        )
        self._pool = ThreadPoolExecutor(max_workers=auto_workers(workers), thread_name_prefix="ingest")
        self._lock = threading.Lock()
        self._wanted: dict[str, set[str]] = {}
        self.on_done: Callable[[str], None] | None = None  # called when a camera finishes ingesting something
        self.on_idle: Callable[[], None] | None = None  # called once the last job of a batch has finished
        self.before_run: Callable[[str], None] | None = None  # e.g. wait for the camera's clock reading
        self._idle_lock = threading.Lock()

    # --- public API ---
    def submit(self, camera_ids: list[str], layers: list[str] | None = None) -> list[IngestJob]:
        wanted = set(layers or self.default_layers)
        unknown = wanted - set(LAYERS)
        if unknown:
            raise ValueError(f"unknown layers: {sorted(unknown)}")
        jobs: list[IngestJob] = []
        for cid in camera_ids:
            cam = cams.get_camera(self.db, cid)  # raises CameraNotFound
            jobs.append(self._enqueue(cam, wanted))
        return jobs

    def recover(self) -> int:
        """Re-queue jobs a crash left queued or running; finished layers are skipped."""
        with self.db.read() as c:
            rows = c.execute("SELECT id, camera_id FROM ingest_jobs WHERE state IN ('queued','running')").fetchall()
        for row in rows:
            with self.db.write() as c:
                c.execute("UPDATE ingest_jobs SET state='error', error='interrupted by restart' WHERE id=?", (row["id"],))
        return sum(1 for r in rows if self._enqueue(cams.get_camera(self.db, r["camera_id"]), set(self.default_layers)))

    def retry_failed(self) -> list[IngestJob]:
        """Index again every recorded-file camera whose indexing stopped with an error (for example after setup was run)."""
        with self.db.read() as c:
            ids = [r["id"] for r in c.execute("SELECT id FROM cameras WHERE status='error' AND kind='file' ORDER BY id")]
        jobs = []
        for cid in ids:
            try:
                jobs.append(self._enqueue(cams.get_camera(self.db, cid), set(self.default_layers)))
            except cams.CameraNotFound:
                continue
        if jobs:
            log.info("re-indexing %d camera(s) that had stopped with an error", len(jobs))
        return jobs

    def get(self, job_id: str) -> IngestJob:
        with self.db.read() as c:
            return self._row_to_job(c.execute("SELECT * FROM ingest_jobs WHERE id=?", (job_id,)).fetchone())

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)

    # --- internals ---
    def _enqueue(self, cam: CameraInfo, wanted: set[str]) -> IngestJob:
        with self._lock:
            with self.db.read() as c:
                active = c.execute(
                    "SELECT * FROM ingest_jobs WHERE camera_id=? AND state IN ('queued','running')", (cam.id,)
                ).fetchone()
            if active is not None:
                return self._row_to_job(active)
            todo = {ly for ly in wanted if ly not in cam.layers}
            if todo - set(self.deferred_layers):
                self._deferred_tried.discard(cam.id)  # new indexing work: captions may follow again
            job_id = f"job_{uuid.uuid4().hex[:8]}"
            state = "queued" if todo else "done"
            with self.db.write() as c:
                c.execute(
                    "INSERT INTO ingest_jobs(id,camera_id,state,layer,progress,updated_at) VALUES(?,?,?,?,?,?)",
                    (job_id, cam.id, state, None, 0.0 if todo else 1.0, time.time()),
                )
            self._wanted[job_id] = todo
        job = self.get(job_id)
        self._publish(job)
        if todo:
            self._pool.submit(self._run, job_id)
        return job

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        todo = self._wanted.pop(job_id, set())
        self._update(job_id, state="running")
        late_only = bool(todo) and todo <= set(self.deferred_layers)  # captions after indexing: the camera stays searchable
        if not late_only:
            cams.set_status(self.db, job.camera_id, "ingesting")
            self.bus.publish("camera", {"camera_id": job.camera_id, "status": "ingesting"})
        started = time.monotonic()

        def on_progress(p: IngestJob) -> None:
            fields: dict = {"layer": p.layer, "progress": p.progress, "rate": p.video_s_per_s}
            self._update(job_id, **fields)
            if p.layer is not None and p.progress >= 1.0:
                cams.add_layers(self.db, job.camera_id, [p.layer])

        try:
            if self.before_run is not None:
                self.before_run(job.camera_id)
            cam = cams.get_camera(self.db, job.camera_id)  # read after the hook: the clock may just have been set
            self._ingest(cam, self.profile, todo, on_progress)
            if not cams.get_camera(self.db, job.camera_id).layers:
                raise RuntimeError("ingest finished without completing any layer")
        except Exception as exc:  # noqa: BLE001 - a failing camera must never stop other jobs
            log.exception("ingest failed for %s", job.camera_id)
            if not late_only:  # a failed caption pass must not mark a searchable camera as broken
                cams.set_status(self.db, job.camera_id, "error")
                self.bus.publish("camera", {"camera_id": job.camera_id, "status": "error"})
            self._update(job_id, state="error", error=str(exc) or exc.__class__.__name__)
            # the cameras that did finish still deserve linking when the failed one was the last
            self._maybe_idle(index_work=not late_only)
            return
        log.info("ingest %s done in %.1fs", job.camera_id, time.monotonic() - started)
        cams.set_status(self.db, job.camera_id, "ready")  # camera first, so a finished job never sees a stale camera
        self.bus.publish("camera", {"camera_id": job.camera_id, "status": "ready"})
        if self.on_done is not None and not late_only:
            try:
                self.on_done(job.camera_id)
            except Exception:  # noqa: BLE001 - a follow-up step must never turn a finished ingest into a failed one
                log.exception("post-ingest hook failed for %s", job.camera_id)
        self._update(job_id, state="done", progress=1.0)
        self._maybe_idle(index_work=not late_only)

    def _maybe_idle(self, index_work: bool = True) -> None:
        """After the last camera of a batch: link identities across cameras, then queue the deferred layers (captions)."""
        with self._idle_lock:
            self._index_work_done = self._index_work_done or index_work
            with self.db.read() as c:
                busy = c.execute("SELECT count(*) FROM ingest_jobs WHERE state IN ('queued','running')").fetchone()[0]
            if busy:
                return
            if self.on_idle is not None and self._index_work_done:  # a captions-only batch has nothing new to link
                self._index_work_done = False
                try:
                    self.on_idle()
                except Exception:  # noqa: BLE001 - a follow-up step must never turn a finished ingest into a failed one
                    log.exception("post-batch hook failed")
            self._index_work_done = False
            self._submit_deferred()

    def _submit_deferred(self) -> None:
        """Queue the deferred layers for every searchable camera that lacks them, once per camera."""
        if not self.deferred_layers:
            return
        wanted = set(self.deferred_layers)
        with self.db.read() as c:
            ready = [r["id"] for r in c.execute("SELECT id FROM cameras WHERE status='ready' AND kind='file'")]
        for cid in ready:
            if cid in self._deferred_tried:
                continue
            try:
                cam = cams.get_camera(self.db, cid)
            except cams.CameraNotFound:
                continue
            self._deferred_tried.add(cid)
            if wanted - set(cam.layers):
                self._enqueue(cam, wanted)

    def _update(self, job_id: str, **fields: object) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.db.write() as c:
            c.execute(f"UPDATE ingest_jobs SET {cols}, updated_at=? WHERE id=?", (*fields.values(), time.time(), job_id))  # noqa: S608 - column names are fixed literals above
        self._publish(self.get(job_id))

    def _publish(self, job: IngestJob) -> None:
        self.bus.publish("ingest", {"job": job.model_dump()})

    @staticmethod
    def _row_to_job(row) -> IngestJob:
        return IngestJob(
            id=row["id"], camera_id=row["camera_id"], state=row["state"], layer=row["layer"],
            progress=row["progress"] or 0.0, video_s_per_s=row["rate"], error=row["error"],
        )
