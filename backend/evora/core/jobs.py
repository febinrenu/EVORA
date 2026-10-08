"""Ingest job runner: one job per camera, resumable by layer, one failure never stops the others."""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

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
    ):
        self.db, self.bus, self.profile = db, bus, profile
        self.default_layers = list(default_layers or LAYERS)
        self._ingest = ingest_fn or (lambda c, p, ly, cb: perception_adapter.ingest(c, p, ly, cb, tick_s=stub_tick_s))
        self._pool = ThreadPoolExecutor(max_workers=auto_workers(workers), thread_name_prefix="ingest")
        self._lock = threading.Lock()
        self._wanted: dict[str, set[str]] = {}

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
        cams.set_status(self.db, job.camera_id, "ingesting")
        self.bus.publish("camera", {"camera_id": job.camera_id, "status": "ingesting"})
        started = time.monotonic()

        def on_progress(p: IngestJob) -> None:
            fields: dict = {"layer": p.layer, "progress": p.progress, "rate": p.video_s_per_s}
            self._update(job_id, **fields)
            if p.layer is not None and p.progress >= 1.0:
                cams.add_layers(self.db, job.camera_id, [p.layer])

        try:
            cam = cams.get_camera(self.db, job.camera_id)
            self._ingest(cam, self.profile, todo, on_progress)
            if not cams.get_camera(self.db, job.camera_id).layers:
                raise RuntimeError("ingest finished without completing any layer")
        except Exception as exc:  # noqa: BLE001 - a failing camera must never stop other jobs
            log.exception("ingest failed for %s", job.camera_id)
            cams.set_status(self.db, job.camera_id, "error")
            self.bus.publish("camera", {"camera_id": job.camera_id, "status": "error"})
            self._update(job_id, state="error", error=str(exc) or exc.__class__.__name__)
            return
        log.info("ingest %s done in %.1fs", job.camera_id, time.monotonic() - started)
        cams.set_status(self.db, job.camera_id, "ready")  # camera first, so a finished job never sees a stale camera
        self.bus.publish("camera", {"camera_id": job.camera_id, "status": "ready"})
        self._update(job_id, state="done", progress=1.0)

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
