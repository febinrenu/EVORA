"""Changing a camera's clock after it was indexed: every stored time of that camera moves with it.

Tracks, track points, events, scene and crop vectors, evidence and alerts all hold absolute times computed as
`t0 + position in the file`. Changing only `cameras.t0` (as before) left them on the old clock, so time filters missed
and clips showed the wrong moment. A clock is never changed while the camera is being indexed: the pipeline is writing
times on the old clock at that moment.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from evora.core import cameras as cams
from evora.core.db import Database
from evora.core.vectors import open_store
from evora.evidence import audit

log = logging.getLogger("evora.clock_shift")

_CAMERA_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
VECTOR_TIME_COLUMNS = {"crops": ("t",), "scenes": ("t",), "captions": ("t",), "reid": ("t_start", "t_end")}


class ClockBusy(Exception):
    """The camera is being indexed; its clock can change before or after, not during."""


def indexing(db: Database, camera_id: str) -> bool:
    with db.read() as c:
        return c.execute(
            "SELECT 1 FROM ingest_jobs WHERE camera_id=? AND state IN ('queued','running') LIMIT 1", (camera_id,),
        ).fetchone() is not None


def _shift_vectors(vectors_dir: Path, camera_id: str, delta: float) -> list[str]:
    store = open_store(vectors_dir)
    existing = set(store.list_tables().tables)
    done: list[str] = []
    for name, columns in VECTOR_TIME_COLUMNS.items():
        if name not in existing:
            continue
        store.open_table(name).update(
            where=f"camera_id = '{camera_id}'",  # the id is checked against a strict pattern above
            values_sql={col: f"{col} + ({delta!r})" for col in columns},
        )
        done.append(name)
    return done


def _shift_rows(db: Database, camera_id: str, delta: float) -> dict[str, int]:
    counts: dict[str, int] = {}
    with db.write() as c:
        counts["tracks"] = c.execute(
            "UPDATE tracks SET t_start=t_start+?, t_end=t_end+?, best_t=best_t+? WHERE camera_id=?",
            (delta, delta, delta, camera_id),
        ).rowcount
        counts["track_points"] = c.execute(
            "UPDATE track_points SET t=t+? WHERE track_id IN (SELECT id FROM tracks WHERE camera_id=?)", (delta, camera_id),
        ).rowcount
        counts["events"] = c.execute("UPDATE events SET t=t+? WHERE camera_id=?", (delta, camera_id)).rowcount
        counts["evidence"] = c.execute(
            "UPDATE evidence SET t_start=t_start+?, t_end=t_end+?, t_peak=t_peak+? WHERE camera_id=?",
            (delta, delta, delta, camera_id),
        ).rowcount
        alerts = c.execute("SELECT id, evidence FROM alerts WHERE camera_id=?", (camera_id,)).fetchall()
        for row in alerts:
            try:
                ev = json.loads(row["evidence"])
                for key in ("t_start", "t_end", "t_peak"):
                    if isinstance(ev.get(key), int | float):
                        ev[key] += delta
                blob = json.dumps(ev)
            except (ValueError, TypeError):
                blob = row["evidence"]  # a damaged row keeps its text; its time still moves below
            c.execute("UPDATE alerts SET t=t+?, evidence=? WHERE id=?", (delta, blob, row["id"]))
        counts["alerts"] = len(alerts)
    return counts


def set_camera_clock(
    db: Database, vectors_dir: Path, camera_id: str, t0: float, source: str = "manual", actor: str = "local",
    check_busy: bool = True,
) -> dict[str, Any]:
    """Set a camera's start time and move everything already stored for it by the same amount."""
    if not _CAMERA_ID.match(camera_id):
        raise ValueError(f"invalid camera id: {camera_id!r}")
    cam = cams.get_camera(db, camera_id)  # raises CameraNotFound
    if check_busy and indexing(db, camera_id):
        raise ClockBusy(f"{cam.name} is being indexed; set its clock when indexing has finished")
    delta = float(t0) - cam.t0
    moved: dict[str, int] = {}
    tables: list[str] = []
    if delta:
        tables = _shift_vectors(vectors_dir, camera_id, delta)
        try:
            moved = _shift_rows(db, camera_id, delta)
        except Exception:
            _shift_vectors(vectors_dir, camera_id, -delta)  # keep vectors and rows on the same clock
            raise
    with db.write() as c:
        c.execute("UPDATE cameras SET t0=?, t0_source=? WHERE id=?", (float(t0), source, camera_id))
    audit.record(db, "clock_change", {
        "camera_id": camera_id, "from_t0": cam.t0, "to_t0": float(t0), "from_source": cam.t0_source, "to_source": source,
        "shift_s": delta, "rows": moved, "vector_tables": tables,
    }, actor=actor)
    log.info("clock of %s moved by %+.3f s (%s)", camera_id, delta, moved)
    return {"shift_s": delta, "rows": moved, "vector_tables": tables}
