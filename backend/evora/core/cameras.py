"""Camera rows: the single place that reads and writes the `cameras` table."""
from __future__ import annotations

import json
import sqlite3
import time

from contracts.models import CameraInfo

from evora.core.db import Database


class CameraNotFound(KeyError):
    pass


def _to_info(row: sqlite3.Row) -> CameraInfo:
    site = (row["site_x"], row["site_y"]) if row["site_x"] is not None and row["site_y"] is not None else None
    return CameraInfo(
        id=row["id"], name=row["name"], kind=row["kind"], source_uri=row["source_uri"],
        fps=row["fps"], width=row["width"], height=row["height"],
        t0=row["t0"], t0_source=row["t0_source"], duration_s=row["duration_s"],
        site_xy=site, status=row["status"], layers=json.loads(row["layers"]), ir_fraction=row["ir_fraction"],
    )


def list_cameras(db: Database) -> list[CameraInfo]:
    with db.read() as c:
        return [_to_info(r) for r in c.execute("SELECT * FROM cameras ORDER BY id")]


def get_camera(db: Database, camera_id: str) -> CameraInfo:
    with db.read() as c:
        row = c.execute("SELECT * FROM cameras WHERE id=?", (camera_id,)).fetchone()
    if row is None:
        raise CameraNotFound(camera_id)
    return _to_info(row)


def find_by_sha(db: Database, sha256: str) -> CameraInfo | None:
    """The camera already registered from a byte-identical upload, if any."""
    with db.read() as c:
        row = c.execute("SELECT * FROM cameras WHERE source_sha256=? ORDER BY id LIMIT 1", (sha256,)).fetchone()
    return _to_info(row) if row else None


def insert_camera(
    db: Database, *, name: str, kind: str, source_uri: str, t0: float, t0_source: str,
    sha256: str | None = None, fps: float | None = None, width: int | None = None,
    height: int | None = None, rotation: int = 0, duration_s: float | None = None,
) -> CameraInfo:
    with db.write() as c:
        last = c.execute("SELECT max(CAST(substr(id, 5) AS INTEGER)) FROM cameras").fetchone()[0] or 0
        cid = f"cam_{last + 1:02d}"
        c.execute(
            "INSERT INTO cameras(id,name,kind,source_uri,source_sha256,fps,width,height,rotation,t0,t0_source,"
            "duration_s,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, name, kind, source_uri, sha256, fps, width, height, rotation, t0, t0_source, duration_s, time.time()),
        )
    return get_camera(db, cid)


def update_camera(
    db: Database, camera_id: str, *, name: str | None = None, t0: float | None = None,
    site_xy: tuple[float, float] | None = None,
) -> CameraInfo:
    get_camera(db, camera_id)
    with db.write() as c:
        if name is not None:
            c.execute("UPDATE cameras SET name=? WHERE id=?", (name, camera_id))
        if t0 is not None:
            c.execute("UPDATE cameras SET t0=?, t0_source='manual' WHERE id=?", (t0, camera_id))
        if site_xy is not None:
            c.execute("UPDATE cameras SET site_x=?, site_y=? WHERE id=?", (site_xy[0], site_xy[1], camera_id))
    return get_camera(db, camera_id)


def set_status(db: Database, camera_id: str, status: str) -> None:
    with db.write() as c:
        c.execute("UPDATE cameras SET status=? WHERE id=?", (status, camera_id))


def add_layers(db: Database, camera_id: str, layers: list[str]) -> list[str]:
    """Record finished layers (kept in L0..L3 order) and return the full list."""
    with db.write() as c:
        row = c.execute("SELECT layers FROM cameras WHERE id=?", (camera_id,)).fetchone()
        if row is None:
            raise CameraNotFound(camera_id)
        done = sorted(set(json.loads(row["layers"])) | set(layers))
        c.execute("UPDATE cameras SET layers=? WHERE id=?", (json.dumps(done), camera_id))
    return done
