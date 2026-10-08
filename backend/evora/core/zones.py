"""Zone rows (line, polygon, whole frame): the only code that reads and writes the `zones` table."""
from __future__ import annotations

import json
import re
import time
import uuid

from contracts.models import Zone

from evora.core import cameras as cams
from evora.core.db import Database

_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class ZoneError(ValueError):
    pass


class ZoneNotFound(KeyError):
    pass


def new_id() -> str:
    return f"z_{uuid.uuid4().hex[:8]}"


def validate(zone: Zone) -> None:
    """Geometry rules shared by the drawing route and the clarify flow."""
    if not _ID.match(zone.id):
        raise ZoneError("zone id may only contain letters, digits, '_' and '-'")
    if any(not (0.0 <= v <= 1.0) for pt in zone.points for v in pt):
        raise ZoneError("region points must be inside the frame (0 to 1)")
    if zone.kind == "line" and len(zone.points) != 2:
        raise ZoneError("a line needs exactly two points")
    if zone.kind == "polygon" and len(zone.points) < 3:
        raise ZoneError("a polygon needs at least three points")
    if zone.kind == "frame" and zone.points:
        raise ZoneError("a whole-frame zone has no points")


def _to_zone(row) -> Zone:
    return Zone(
        id=row["id"], camera_id=row["camera_id"], kind=row["kind"],
        points=[tuple(p) for p in json.loads(row["points"] or "[]")], direction=row["direction"],
    )


def list_zones(db: Database, camera_id: str | None = None) -> list[Zone]:
    sql, args = "SELECT * FROM zones", ()
    if camera_id:
        sql, args = sql + " WHERE camera_id=?", (camera_id,)
    with db.read() as c:
        # rowid breaks ties in insertion order: zones saved within one clock tick (about 16 ms on Windows) share created_at
        return [_to_zone(r) for r in c.execute(sql + " ORDER BY created_at, rowid", args)]


def get_zone(db: Database, zone_id: str) -> Zone:
    with db.read() as c:
        row = c.execute("SELECT * FROM zones WHERE id=?", (zone_id,)).fetchone()
    if row is None:
        raise ZoneNotFound(zone_id)
    return _to_zone(row)


def fact_id_of(db: Database, zone_id: str) -> str | None:
    with db.read() as c:
        row = c.execute("SELECT fact_id FROM zones WHERE id=?", (zone_id,)).fetchone()
    return row["fact_id"] if row else None


def save(db: Database, zone: Zone, fact_id: str | None = None) -> Zone:
    """Insert a zone, or replace the geometry of an existing one. The camera of an existing zone cannot change."""
    validate(zone)
    cams.get_camera(db, zone.camera_id)  # raises CameraNotFound
    try:
        existing = get_zone(db, zone.id)
    except ZoneNotFound:
        existing = None
    if existing is not None and existing.camera_id != zone.camera_id:
        raise ZoneError("a zone cannot move to another camera; draw a new one")
    with db.write() as c:
        if existing is None:
            c.execute(
                "INSERT INTO zones(id,camera_id,kind,points,direction,fact_id,created_at) VALUES(?,?,?,?,?,?,?)",
                (zone.id, zone.camera_id, zone.kind, json.dumps(zone.points), zone.direction, fact_id, time.time()),
            )
        else:
            c.execute(
                "UPDATE zones SET kind=?, points=?, direction=?, fact_id=COALESCE(?, fact_id) WHERE id=?",
                (zone.kind, json.dumps(zone.points), zone.direction, fact_id, zone.id),
            )
    return get_zone(db, zone.id)


def link_fact(db: Database, zone_id: str, fact_id: str) -> None:
    with db.write() as c:
        c.execute("UPDATE zones SET fact_id=? WHERE id=?", (fact_id, zone_id))


def clear_events(db: Database, zone_id: str) -> int:
    with db.write() as c:
        return c.execute("DELETE FROM events WHERE zone_id=?", (zone_id,)).rowcount


def delete(db: Database, zone_id: str) -> None:
    get_zone(db, zone_id)
    with db.write() as c:
        c.execute("DELETE FROM events WHERE zone_id=?", (zone_id,))
        c.execute("DELETE FROM zones WHERE id=?", (zone_id,))
