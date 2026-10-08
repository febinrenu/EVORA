"""Evidence registry: resolve an evidence id to camera, time window and box for the media routes."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass

from contracts.models import Evidence

from evora.core.db import Database

_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$")


class EvidenceError(ValueError):
    pass


class EvidenceNotFound(KeyError):
    pass


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    camera_id: str
    t_start: float
    t_end: float
    t_peak: float
    bbox: tuple[float, float, float, float] | None
    track_id: str | None


def valid_id(evidence_id: str) -> bool:
    return bool(_ID.match(evidence_id)) and ".." not in evidence_id


def register(db: Database, ev: Evidence) -> None:
    """Store (or refresh) one Evidence so its thumbnail and clip can be rendered later."""
    if not valid_id(ev.id):
        raise EvidenceError(f"invalid evidence id: {ev.id!r}")
    try:
        with db.write() as c:
            c.execute(
                "INSERT OR REPLACE INTO evidence(id,camera_id,t_start,t_end,t_peak,bbox,track_id,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (ev.id, ev.camera_id, ev.t_start, ev.t_end, ev.t_peak,
                 json.dumps(ev.bbox) if ev.bbox else None, ev.track_id, time.time()),
            )
    except sqlite3.IntegrityError as exc:
        raise EvidenceError(f"unknown camera {ev.camera_id!r} for evidence {ev.id!r}") from exc


def _from_row(row: sqlite3.Row) -> EvidenceRecord:
    bbox = tuple(json.loads(row["bbox"])) if row["bbox"] else None
    return EvidenceRecord(row["id"], row["camera_id"], row["t_start"], row["t_end"], row["t_peak"], bbox, row["track_id"])


def _candidates(data: object) -> list[object]:
    if not isinstance(data, dict):
        return []
    inner = data.get("evidence")
    if isinstance(inner, list):  # a stored Answer
        return [*inner, data.get("nearest_miss")]
    if isinstance(inner, dict):  # a stored Alert
        return [inner]
    return [data]  # a bare Evidence


def _scan(db: Database, evidence_id: str) -> Evidence | None:
    """Fallback for ids nobody registered: look inside stored answers and alerts."""
    with db.read() as c:
        blobs = [r["answer"] for r in c.execute("SELECT answer FROM query_log ORDER BY created_at DESC")]
        blobs += [r["evidence"] for r in c.execute("SELECT evidence FROM alerts")]
    for blob in blobs:
        if not blob or evidence_id not in blob:
            continue
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        candidates = _candidates(data)
        for cand in candidates:
            if isinstance(cand, dict) and cand.get("id") == evidence_id:
                return Evidence.model_validate(cand)
    return None


def get(db: Database, evidence_id: str) -> EvidenceRecord:
    if not valid_id(evidence_id):
        raise EvidenceError(f"invalid evidence id: {evidence_id!r}")
    with db.read() as c:
        row = c.execute("SELECT * FROM evidence WHERE id=?", (evidence_id,)).fetchone()
    if row is not None:
        return _from_row(row)
    found = _scan(db, evidence_id)
    if found is None:
        raise EvidenceNotFound(evidence_id)
    register(db, found)
    return get(db, evidence_id)
