"""Build and register an Evidence for any stored track, so media and evidence-pack routes can serve it."""
from __future__ import annotations

import json
from collections.abc import Sequence

from contracts.models import Evidence

from evora.core import live_sessions
from evora.core.db import Database
from evora.evidence import store

TRACK_PAD_S = 5.0  # evidence window around the best moment of a track (the same padding the query router uses)


class TrackNotFound(KeyError):
    pass


def evidence_id_for(track_id: str) -> str:
    """A track id as an id that is safe in media URLs (`cam_01:t000012` becomes `cam_01_t000012`)."""
    return track_id.replace(":", "_")


def _offset(db: Database, row, t: float) -> float:  # noqa: ANN001
    mapped = live_sessions.file_offset(db, row["camera_id"], row["duration_s"], t)  # footage replayed as live
    return mapped if mapped is not None else max(t - row["t0"], 0.0)


def evidence_for_track(
    db: Database, track_id: str, *, evidence_id: str | None = None, score: float = 1.0, why: Sequence[str] = (),
) -> Evidence:
    with db.read() as c:
        row = c.execute(
            "SELECT t.id, t.camera_id, t.t_start, t.t_end, t.best_t, t.best_bbox, t.global_id, c.name, c.t0, c.duration_s "
            "FROM tracks t JOIN cameras c ON c.id = t.camera_id WHERE t.id=?", (track_id,),
        ).fetchone()
        if row is None:
            raise TrackNotFound(track_id)
        peak = row["best_t"] if row["best_t"] is not None else (row["t_start"] + row["t_end"]) / 2
        peak = min(max(peak, row["t_start"]), row["t_end"])
        point = c.execute(
            "SELECT x1,y1,x2,y2 FROM track_points WHERE track_id=? ORDER BY ABS(t-?) LIMIT 1", (track_id, peak)
        ).fetchone()
    bbox = tuple(point) if point is not None and None not in tuple(point) else None
    if bbox is None and row["best_bbox"]:
        try:
            bbox = tuple(json.loads(row["best_bbox"]))
        except (ValueError, TypeError):
            bbox = None
    eid = evidence_id or evidence_id_for(track_id)
    ev = Evidence(
        id=eid, camera_id=row["camera_id"], camera_name=row["name"], t_start=max(peak - TRACK_PAD_S, row["t_start"]),
        t_end=min(peak + TRACK_PAD_S, row["t_end"]), t_peak=peak, offset_s=_offset(db, row, peak), track_id=track_id,
        global_id=row["global_id"], bbox=bbox,  # type: ignore[arg-type]
        thumb_url=f"/api/media/thumb/{eid}.jpg", clip_url=f"/api/media/clip/{eid}.mp4", score=score, why=list(why),
    )
    store.register(db, ev)
    return ev
