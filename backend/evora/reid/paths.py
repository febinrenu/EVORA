"""Paths: the cameras an identity passed through, in time order."""
from __future__ import annotations

from contracts.models import PathHop

from evora.core.db import Database, open_db
from evora.core.workspace import Workspace


def evidence_id_for(track_id: str) -> str:
    """A track id as an id that is safe in media URLs (`cam_01:t000012` -> `cam_01_t000012`)."""
    return track_id.replace(":", "_")


def path_for(global_id: str, workspace: Workspace | None = None, *, db: Database | None = None) -> list[PathHop]:
    """Hops for one identity. Consecutive tracks in the same camera merge into one hop."""
    if db is None:
        from evora.perception.pipeline import resolve_workspace

        db = open_db((workspace or resolve_workspace()).db_path)
    with db.read() as c:
        rows = c.execute(
            "SELECT t.id, t.camera_id, t.t_start, t.t_end, t.quality, c.name FROM tracks t "
            "JOIN cameras c ON c.id = t.camera_id WHERE t.global_id=? ORDER BY t.t_start, t.id", (global_id,),
        ).fetchall()
    hops: list[dict] = []
    for r in rows:
        quality = r["quality"] or 0.0
        if hops and hops[-1]["camera_id"] == r["camera_id"]:
            h = hops[-1]
            h["t_out"] = max(h["t_out"], r["t_end"])
            if quality > h["quality"]:
                h["track_id"], h["quality"] = r["id"], quality
        else:
            hops.append({"camera_id": r["camera_id"], "camera_name": r["name"], "t_in": r["t_start"],
                         "t_out": r["t_end"], "track_id": r["id"], "quality": quality})
    return [
        PathHop(camera_id=h["camera_id"], camera_name=h["camera_name"], t_in=h["t_in"], t_out=h["t_out"],
                evidence_id=evidence_id_for(h["track_id"]))
        for h in hops
    ]
