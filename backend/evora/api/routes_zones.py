"""Zone routes: list, draw (or redraw) with an instant event recompute, delete."""
from __future__ import annotations

import json

from contracts.models import Zone
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from evora.api.context import AppContext
from evora.core import cameras as cams
from evora.core import zones


def _payload(raw: str) -> dict:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/zones")

    @router.get("")
    def list_zones(camera_id: str | None = None) -> list[Zone]:
        return zones.list_zones(ctx.db, camera_id)

    @router.post("")
    async def save_zone(zone: Zone):
        try:
            result = await run_in_threadpool(ctx.zones.save, zone)
        except cams.CameraNotFound:
            raise HTTPException(404, f"unknown camera: {zone.camera_id}") from None
        except zones.ZoneError as exc:
            raise HTTPException(422, str(exc)) from None
        header = "pending" if result.events is None else str(result.events)
        return JSONResponse(result.zone.model_dump(mode="json"), headers={"X-Evora-Events": header})

    @router.get("/{zone_id}/events")
    def zone_events(
        zone_id: str, kind: str | None = None, since: float | None = None, until: float | None = None,
        limit: int = Query(200, ge=1, le=1000),
    ) -> list[dict]:
        try:
            zone = zones.get_zone(ctx.db, zone_id)
        except zones.ZoneNotFound:
            raise HTTPException(404, "unknown zone") from None
        sql = "SELECT id,camera_id,track_id,kind,zone_id,t,payload FROM events WHERE camera_id=? AND zone_id=?"
        args: list = [zone.camera_id, zone_id]
        for clause, value in (("kind=?", kind), ("t>=?", since), ("t<=?", until)):
            if value is not None:
                sql, args = f"{sql} AND {clause}", [*args, value]
        with ctx.db.read() as c:
            rows = c.execute(f"{sql} ORDER BY t DESC LIMIT ?", [*args, limit]).fetchall()
        return [{**dict(r), "payload": _payload(r["payload"])} for r in rows]

    @router.delete("/{zone_id}")
    async def delete_zone(zone_id: str):
        try:
            await run_in_threadpool(ctx.zones.delete, zone_id)
        except zones.ZoneNotFound:
            raise HTTPException(404, "unknown zone") from None
        return {"ok": True}

    return router
