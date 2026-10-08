"""Zone routes: list, draw (or redraw) with an instant event recompute, delete."""
from __future__ import annotations

from contracts.models import Zone
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from evora.api.context import AppContext
from evora.core import cameras as cams
from evora.core import zones


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

    @router.delete("/{zone_id}")
    async def delete_zone(zone_id: str):
        try:
            await run_in_threadpool(ctx.zones.delete, zone_id)
        except zones.ZoneNotFound:
            raise HTTPException(404, "unknown zone") from None
        return {"ok": True}

    return router
