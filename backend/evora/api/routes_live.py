"""Replay-as-live: restream recorded files as RTSP so the rest of the system sees a live source."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from evora.api.context import AppContext
from evora.live.restream import LiveError


class ReplayRequest(BaseModel):
    camera_ids: list[str] = Field(min_length=1, max_length=64)
    speed: float | None = None


class StopRequest(BaseModel):
    camera_ids: list[str] | None = None


CANNED = {
    "server": {"ready": True, "port": 8554, "binary_found": True},
    "streams": [
        {"camera_id": "cam_01", "url": "rtsp://127.0.0.1:8554/cam_01", "state": "running", "speed": 1.0, "restarts": 0,
         "transcoding": False, "started_at": 1790000000.0, "error": None},
    ],
}


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/live")

    @router.get("")
    def status():
        return CANNED if ctx.mock else ctx.live.status()

    @router.post("/replay")
    async def start(body: ReplayRequest):
        if ctx.mock:
            return CANNED
        try:
            return await run_in_threadpool(ctx.live.start, body.camera_ids, body.speed)
        except LiveError as exc:
            raise HTTPException(exc.status, exc.message) from None

    @router.post("/replay/stop")
    async def stop(body: StopRequest | None = None):
        if ctx.mock:
            return {**CANNED, "streams": []}
        return await run_in_threadpool(ctx.live.stop, body.camera_ids if body else None)

    return router
