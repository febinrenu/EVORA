"""Replay-as-live, live analysis and live tiles. `evora_MOCK=1` returns canned answers."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from evora.api.context import AppContext
from evora.core import cameras as cams
from evora.live import mjpeg
from evora.live.restream import LiveError


class ReplayRequest(BaseModel):
    camera_ids: list[str] = Field(min_length=1, max_length=64)
    speed: float | None = None
    analyze: bool = False  # also run live analysis on the replayed cameras (events, alerts)


class StopRequest(BaseModel):
    camera_ids: list[str] | None = None


class AnalyzeRequest(BaseModel):
    camera_ids: list[str] = Field(min_length=1, max_length=64)


CANNED = {
    "server": {"ready": True, "port": 8554, "binary_found": True},
    "streams": [
        {"camera_id": "cam_01", "url": "rtsp://127.0.0.1:8554/cam_01", "state": "running", "speed": 1.0, "restarts": 0,
         "transcoding": False, "started_at": 1790000000.0, "error": None},
    ],
    "analyzers": [{"camera_id": "cam_01", "state": "running", "error": None}],
}


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/live")

    def _status() -> dict:
        return {**ctx.live.status(), **ctx.live_runner.status()}

    @router.get("")
    def status():
        return CANNED if ctx.mock else _status()

    @router.post("/replay")
    async def start(body: ReplayRequest):
        if ctx.mock:
            return CANNED
        try:
            await run_in_threadpool(ctx.live.start, body.camera_ids, body.speed)
            if body.analyze:
                await run_in_threadpool(ctx.live_runner.start, body.camera_ids)
        except LiveError as exc:
            raise HTTPException(exc.status, exc.message) from None
        return _status()

    @router.post("/replay/stop")
    async def stop(body: StopRequest | None = None):
        ids = body.camera_ids if body else None
        if ctx.mock:
            return {**CANNED, "streams": [], "analyzers": []}
        await run_in_threadpool(ctx.live_runner.stop, ids)  # stop analysing before the stream disappears
        await run_in_threadpool(ctx.live.stop, ids)
        return _status()

    @router.post("/analyze")
    async def analyze(body: AnalyzeRequest):
        if ctx.mock:
            return CANNED
        try:
            await run_in_threadpool(ctx.live_runner.start, body.camera_ids)
        except LiveError as exc:
            raise HTTPException(exc.status, exc.message) from None
        return _status()

    @router.post("/analyze/stop")
    async def analyze_stop(body: StopRequest | None = None):
        if ctx.mock:
            return {**CANNED, "analyzers": []}
        await run_in_threadpool(ctx.live_runner.stop, body.camera_ids if body else None)
        return _status()

    return router


def make_tile_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/cameras")

    @router.get("/{cid}/live.mjpg")
    async def tile(cid: str, fps: int = Query(2, ge=1, le=mjpeg.MAX_FPS), unblur: str | None = None):
        try:
            cams.get_camera(ctx.db, cid)
        except cams.CameraNotFound:
            raise HTTPException(404, "unknown camera") from None
        if ctx.mock:
            raise HTTPException(409, "live tiles are not available in mock mode")
        try:
            url = ctx.live_runner.stream_url(cid)
        except LiveError as exc:
            raise HTTPException(exc.status, exc.message) from None
        want_blur = bool(ctx.settings["blur_faces"]) and not ctx.unblur.valid(unblur)
        blur, status = ctx.media.blur_state(want_blur)
        if not ctx.tiles.acquire():
            raise HTTPException(503, "too many live tiles are open; close one and try again")

        released = {"done": False}

        def release() -> None:  # idempotent: the generator's finally and the response cleanup may both run
            if not released["done"]:
                released["done"] = True
                ctx.tiles.release()

        async def frames():
            try:
                async for chunk in mjpeg.stream_frames(url, fps, blur, ctx.media._blur):
                    yield chunk
            finally:
                release()

        return StreamingResponse(
            frames(), media_type=f"multipart/x-mixed-replace; boundary={mjpeg.BOUNDARY}",
            headers={"X-Evora-Blur": status, "Cache-Control": "no-store"}, background=BackgroundTask(release),
        )

    return router
