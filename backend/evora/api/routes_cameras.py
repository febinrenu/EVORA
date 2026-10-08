"""Camera routes: list, upload (files or rtsp uri), patch, frame, live tile."""
from __future__ import annotations

import asyncio
import base64
import json

from contracts.models import CameraInfo
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import UploadFile

from evora.api.context import AppContext
from evora.core import cameras as cams
from evora.core import media, perception_adapter

CHUNK = 1024 * 1024

# 1x1 JPEG standing in for real frames until the media service (P1.7) lands
PLACEHOLDER_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////"
    "////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)


def _register_file(ctx: AppContext, sink: media.UploadSink, sha: str) -> CameraInfo:
    try:
        probed = media.probe(sink.path)
    except media.UploadError:
        sink.discard()
        raise
    t0, source = perception_adapter.detect_clock(sink.path, probed)
    return cams.insert_camera(
        ctx.db, name=sink.path.stem.split("_", 1)[-1], kind="file", source_uri=str(sink.path),
        t0=t0, t0_source=source, sha256=sha, fps=probed.fps, width=probed.width, height=probed.height,
        rotation=probed.rotation, duration_s=probed.duration_s,
    )


async def _save_one(ctx: AppContext, upload: UploadFile) -> CameraInfo:
    up = ctx.cfg["uploads"]
    sink = media.UploadSink(ctx.ws.uploads_dir, upload.filename or "", up["max_bytes"], up["extensions"])
    try:
        while chunk := await upload.read(CHUNK):
            sink.write(chunk)
        path, sha = sink.finish()
    except media.UploadError:
        sink.discard()
        raise
    return await run_in_threadpool(_register_file, ctx, sink, sha)


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/cameras")

    def _require(cid: str) -> CameraInfo:
        try:
            return cams.get_camera(ctx.db, cid)
        except cams.CameraNotFound:
            raise HTTPException(404, "unknown camera") from None

    @router.get("")
    def list_cameras():
        return cams.list_cameras(ctx.db)

    @router.post("")
    async def add_cameras(request: Request):
        try:
            if request.headers.get("content-type", "").startswith("application/json"):
                body = await request.json()
                uri = media.validate_rtsp_uri(str(body.get("uri", "")))
                name = str(body.get("name") or "").strip() or uri.rsplit("/", 1)[-1] or "Live camera"
                import time

                cam = cams.insert_camera(ctx.db, name=name, kind="rtsp", source_uri=uri, t0=time.time(), t0_source="live")
                return [cam]
            form = await request.form()
            uploads = [v for v in form.getlist("files") if isinstance(v, UploadFile)]
        except media.UploadError as exc:
            raise HTTPException(exc.status, exc.message) from None
        if not uploads:
            raise HTTPException(422, "send one or more files, or a JSON body with a uri")
        ok: list[CameraInfo] = []
        rejected: list[dict[str, str]] = []
        for upload in uploads:
            try:
                ok.append(await _save_one(ctx, upload))
            except media.UploadError as exc:
                rejected.append({"filename": upload.filename or "", "error": exc.message, "status": str(exc.status)})
        if not ok:
            first = rejected[0]
            raise HTTPException(int(first["status"]), "; ".join(f"{r['filename']}: {r['error']}" for r in rejected))
        for cam in ok:
            ctx.bus.publish("camera", {"camera_id": cam.id, "status": cam.status})
        headers = {"X-Evora-Rejected": json.dumps(rejected)} if rejected else None
        return JSONResponse([c.model_dump(mode="json") for c in ok], headers=headers)

    @router.patch("/{cid}")
    def patch_camera(cid: str, body: dict):
        _require(cid)
        site = body.get("site_xy")
        if site is not None and not (isinstance(site, list) and len(site) == 2 and all(isinstance(v, int | float) for v in site)):
            raise HTTPException(422, "site_xy must be [x, y]")
        name = body.get("name")
        if name is not None and not str(name).strip():
            raise HTTPException(422, "name cannot be empty")
        t0 = body.get("t0")
        if t0 is not None and not isinstance(t0, int | float):
            raise HTTPException(422, "t0 must be epoch seconds")
        return cams.update_camera(ctx.db, cid, name=name, t0=t0, site_xy=tuple(site) if site else None)

    @router.get("/{cid}/frame")
    def frame(cid: str, t: float = Query(...)):
        _require(cid)
        return Response(PLACEHOLDER_JPEG, media_type="image/jpeg")

    @router.get("/{cid}/live.mjpg")
    async def live(cid: str):
        _require(cid)

        async def gen():
            for _ in range(3):
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + PLACEHOLDER_JPEG + b"\r\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return router
