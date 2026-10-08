"""Camera routes: list, upload (files or rtsp uri), patch, frame, live tile."""
from __future__ import annotations

import base64
import json

from contracts.models import CameraInfo
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from evora.api.context import AppContext
from evora.core import cameras as cams
from evora.core import clock_shift, media, perception_adapter
from evora.evidence import audit

CHUNK = 1024 * 1024

# 1x1 JPEG standing in for real frames until the media service (P1.7) lands
PLACEHOLDER_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////"
    "////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)


def _register_file(ctx: AppContext, sink: media.UploadSink, sha: str) -> CameraInfo:
    up = ctx.cfg["uploads"]
    try:
        probed = media.probe(sink.path)
        path = sink.path
        if probed.codec not in up["native_codecs"]:
            ctx.bus.publish("note", {"message": f"converting {sink.name} ({probed.codec}) to H.264"})
            path = sink.path.with_suffix(".h264.mp4")
            media.transcode_to_h264(sink.path, path, up["transcode_timeout_s"])
            converted = media.probe(path)
            audit.record(ctx.db, "transcode", {"file": sink.name, "from": probed.codec, "original_sha256": sha})
            t0, source = perception_adapter.detect_clock(sink.path, probed, quick=True)  # metadata lives on the original
            sink.path.unlink(missing_ok=True)
            probed = converted
        else:
            t0, source = perception_adapter.detect_clock(path, probed, quick=True)
    except media.UploadError:
        sink.discard()
        raise
    cam = cams.insert_camera(
        ctx.db, name=sink.name.rsplit(".", 1)[0], kind="file", source_uri=str(path),
        t0=t0, t0_source=source, sha256=sha, fps=probed.fps, width=probed.width, height=probed.height,
        rotation=probed.rotation, duration_s=probed.duration_s,
    )
    if source == "manual" and ctx.clock is not None:  # only the file time: read the on-screen clock in the background
        ctx.clock.submit(cam.id, path)
    return cam


async def _save_one(ctx: AppContext, upload: UploadFile) -> tuple[CameraInfo, bool]:
    """Returns (camera, was_duplicate). A byte-identical upload reuses the camera it already created."""
    up = ctx.cfg["uploads"]
    sink = media.UploadSink(ctx.ws.uploads_dir, upload.filename or "", up["max_bytes"], up["extensions"])
    try:
        while chunk := await upload.read(CHUNK):
            sink.write(chunk)
        path, sha = sink.finish()
    except media.UploadError:
        sink.discard()
        raise
    existing = await run_in_threadpool(cams.find_by_sha, ctx.db, sha)
    if existing is not None:
        sink.discard()
        if existing.status == "error":  # uploading a failed file again means "try again": it can be indexed once more
            await run_in_threadpool(cams.set_status, ctx.db, existing.id, "pending")
            existing = existing.model_copy(update={"status": "pending"})
        return existing, True
    return await run_in_threadpool(_register_file, ctx, sink, sha), False


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
        duplicates: list[str] = []
        rejected: list[dict[str, str]] = []
        for upload in uploads:
            try:
                cam, was_duplicate = await _save_one(ctx, upload)
                ok.append(cam)
                if was_duplicate:
                    duplicates.append(cam.id)
            except media.UploadError as exc:
                rejected.append({"filename": upload.filename or "", "error": exc.message, "status": str(exc.status)})
        if not ok:
            first = rejected[0]
            raise HTTPException(int(first["status"]), "; ".join(f"{r['filename']}: {r['error']}" for r in rejected))
        for cam in ok:
            ctx.bus.publish("camera", {"camera_id": cam.id, "status": cam.status})
        headers = {}
        if rejected:
            headers["X-Evora-Rejected"] = json.dumps(rejected)
        if duplicates:
            headers["X-Evora-Duplicate"] = json.dumps(duplicates)
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
        if t0 is not None and (isinstance(t0, bool) or not isinstance(t0, int | float)):
            raise HTTPException(422, "t0 must be epoch seconds")
        if t0 is not None:
            try:  # everything already indexed moves with the clock, so answers and clips stay right
                clock_shift.set_camera_clock(ctx.db, ctx.ws.vectors_dir, cid, float(t0), source="manual")
            except clock_shift.ClockBusy as exc:
                raise HTTPException(409, str(exc)) from None
            ctx.bus.publish("camera", {"camera_id": cid, "status": cams.get_camera(ctx.db, cid).status, "clock": "manual"})
        return cams.update_camera(ctx.db, cid, name=name, site_xy=tuple(site) if site else None)

    return router
