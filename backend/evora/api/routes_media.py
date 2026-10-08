"""Media routes: camera frames, evidence thumbnails and clips (range requests), audited unblur tokens."""
from __future__ import annotations

from contracts.models import CameraInfo
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from evora.api.context import AppContext
from evora.core import cameras as cams
from evora.core.media_service import MediaError
from evora.evidence import audit, store


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api")

    def _want_blur(token: str | None, what: dict) -> bool:
        """Blur unless it is switched off or the viewer holds a valid unblur token (audited)."""
        if not ctx.settings["blur_faces"]:
            return False
        if ctx.unblur.valid(token):
            audit.record(ctx.db, "unblur_view", what)
            return False
        return True

    def _camera(cid: str) -> CameraInfo:
        try:
            return cams.get_camera(ctx.db, cid)
        except cams.CameraNotFound:
            raise HTTPException(404, "unknown camera") from None

    def _evidence(evidence_id: str) -> tuple[store.EvidenceRecord, CameraInfo]:
        try:
            rec = store.get(ctx.db, evidence_id)
        except store.EvidenceError:
            raise HTTPException(422, "invalid evidence id") from None
        except store.EvidenceNotFound:
            raise HTTPException(404, "unknown evidence") from None
        return rec, _camera(rec.camera_id)

    @router.get("/cameras/{cid}/frame")
    async def frame(cid: str, t: float = Query(...), unblur: str | None = None):
        cam = _camera(cid)
        blur = _want_blur(unblur, {"camera_id": cid, "t": t})
        try:
            data, status = await run_in_threadpool(ctx.media.frame, cam, t, blur)
        except MediaError as exc:
            raise HTTPException(exc.status, exc.message) from None
        return Response(data, media_type="image/jpeg", headers={"X-Evora-Blur": status})

    @router.get("/media/thumb/{evidence_id}.jpg")
    async def thumb(evidence_id: str, unblur: str | None = None):
        rec, cam = _evidence(evidence_id)
        blur = _want_blur(unblur, {"evidence_id": evidence_id, "kind": "thumb"})
        try:
            path, status = await run_in_threadpool(ctx.media.thumb, cam, rec, blur)
        except MediaError as exc:
            raise HTTPException(exc.status, exc.message) from None
        return FileResponse(path, media_type="image/jpeg", headers={"X-Evora-Blur": status})

    @router.get("/media/clip/{evidence_id}.mp4")
    async def clip(evidence_id: str, unblur: str | None = None):
        rec, cam = _evidence(evidence_id)
        blur = _want_blur(unblur, {"evidence_id": evidence_id, "kind": "clip"})
        try:
            path, status = await run_in_threadpool(ctx.media.clip, cam, rec, blur)
        except MediaError as exc:
            raise HTTPException(exc.status, exc.message) from None
        return FileResponse(path, media_type="video/mp4", headers={"X-Evora-Blur": status})

    @router.post("/media/unblur")
    def unblur_token(body: dict):
        reason = str(body.get("reason", "")).strip()
        if not reason or len(reason) > 200:
            raise HTTPException(422, "a reason of 1 to 200 characters is required")
        token, expires = ctx.unblur.issue(reason)
        audit.record(ctx.db, "unblur_token", {"reason": reason, "evidence_id": body.get("evidence_id")})
        return {"token": token, "expires_at": expires}

    return router
