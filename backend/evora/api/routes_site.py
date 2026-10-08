"""The site plan picture of a workspace (a floor plan under the camera nodes): one image, kept with the workspace."""
from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from evora.api.context import AppContext

MAX_BYTES = 5 * 1024 * 1024
TYPES = {"JPEG": ("image/jpeg", ".jpg"), "PNG": ("image/png", ".png"), "WEBP": ("image/webp", ".webp")}


def _stored(root: Path) -> Path | None:
    return next((p for p in sorted(root.glob("site_plan.*")) if p.suffix in {".jpg", ".png", ".webp"}), None)


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/site")
    root = ctx.ws.root

    @router.get("/plan")
    def get_plan():
        path = _stored(root)
        if path is None:
            raise HTTPException(404, "no site plan picture for this workspace")
        media = next(m for m, ext in TYPES.values() if ext == path.suffix)
        return FileResponse(path, media_type=media, headers={"Cache-Control": "no-cache"})

    @router.put("/plan")
    async def put_plan(request: Request):
        data = await request.body()
        if not data:
            raise HTTPException(422, "send the picture as the request body")
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"the picture is larger than {MAX_BYTES // (1024 * 1024)} MB; downscale it first")
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(io.BytesIO(data)) as img:
                kind, size = img.format, img.size
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(415, "not a readable JPEG, PNG or WebP picture") from None
        if kind not in TYPES:
            raise HTTPException(415, f"{kind} pictures are not accepted; use JPEG, PNG or WebP")
        media, ext = TYPES[kind]
        old = _stored(root)
        tmp = root / f"site_plan{ext}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, root / f"site_plan{ext}")
        if old is not None and old.suffix != ext:
            old.unlink(missing_ok=True)
        ctx.bus.publish("site", {"plan": True})
        return {"ok": True, "type": media, "bytes": len(data), "width": size[0], "height": size[1]}

    @router.delete("/plan")
    def delete_plan():
        path = _stored(root)
        if path is not None:
            path.unlink(missing_ok=True)
            ctx.bus.publish("site", {"plan": False})
        return Response(status_code=204)

    return router
