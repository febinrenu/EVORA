"""Evidence pack export. `evora_MOCK=1` keeps the fixture bytes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from evora.api.context import AppContext
from evora.core import cameras as cams
from evora.core.media_service import MediaError
from evora.evidence import pack, signing, store


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/evidence")

    @router.get("/signer")
    def signer() -> dict:
        """The key that signs this workspace's evidence packs: publish the fingerprint so recipients can check it."""
        if ctx.mock:
            return {"algorithm": signing.ALGORITHM, "fingerprint": "0123456789abcdef",
                    "public_key_pem": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----\n"}
        try:
            s = signing.load_or_create(ctx.ws.root)
        except signing.SigningError as exc:
            raise HTTPException(500, str(exc)) from None
        return {"algorithm": signing.ALGORITHM, "fingerprint": s.fingerprint, "public_key_pem": s.public_pem.decode("ascii")}

    @router.post("/{evidence_id}/pack")
    async def export_pack(evidence_id: str, unblur: str | None = None):
        if ctx.mock:
            return Response(b"PK\x05\x06" + b"\x00" * 18, media_type="application/zip")
        try:
            result = await run_in_threadpool(
                pack.build_pack, ctx.db, ctx.ws, ctx.media, evidence_id,
                blur_setting=bool(ctx.settings["blur_faces"]), unblur_token_valid=ctx.unblur.valid(unblur),
                unblur_reason=ctx.unblur.reason_of(unblur),
            )
        except store.EvidenceError:
            raise HTTPException(422, "invalid evidence id") from None
        except (store.EvidenceNotFound, cams.CameraNotFound):
            raise HTTPException(404, "unknown evidence") from None
        except pack.PackRefused as exc:
            raise HTTPException(409, exc.message) from None
        except signing.SigningError as exc:
            raise HTTPException(500, str(exc)) from None
        except MediaError as exc:
            raise HTTPException(exc.status, exc.message) from None
        return FileResponse(
            result.path, media_type="application/zip", filename=f"evora_evidence_{evidence_id}.zip",
            headers={"X-Evora-Blur": "applied" if result.faces_blurred else "off", "X-Evora-Pack-SHA256": result.sha256},
        )

    return router
