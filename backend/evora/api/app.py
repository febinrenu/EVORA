"""FastAPI app for contract v1. Routes whose owner has not wired a service yet still return fixtures."""
from __future__ import annotations

import base64
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from evora.api import fixtures, routes_cameras, routes_ingest, routes_media, routes_memory, routes_query, routes_zones
from evora.api.context import AppContext
from evora.core.config import load_config
from evora.core.jobs import IngestFn
from evora.core.media_service import BlurFn
from evora.evidence import audit
from evora.memory.embedder import TextEmbedder
from evora.memory.resolve import Equivalence

# 1x1 JPEG standing in for thumbnails and frames in the skeleton
_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////"
    "////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)


def create_app(
    workspaces_root: Path | None = None, ingest_fn: IngestFn | None = None,
    blur_provider: Callable[[], BlurFn | None] | None = None, embedder: TextEmbedder | None = None,
    equivalence: Equivalence | None = None, gateway: Any = None, mock: bool | None = None,
) -> FastAPI:
    cfg = load_config()
    ctx = AppContext.build(cfg, workspaces_root, ingest_fn, blur_provider, embedder, equivalence, gateway, mock)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        ctx.runner.shutdown(wait=False)
        ctx.prerender.shutdown()
        if ctx.http is not None:
            await ctx.http.aclose()

    app = FastAPI(title="evora", version="0.1.0", lifespan=lifespan)
    app.state.ctx = ctx
    app.include_router(routes_cameras.make_router(ctx))
    app.include_router(routes_ingest.make_router(ctx))
    app.include_router(routes_media.make_router(ctx))
    app.include_router(routes_memory.make_router(ctx))
    app.include_router(routes_query.make_router(ctx))
    app.include_router(routes_zones.make_router(ctx))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg["server"]["cors_origins"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state: dict[str, Any] = {
        "settings": ctx.settings,
        "workspaces": [{"slug": ctx.ws.slug, "name": ctx.ws.slug, "active": True}],
    }

    # --- health, workspaces ---
    @app.get("/api/health")
    def health():
        return {**fixtures.load("health"), "workspace": ctx.ws.slug, "onprem": state["settings"]["onprem"]}

    @app.get("/api/workspaces")
    def workspaces():
        return state["workspaces"]

    @app.post("/api/workspaces")
    def create_workspace(body: dict):
        name = str(body.get("name", "")).strip()
        if not name:
            raise HTTPException(422, "name required")
        ws = {"slug": name.lower().replace(" ", "-"), "name": name, "active": False}
        state["workspaces"].append(ws)
        return ws

    @app.post("/api/workspaces/{slug}/activate")
    def activate_workspace(slug: str):
        if slug not in {w["slug"] for w in state["workspaces"]}:
            raise HTTPException(404, "unknown workspace")
        for w in state["workspaces"]:
            w["active"] = w["slug"] == slug
        return state["workspaces"]

    # --- zones, tracks, globals ---
    @app.get("/api/tracks/{track_id}")
    def track(track_id: str):
        ev = fixtures.load("answer")["evidence"][0]
        return {
            "id": track_id, "camera_id": ev["camera_id"], "cls": "person",
            "t_start": ev["t_start"], "t_end": ev["t_end"],
            "attrs": {"color": "red", "carrying": ["backpack"]}, "points": [],
        }

    @app.get("/api/tracks/{track_id}/similar")
    def similar(track_id: str, k: int = 20):
        return fixtures.load("answer")["evidence"][:k]

    @app.get("/api/globals/{gid}/path")
    def path(gid: str):
        return fixtures.load("path")

    # --- evidence ---
    @app.post("/api/evidence/{evidence_id}/pack")
    def pack(evidence_id: str):
        return Response(b"PK\x05\x06" + b"\x00" * 18, media_type="application/zip")

    # --- standing queries, alerts ---
    @app.post("/api/standing")
    def add_standing(body: dict):
        return {**fixtures.load("standing_query"), "text": body.get("text", "")}

    @app.get("/api/standing")
    def standing():
        return [fixtures.load("standing_query")]

    @app.patch("/api/standing/{sid}")
    def patch_standing(sid: str, body: dict):
        return {**fixtures.load("standing_query"), "id": sid, **{k: v for k, v in body.items() if k == "active"}}

    @app.get("/api/alerts")
    def alerts():
        return [fixtures.load("alert")]

    @app.post("/api/alerts/{aid}/ack")
    def ack(aid: str):
        return {**fixtures.load("alert"), "id": aid, "acknowledged": True}

    # --- settings, voice, report, dev ---
    @app.post("/api/settings")
    def settings(body: dict):
        changes = {k: v for k, v in body.items() if k in state["settings"]}
        if "blur_faces" in changes and changes["blur_faces"] != state["settings"]["blur_faces"]:
            audit.record(ctx.db, "blur_setting", {"blur_faces": bool(changes["blur_faces"])})
        state["settings"].update(changes)
        return state["settings"]

    @app.get("/api/report")
    def report():
        return {"eval": None, "ablations": None}

    @app.post("/api/dev/gt")
    def dev_gt(body: dict):
        return {"ok": True, "item": body}

    return app

