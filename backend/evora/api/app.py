"""FastAPI app for contract v1. Routes whose owner has not wired a service yet still return fixtures."""
from __future__ import annotations

import asyncio
import base64
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from evora import __version__
from evora.api import (
    routes_admin,
    routes_alerts,
    routes_cameras,
    routes_evidence,
    routes_ingest,
    routes_live,
    routes_media,
    routes_memory,
    routes_query,
    routes_tracks,
    routes_zones,
)
from evora.api.context import AppContext
from evora.api.ui import mount_ui
from evora.core import cameras as cams
from evora.core import perception_adapter
from evora.core import settings as app_settings
from evora.core.config import REPO_ROOT, load_config
from evora.core.jobs import IngestFn
from evora.core.media_service import BlurFn
from evora.core.privacy_guard import guard
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
    equivalence: Equivalence | None = None, gateway: Any = None, mock: bool | None = None, ui_dir: Path | None = None,
) -> FastAPI:
    cfg = load_config()
    ctx = AppContext.build(cfg, workspaces_root, ingest_fn, blur_provider, embedder, equivalence, gateway, mock)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ctx.notifier.loop = asyncio.get_running_loop()
        perception_adapter.register_vision(ctx.gateway, ctx.notifier.loop)
        yield
        perception_adapter.unregister_vision()
        ctx.runner.shutdown(wait=False)
        ctx.live_runner.shutdown()
        ctx.live.shutdown()
        ctx.prerender.shutdown()
        if ctx.http is not None:
            await ctx.http.aclose()

    app = FastAPI(title="evora", version=__version__, lifespan=lifespan)
    app.state.ctx = ctx
    app.include_router(routes_cameras.make_router(ctx))
    app.include_router(routes_ingest.make_router(ctx))
    app.include_router(routes_media.make_router(ctx))
    app.include_router(routes_memory.make_router(ctx))
    app.include_router(routes_query.make_router(ctx))
    app.include_router(routes_zones.make_router(ctx))
    app.include_router(routes_alerts.make_router(ctx))
    app.include_router(routes_evidence.make_router(ctx))
    app.include_router(routes_live.make_router(ctx))
    app.include_router(routes_live.make_tile_router(ctx))
    app.include_router(routes_tracks.make_router(ctx))
    app.include_router(routes_admin.make_router(ctx))
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
        usable = [c for c in cams.list_cameras(ctx.db) if c.status != "error"]
        ready = sorted(set.intersection(*(set(c.layers) for c in usable))) if usable else []
        return {
            "ok": True, "version": __version__, "workspace": ctx.ws.slug, "profile": os.environ.get("evora_PROFILE", "cpu"),
            "onprem": bool(state["settings"]["onprem"]), "layers_ready": ready,
            "egress_blocked": guard.blocked, "blur": ctx.media.blur_status(bool(state["settings"]["blur_faces"])),
        }

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
    # --- settings, voice, report, dev ---
    @app.post("/api/settings")
    def settings(body: dict):
        try:
            changes = app_settings.validate(body)
        except app_settings.SettingsError as exc:
            raise HTTPException(422, str(exc)) from None
        current = state["settings"]
        if "blur_faces" in changes and changes["blur_faces"] != current["blur_faces"]:
            audit.record(ctx.db, "blur_setting", {"blur_faces": changes["blur_faces"]})
        if "onprem" in changes and changes["onprem"] != current["onprem"]:
            audit.record(ctx.db, "onprem_setting", {"onprem": changes["onprem"]})
            ctx.bus.publish("privacy", {"onprem": changes["onprem"]})
        current.update(changes)
        app_settings.save(ctx.db, changes)
        return current

    @app.get("/api/report")
    def report():
        return {"eval": None, "ablations": None}

    @app.post("/api/dev/gt")
    def dev_gt(body: dict):
        return {"ok": True, "item": body}

    if not ctx.mock and cfg["server"].get("serve_ui", True):
        mount_ui(app, ui_dir or REPO_ROOT / cfg["server"].get("ui_dir", "frontend/dist"))  # last, so every API route wins
    return app

