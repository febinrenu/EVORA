"""FastAPI app for contract v1. Every route returns fixtures until its owner wires the real service."""
from __future__ import annotations

import asyncio
import base64
from typing import Any

from contracts.models import ClarifyResponse, MemoryFact, Zone
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from evora.api import fixtures
from evora.api.sse import heartbeat, stream_events
from evora.core.config import load_config

# 1x1 JPEG standing in for thumbnails and frames in the skeleton
_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////"
    "////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)


def create_app() -> FastAPI:
    cfg = load_config()
    app = FastAPI(title="evora", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg["server"]["cors_origins"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state: dict[str, Any] = {
        "settings": {"onprem": False, "blur_faces": True, "reference_now": None},
        "memory": [dict(f) for f in fixtures.load("memory_facts")],
        "workspaces": [{"slug": "own-campus", "name": "own-campus", "active": True}],
    }

    # --- health, workspaces ---
    @app.get("/api/health")
    def health():
        return {**fixtures.load("health"), "onprem": state["settings"]["onprem"]}

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

    # --- cameras ---
    def _cam(cid: str) -> dict:
        for c in fixtures.load("cameras"):
            if c["id"] == cid:
                return c
        raise HTTPException(404, "unknown camera")

    @app.get("/api/cameras")
    def cameras():
        return fixtures.load("cameras")

    @app.post("/api/cameras")
    async def add_cameras(request: Request, files: list[UploadFile] | None = File(default=None)):
        if files is None and "json" in request.headers.get("content-type", ""):
            body = await request.json()
            if not body.get("uri"):
                raise HTTPException(422, "uri required")
        return [{**c, "status": "pending", "layers": []} for c in fixtures.load("cameras")[:1]]

    @app.patch("/api/cameras/{cid}")
    def patch_camera(cid: str, body: dict):
        return {**_cam(cid), **{k: v for k, v in body.items() if k in {"name", "t0", "site_xy"}}}

    @app.get("/api/cameras/{cid}/frame")
    def frame(cid: str, t: float = Query(...)):
        _cam(cid)
        return Response(_JPEG, media_type="image/jpeg")

    @app.get("/api/cameras/{cid}/live.mjpg")
    async def live(cid: str):
        _cam(cid)

        async def gen():
            for _ in range(3):
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _JPEG + b"\r\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/ingest")
    def ingest(body: dict):
        return fixtures.load("ingest_jobs")

    # --- query, clarify ---
    @app.post("/api/query")
    def query(body: dict):
        if not str(body.get("text", "")).strip():
            raise HTTPException(422, "text required")
        name = "stream_clarify" if "back entrance" in body["text"].lower() else "stream_query"
        return stream_events(fixtures.load(name))

    @app.post("/api/clarify")
    def clarify(resp: ClarifyResponse):
        return stream_events(fixtures.load("stream_query"))

    # --- memory ---
    @app.get("/api/memory")
    def memory():
        return state["memory"]

    @app.post("/api/memory")
    def add_memory(fact: MemoryFact):
        state["memory"].append(fact.model_dump())
        return fact

    def _fact(fid: str) -> dict:
        for f in state["memory"]:
            if f["id"] == fid:
                return f
        raise HTTPException(404, "unknown fact")

    @app.patch("/api/memory/{fid}")
    def patch_memory(fid: str, body: dict):
        f = _fact(fid)
        f.update({k: v for k, v in body.items() if k in {"canonical", "aliases", "binding"}})
        return f

    @app.delete("/api/memory/{fid}")
    def delete_memory(fid: str):
        state["memory"].remove(_fact(fid))
        return state["memory"]

    # --- zones, tracks, globals ---
    @app.get("/api/zones")
    def zones(camera_id: str | None = None):
        z = fixtures.load("zone")
        return [z] if camera_id in (None, z["camera_id"]) else []

    @app.post("/api/zones")
    def add_zone(zone: Zone):
        return zone

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

    # --- media, evidence ---
    @app.get("/api/media/thumb/{evidence_id}.jpg")
    def thumb(evidence_id: str):
        return Response(_JPEG, media_type="image/jpeg")

    @app.get("/api/media/clip/{evidence_id}.mp4")
    def clip(evidence_id: str):
        return Response(b"", media_type="video/mp4")

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

    @app.get("/api/events")
    async def events():
        return EventSourceResponse(heartbeat(5.0))

    # --- settings, voice, report, dev ---
    @app.post("/api/settings")
    def settings(body: dict):
        state["settings"].update({k: v for k, v in body.items() if k in state["settings"]})
        return state["settings"]

    @app.post("/api/voice")
    async def voice(request: Request):
        await request.body()
        return {"text": "person in red at the main gate"}

    @app.get("/api/report")
    def report():
        return {"eval": None, "ablations": None}

    @app.post("/api/dev/gt")
    def dev_gt(body: dict):
        return {"ok": True, "item": body}

    return app


app = create_app()
