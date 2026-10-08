"""Tracks, query by example and cross-camera paths. `evora_MOCK=1` keeps the fixtures."""
from __future__ import annotations

import json
import re
from datetime import datetime

from contracts.models import Evidence, PathHop
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool

from evora.api import fixtures
from evora.api.context import AppContext
from evora.core import perception_adapter
from evora.evidence import store
from evora.evidence.builder import TrackNotFound, evidence_for_track
from evora.query.planner import workspace_tz

_TRACK_ID = re.compile(r"^[A-Za-z0-9_]{1,40}:t\d{1,10}$")
_GLOBAL_ID = re.compile(r"^[A-Za-z0-9_.-]{1,60}$")


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api")

    def _check_track_id(track_id: str) -> None:
        if not _TRACK_ID.match(track_id):
            raise HTTPException(422, "invalid track id")

    @router.get("/tracks/{track_id}")
    def track(track_id: str):
        if ctx.mock:
            ev = fixtures.load("answer")["evidence"][0]
            return {"id": track_id, "camera_id": ev["camera_id"], "cls": "person", "t_start": ev["t_start"],
                    "t_end": ev["t_end"], "attrs": {"color": "red", "carrying": ["backpack"]}, "points": []}
        _check_track_id(track_id)
        with ctx.db.read() as c:
            row = c.execute(
                "SELECT t.*, c.name AS camera_name FROM tracks t JOIN cameras c ON c.id = t.camera_id WHERE t.id=?",
                (track_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(404, "unknown track")
            points = [
                {"t": p["t"], "bbox": [p["x1"], p["y1"], p["x2"], p["y2"]], "conf": p["conf"]}
                for p in c.execute("SELECT t,x1,y1,x2,y2,conf FROM track_points WHERE track_id=? ORDER BY t", (track_id,))
            ]
        return {
            "id": row["id"], "camera_id": row["camera_id"], "camera_name": row["camera_name"], "cls": row["cls"],
            "cls_conf": row["cls_conf"], "t_start": row["t_start"], "t_end": row["t_end"], "n_obs": row["n_obs"],
            "attrs": json.loads(row["attrs"] or "{}"), "direction": row["direction"], "global_id": row["global_id"],
            "quality": row["quality"], "points": points,
        }

    @router.get("/tracks/{track_id}/similar")
    async def similar(track_id: str, response: Response, k: int = Query(20, ge=1, le=100)) -> list[Evidence]:
        if ctx.mock:
            return [Evidence.model_validate(e) for e in fixtures.load("answer")["evidence"][:k]]
        _check_track_id(track_id)
        hits = await run_in_threadpool(
            perception_adapter.similar_tracks, track_id, k, ctx.ws, ctx.memory.kb.store
        )
        if hits is None:
            response.headers["X-Evora-Reid"] = "pending"  # re-identification is not installed or has no vectors yet
            return []
        tz = workspace_tz(ctx.db)
        out: list[Evidence] = []
        for other_id, score in hits:
            try:
                ev = await run_in_threadpool(_similar_evidence, ctx, other_id, score, tz)
            except TrackNotFound:
                continue
            out.append(ev)
        return out

    @router.get("/globals/{gid}/path")
    async def path(gid: str, response: Response) -> list[PathHop]:
        if ctx.mock:
            return [PathHop.model_validate(h) for h in fixtures.load("path")]
        if not _GLOBAL_ID.match(gid):
            raise HTTPException(422, "invalid identity id")
        with ctx.db.read() as c:
            known = c.execute("SELECT 1 FROM tracks WHERE global_id=? LIMIT 1", (gid,)).fetchone()
        if known is None:
            raise HTTPException(404, "unknown identity")
        hops = await run_in_threadpool(perception_adapter.path_for, gid, ctx.ws, ctx.db)
        if hops is None:
            response.headers["X-Evora-Reid"] = "pending"
            return []
        for hop in hops:  # every hop must be playable: register the evidence the hop points at
            with ctx.db.read() as c:
                row = c.execute("SELECT id FROM tracks WHERE replace(id, ':', '_') = ?", (hop.evidence_id,)).fetchone()
            if row is not None and not _registered(ctx, hop.evidence_id):
                await run_in_threadpool(evidence_for_track, ctx.db, row["id"], evidence_id=hop.evidence_id)
        return hops

    return router


def _registered(ctx: AppContext, evidence_id: str) -> bool:
    try:
        store.get(ctx.db, evidence_id)
    except (store.EvidenceNotFound, store.EvidenceError):
        return False
    return True


def _similar_evidence(ctx: AppContext, track_id: str, score: float, tz) -> Evidence:  # noqa: ANN001
    ev = evidence_for_track(ctx.db, track_id, score=round(float(score), 3))
    when = datetime.fromtimestamp(ev.t_peak, tz).strftime("%H:%M:%S")
    ev.why = [f"looks like the query: {ev.camera_name} at {when}, similarity {score:.2f}"]
    store.register(ctx.db, ev)
    return ev
