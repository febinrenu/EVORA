"""Ingest and event-stream routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from evora.api.context import AppContext
from evora.core import cameras as cams


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/ingest")
    def ingest(body: dict):
        ids = body.get("camera_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            raise HTTPException(422, "camera_ids must be a non-empty list")
        layers = body.get("layers")
        if layers is not None and not isinstance(layers, list):
            raise HTTPException(422, "layers must be a list")
        try:
            return ctx.runner.submit(ids, layers)
        except cams.CameraNotFound as exc:
            raise HTTPException(404, f"unknown camera: {exc.args[0]}") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @router.get("/events")
    async def events():
        return EventSourceResponse(ctx.bus.stream())

    return router
