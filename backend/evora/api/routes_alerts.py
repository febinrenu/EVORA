"""Standing queries (watches) and alerts. `evora_MOCK=1` keeps the fixture answers."""
from __future__ import annotations

from contracts.models import Alert, StandingQuery
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from evora.alerts import store
from evora.alerts.compiler import CompileError, NeedsClarification
from evora.api import fixtures
from evora.api.context import AppContext
from evora.evidence import audit

MAX_TEXT = 500


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/standing")
    async def create_standing(body: dict):
        text = str(body.get("text", "")).strip()
        if ctx.mock:
            return {**fixtures.load("standing_query"), "text": text}
        if not text:
            raise HTTPException(422, "text required")
        if len(text) > MAX_TEXT:
            raise HTTPException(422, f"watches are limited to {MAX_TEXT} characters")
        try:
            result = await ctx.compiler.compile(text)
        except CompileError as exc:
            raise HTTPException(422, str(exc)) from None
        if isinstance(result, NeedsClarification):  # answer it through /api/clarify, then post the same text again
            return JSONResponse({"clarify": result.request.model_dump(mode="json")}, status_code=409)
        sq = await run_in_threadpool(store.create_standing, ctx.db, text, result.rule)
        ctx.alerts.invalidate()
        await run_in_threadpool(audit.record, ctx.db, "standing_create", {"id": sq.id, "text": text[:200]})
        await run_in_threadpool(ctx.alerts.backfill, None, sq.id)  # a new watch reports what already happened once
        return sq

    @router.get("/standing")
    def list_standing() -> list[StandingQuery]:
        if ctx.mock:
            return [StandingQuery.model_validate(fixtures.load("standing_query"))]
        return store.list_standing(ctx.db)

    @router.patch("/standing/{sq_id}")
    async def patch_standing(sq_id: str, body: dict):
        if ctx.mock:
            return {**fixtures.load("standing_query"), "id": sq_id, "active": bool(body.get("active", True))}
        if "active" not in body or not isinstance(body["active"], bool):
            raise HTTPException(422, "active must be true or false")
        try:
            sq = await run_in_threadpool(store.set_active, ctx.db, sq_id, body["active"])
        except store.StandingNotFound:
            raise HTTPException(404, "unknown watch") from None
        ctx.alerts.invalidate()
        await run_in_threadpool(audit.record, ctx.db, "standing_toggle", {"id": sq_id, "active": body["active"]})
        if sq.active:
            await run_in_threadpool(ctx.alerts.backfill, None, sq_id)
        return sq

    @router.get("/alerts")
    def list_alerts(acknowledged: bool | None = None) -> list[Alert]:
        if ctx.mock:
            return [Alert.model_validate(fixtures.load("alert"))]
        return store.list_alerts(ctx.db, acknowledged)

    @router.post("/alerts/{alert_id}/ack")
    def ack(alert_id: str) -> Alert:
        if ctx.mock:
            return Alert.model_validate({**fixtures.load("alert"), "id": alert_id, "acknowledged": True})
        try:
            return store.acknowledge(ctx.db, alert_id)
        except store.AlertNotFound:
            raise HTTPException(404, "unknown alert") from None

    return router
