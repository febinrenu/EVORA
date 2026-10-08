"""Query, clarify and voice routes. `evora_MOCK=1` keeps the fixture streams for UI development."""
from __future__ import annotations

from contracts.models import ClarifyResponse, StreamEvent
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from evora.api import fixtures
from evora.api.context import AppContext
from evora.api.sse import relay, stream_async, stream_events
from evora.evidence import audit
from evora.llm.schemas import LLMError
from evora.memory.clarify import ClarifyError, QuestionClosed

MAX_QUERY_CHARS = 2000
MAX_VOICE_BYTES = 10 * 1024 * 1024


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api")

    def _after_event(ev: StreamEvent) -> None:
        """Warm the media cache for the best evidence as soon as the answer exists."""
        if ev.type == "answer":
            ids = [e["id"] for e in ev.data.get("evidence", []) if isinstance(e, dict) and "id" in e]
            ctx.prerender.schedule(ids)

    @router.post("/query")
    async def query(body: dict):
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(422, "text required")
        if len(text) > MAX_QUERY_CHARS:
            raise HTTPException(422, f"questions are limited to {MAX_QUERY_CHARS} characters")
        session_id = str(body.get("session_id") or "default")[:64]
        if ctx.mock:
            name = "stream_clarify" if "back entrance" in text.lower() else "stream_query"
            return stream_events(fixtures.load(name))
        await run_in_threadpool(audit.record, ctx.db, "query", {"text": text[:200], "session_id": session_id})
        return stream_async(relay(ctx.router.answer(text, session_id), _after_event))

    @router.post("/clarify")
    async def clarify(resp: ClarifyResponse):
        if ctx.mock:
            return stream_events(fixtures.load("stream_query"))
        try:
            outcome = await run_in_threadpool(ctx.memory.apply, resp)
        except QuestionClosed as exc:
            raise HTTPException(410, str(exc)) from None
        except ClarifyError as exc:  # the question stays open so the UI can ask again
            raise HTTPException(422, str(exc)) from None
        ctx.clarifier.stash(outcome)
        await run_in_threadpool(
            audit.record, ctx.db, "clarify", {"query_id": resp.query_id, "fact_id": outcome.fact.id, "kind": outcome.fact.kind}
        )
        return stream_async(relay(ctx.router.resume(resp), _after_event))

    @router.post("/voice")
    async def voice(request: Request):
        audio = await request.body()
        if ctx.mock:
            return {"text": "person in red at the main gate"}
        if not audio:
            raise HTTPException(422, "no audio received")
        if len(audio) > MAX_VOICE_BYTES:
            raise HTTPException(413, "audio is too large")
        if ctx.settings["onprem"]:
            raise HTTPException(
                503, "Voice needs a cloud model and on-prem mode is on. Use the browser microphone or type the question."
            )
        try:
            return {"text": await ctx.gateway.transcribe(audio)}
        except LLMError as exc:
            raise HTTPException(502, str(exc)) from None

    return router
