"""Operator views: the audit log, past questions and a live health check. `evora_MOCK=1` returns canned answers."""
from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool

from evora.api.context import AppContext
from evora.core import doctor
from evora.core.config import REPO_ROOT
from evora.evidence import audit

CANNED_AUDIT = [{"id": "aud_000000000001", "actor": "local", "action": "export", "t": 1790000000.0,
                 "detail": {"evidence_id": "ev_001", "faces_blurred": True}}]
CANNED_QUERIES = [{"id": "q_001", "text": "was the white van at the main gate?", "intent": "exists", "verdict": "yes",
                   "count": None, "n_results": 1, "confidence": 0.9, "timings_ms": {"total": 800.0}, "created_at": 1790000000.0}]


def _loads(raw: str | None) -> Any:
    try:
        return json.loads(raw or "null")
    except json.JSONDecodeError:
        return None


def _query_row(row: Any, full: bool) -> dict[str, Any]:
    """One past question. A damaged row still lists (with what is readable) instead of breaking the page."""
    plan, answer = _loads(row["plan"]), _loads(row["answer"])
    plan = plan if isinstance(plan, dict) else {}
    answer = answer if isinstance(answer, dict) else {}
    timings = _loads(row["timings"])
    out = {
        "id": row["id"], "text": row["text"], "intent": plan.get("intent"), "verdict": answer.get("verdict"),
        "count": answer.get("count"), "n_results": len(answer.get("evidence") or []), "confidence": answer.get("confidence"),
        "timings_ms": timings if isinstance(timings, dict) else {}, "created_at": row["created_at"],
    }
    if full:
        out.update(plan=plan or None, answer=answer or None)
    return out


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/audit")
    def audit_log(action: str | None = None, limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
        if ctx.mock:
            return CANNED_AUDIT
        return audit.entries(ctx.db, action, limit=limit, newest_first=True)

    @router.get("/queries")
    def past_queries(limit: int = Query(50, ge=1, le=500), full: bool = False) -> list[dict[str, Any]]:
        if ctx.mock:
            return CANNED_QUERIES
        with ctx.db.read() as c:
            rows = c.execute(
                "SELECT id, text, plan, answer, timings, created_at FROM query_log ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        return [_query_row(r, full) for r in rows]

    @router.get("/doctor")
    async def health_check() -> dict[str, Any]:
        """The same checks as `make doctor --quick`, for this running app (no network probes)."""
        if ctx.mock:
            return {"verdict": "Demo-ready", "ok": True, "checks": [], "took_s": 0.0}
        env = doctor.Env(root=REPO_ROOT, cfg=ctx.cfg, on_prem=bool(ctx.settings["onprem"]), quick=True)
        started = time.monotonic()
        checks = await run_in_threadpool(doctor.run_checks, env)
        return {**json.loads(doctor.to_json(checks)), "took_s": round(time.monotonic() - started, 2)}

    return router
