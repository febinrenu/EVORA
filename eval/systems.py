"""The systems the harness can evaluate: ours (the real pipeline) and baseline B0.

Both run against an already-indexed workspace and receive the same planner output, camera
filter and time window, so only retrieval differs (PLAN section 9.4).

`ours` is driven exactly as the UI drives it: it asks the question, and when the router stops
with a clarification it answers from the query's ground-truth `clarify_answer`, through the same
memory code the /api/clarify route uses.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from datetime import tzinfo
from pathlib import Path
from typing import Any

import numpy as np
from contracts.models import Answer, ClarifyResponse, QueryPlan, StreamEvent, Target, Zone

from eval.metrics import RunResult
from eval.queries import QueryItem
from evora.baseline.b0_frames import FrameIndex, b0_search, windows_to_evidence
from evora.query.planner import PlanResult, reference_now, workspace_tz

MAX_CLARIFICATIONS = 3  # a question that still is not answered after three clarifications is a failure


# ------------------------------------------------------------------ planner ablation
class RawPlanner:
    """Ablation C3 (`query.planner = raw`): no parsing, no model. The whole question is the search text."""

    async def plan(self, text: str, cameras, reference: float, tz: tzinfo) -> PlanResult:  # noqa: ANN001
        plan = QueryPlan(
            intent="list", targets=[Target(noun="object", cls=[], attributes=[], embed_text=text)], limit=10,
            source="fastpath",
        )
        return PlanResult(plan=plan, notes=["Raw-text retrieval: the question was not parsed."])


# ------------------------------------------------------------------------- ours
def clarify_response(query_id: str, answer: dict[str, Any] | None) -> ClarifyResponse | None:
    """Turn a ground-truth `clarify_answer` ({camera_id, zone: {...}} or tod bounds or a track) into a response."""
    if not answer:
        return None
    fields = {k: v for k, v in answer.items() if k in ClarifyResponse.model_fields and k != "zone"}
    zone = answer.get("zone")
    if zone:
        camera_id = answer.get("camera_id") or zone.get("camera_id")
        fields["zone"] = Zone(id=zone.get("id", "eval_zone"), camera_id=camera_id, kind=zone["kind"],
                              points=[tuple(p) for p in zone.get("points", [])], direction=zone.get("direction", "any"))
    return ClarifyResponse(query_id=query_id, **fields)


class OursSystem:
    name = "ours"

    def __init__(self, ctx: Any, name: str = "ours") -> None:
        self.ctx = ctx
        self.name = name

    async def run(self, item: QueryItem) -> RunResult:
        started = time.perf_counter()
        events: list[StreamEvent] = []
        marks: dict[str, float] = {}

        async def drain(stream) -> None:  # noqa: ANN001
            async for ev in stream:
                events.append(ev)
                if ev.type in ("evidence", "answer"):
                    marks.setdefault("first", (time.perf_counter() - started) * 1000)

        await drain(self.ctx.router.answer(item.text, "eval"))
        clarifications = 0
        while events and events[-1].type == "clarify" and clarifications < MAX_CLARIFICATIONS:
            clarifications += 1
            request = events[-1].data
            resp = clarify_response(request["query_id"], item.clarify_answer)
            if resp is None:
                break  # the query has no scripted answer: the question stays open, which scores as a miss
            outcome = await asyncio.to_thread(self.ctx.memory.apply, resp)
            self.ctx.clarifier.stash(outcome)
            await drain(self.ctx.router.resume(resp))

        total = (time.perf_counter() - started) * 1000
        answers = [e for e in events if e.type == "answer"]
        plans = [e for e in events if e.type == "plan"]
        errors = [e.data.get("message", "error") for e in events if e.type == "error"]
        required = len(item.first_time_requires_clarify)
        return RunResult(
            query_id=item.id,
            answer=Answer.model_validate(answers[-1].data) if answers else None,
            asked_clarify=clarifications > 0,
            reasks=max(0, clarifications - required),
            ttfa_ms=marks.get("first", total),
            ttva_ms=total,
            error="; ".join(errors) if errors and not answers else None,
            plan_source=plans[0].data.get("source") if plans else None,
        )

    async def aclose(self) -> None:
        http = getattr(self.ctx, "http", None)
        if http is not None:
            await http.aclose()


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Set dotted config keys ({"retrieval.unit": "frame"}) on a copy of the configuration."""
    import copy

    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = out
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return out


def build_ours(
    slug: str,
    root: Path | None = None,
    overrides: dict[str, Any] | None = None,
    replay_path: Path | None = None,
    gateway: Any = None,
    name: str = "ours",
) -> OursSystem:
    """The full pipeline on a stored workspace, with config switches applied.

    Structured model calls are recorded to / replayed from `replay_path` (mode auto) when given, so a
    second run, or an ablation that plans identically, costs no Groq calls.
    """
    from evora.api.context import AppContext
    from evora.core.config import load_config

    overrides = overrides or {}
    cfg = apply_overrides(load_config(), {k: v for k, v in overrides.items() if not k.startswith("query.")})
    os.environ["evora_WORKSPACE"] = slug
    if gateway is None:
        gateway = _gateway(replay_path)
    ctx = AppContext.build(cfg, root, gateway=gateway, mock=False)
    if overrides.get("query.planner") == "raw":
        ctx.router._planner = RawPlanner()
    if overrides.get("query.verify") is False:
        ctx.router._verifier = None
    return OursSystem(ctx, name)


def _gateway(replay_path: Path | None) -> Any:
    import httpx

    from evora.core.config import load_env_file
    from evora.llm.gateway import Gateway
    from evora.llm.keypool import KeyPool
    from evora.llm.schemas import GatewayConfig

    load_env_file()
    extra: dict[str, Any] = {"replay_path": replay_path, "replay_mode": "auto"} if replay_path else {}
    return Gateway(GatewayConfig.from_env(os.environ, **extra), KeyPool.from_env(os.environ), httpx.AsyncClient())


# --------------------------------------------------------------------------- B0
class B0System:
    """Baseline B0: whole-frame embeddings ranked by cosine to the text, adjacent hits merged.

    The index is the workspace's own whole-frame scene tiles (one every ~2 s, the same SigLIP2
    vectors), so no extra decoding is needed. It takes the same plan, camera filter and window as `ours`.
    """

    name = "b0"

    def __init__(self, db: Any, store: Any, planner: Any, embedder: Any, name: str = "b0") -> None:
        self._db, self._planner, self._embedder = db, planner, embedder
        self.name = name
        self.index = FrameIndex()
        self._cameras = self._load_cameras()
        self._load_frames(store)

    def _load_cameras(self) -> list[Any]:
        from types import SimpleNamespace

        with self._db.read() as conn:
            return [SimpleNamespace(id=r["id"], name=r["name"], t0=r["t0"])
                    for r in conn.execute("SELECT id, name, t0 FROM cameras ORDER BY id")]

    def _load_frames(self, store: Any) -> None:
        if "scenes" not in set(store.list_tables().tables):
            return
        rows = store.open_table("scenes").to_arrow().to_pylist()
        by_cam: dict[str, list[tuple[float, Any]]] = {}
        for r in rows:
            if r["tile"] == "full":
                by_cam.setdefault(r["camera_id"], []).append((float(r["t"]), r["vector"]))
        for cam, frames in by_cam.items():
            frames.sort(key=lambda f: f[0])
            self.index.add(cam, [t for t, _ in frames], np.asarray([v for _, v in frames], dtype=np.float32))

    async def run(self, item: QueryItem) -> RunResult:
        started = time.perf_counter()
        ref, tz = reference_now(self._db), workspace_tz(self._db)
        planned = await self._planner.plan(item.text, self._cameras, ref, tz)
        plan = planned.plan
        answer_cam = (item.clarify_answer or {}).get("camera_id")
        if answer_cam and answer_cam not in plan.camera_ids:  # the camera filter ours ends up with after clarifying
            plan = plan.model_copy(update={"camera_ids": [*plan.camera_ids, answer_cam]})
        windows = await asyncio.to_thread(b0_search, self.index, self._embedder, plan, tz, limit=plan.limit)
        evidence = windows_to_evidence(windows, self._cameras)
        verdict = "found" if evidence else "not_found"
        answer = Answer(query_id=item.id, text=f"{len(evidence)} frame matches", verdict=verdict, evidence=evidence,
                        confidence=evidence[0].score if evidence else 0.0, plan=plan)
        ms = (time.perf_counter() - started) * 1000
        return RunResult(item.id, answer, ttfa_ms=ms, ttva_ms=ms, plan_source=plan.source)


def build_b0(slug: str, root: Path | None = None, replay_path: Path | None = None, gateway: Any = None,
             embedder: Any = None, name: str = "b0") -> B0System:
    from evora.core import perception_adapter
    from evora.core import workspace as wsmod
    from evora.core.db import open_db
    from evora.core.vectors import open_store
    from evora.query.planner import Planner, SqlitePlanCache

    ws = wsmod.get(slug, root)
    db = open_db(ws.db_path)
    if gateway is None:
        gateway = _gateway(replay_path)
    embedder = embedder or perception_adapter.get_query_embedder()
    if embedder is None:
        raise RuntimeError("B0 needs the SigLIP2 query embedder (install the perception extra)")
    return B0System(db, open_store(ws.vectors_dir), Planner(gateway, SqlitePlanCache(db)), embedder, name)


SYSTEM_BUILDERS: dict[str, Callable[..., Any]] = {"ours": build_ours, "b0": build_b0}
