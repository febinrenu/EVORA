"""Question -> QueryPlan: fast path, then the plan cache, then the language model.

The gateway already falls back from Groq to the local model, so this module only
decides *whether* a model call is needed. Plans are cached unanchored (time phrase
only); times are resolved against reference_now on every request.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, tzinfo
from typing import Protocol

from contracts.models import QueryPlan, Referent
from pydantic import ValidationError

from evora.core.db import Database
from evora.llm.gateway import Gateway
from evora.llm.prompts import build_planner_messages
from evora.llm.schemas import LLMError
from evora.query import fastpath
from evora.query.timeparse import parse_tz, resolve_window

log = logging.getLogger("evora.query.planner")

DETECTOR_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase",
                    "umbrella"}
MAX_LIMIT = 50


class PlanningError(RuntimeError):
    """No fast-path, cached or model plan could be produced."""


CameraLike = fastpath.CameraLike


class PlanCache(Protocol):
    def get(self, key: str) -> QueryPlan | None: ...
    def put(self, key: str, plan: QueryPlan) -> None: ...


class MemoryPlanCache:
    def __init__(self) -> None:
        self._plans: dict[str, str] = {}

    def get(self, key: str) -> QueryPlan | None:
        raw = self._plans.get(key)
        return QueryPlan.model_validate_json(raw) if raw else None

    def put(self, key: str, plan: QueryPlan) -> None:
        self._plans[key] = plan.model_dump_json()


class SqlitePlanCache:
    """Backed by the plan_cache table of a workspace database."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, key: str) -> QueryPlan | None:
        with self._db.read() as conn:
            row = conn.execute("SELECT plan FROM plan_cache WHERE norm_text=?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return QueryPlan.model_validate_json(row["plan"])
        except ValidationError:
            return None  # written by an older schema: treat as a miss

    def put(self, key: str, plan: QueryPlan) -> None:
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO plan_cache(norm_text, plan, created_at) VALUES(?, ?, ?) "
                "ON CONFLICT(norm_text) DO UPDATE SET plan=excluded.plan, created_at=excluded.created_at",
                (key, plan.model_dump_json(), time.time()),
            )


def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s:]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def cache_key(text: str, cameras: Sequence[CameraLike]) -> str:
    """Plans name camera ids, so the key includes the camera set they were made for."""
    sig = hashlib.sha1("|".join(sorted(f"{c.id}={c.name}" for c in cameras)).encode()).hexdigest()[:8]
    return f"{normalize_text(text)}\x1f{sig}"


def reference_now(db: Database, fallback: float | None = None) -> float:
    """Anchor for relative phrases: the meta override, else the end of the latest footage."""
    override = db.get_meta("reference_now")
    if override:
        try:
            return float(override)
        except ValueError:
            log.warning("ignoring non-numeric meta.reference_now %r", override)
    with db.read() as conn:
        row = conn.execute("SELECT MAX(t0 + COALESCE(duration_s, 0)) AS t FROM cameras").fetchone()
    if row is not None and row["t"] is not None:
        return float(row["t"])
    return fallback if fallback is not None else time.time()


def workspace_tz(db: Database) -> tzinfo:
    return parse_tz(db.get_meta("tz"))


def sanitize(plan: QueryPlan, cameras: Sequence[CameraLike]) -> QueryPlan:
    """Enforce the rules the prompt asks for, whatever the model returned."""
    known = {c.id for c in cameras}
    by_name = {fastpath.norm_name(c.name): c.id for c in cameras}
    plan = plan.model_copy(deep=True)

    camera_ids = [c for c in plan.camera_ids if c in known]
    for target in plan.targets:
        target.cls = [c for c in target.cls if c in DETECTOR_CLASSES]
    plan.limit = max(1, min(plan.limit, MAX_LIMIT))

    # a place that is really a camera name is a camera filter, not something to ask about
    if plan.place is not None:
        cam = by_name.get(fastpath.norm_name(plan.place.text))
        if cam is not None:
            if cam not in camera_ids:
                camera_ids.append(cam)
            plan.unresolved = [r for r in plan.unresolved if r.text != plan.place.text]
            plan.place = None
    plan.camera_ids = camera_ids

    # unresolved holds only things that are not cameras; the place is always checked against memory
    unresolved = [r for r in plan.unresolved if fastpath.norm_name(r.text) not in by_name]
    if plan.place is not None and not any(r.text == plan.place.text for r in unresolved):
        unresolved.append(Referent(text=plan.place.text, role="place"))
    plan.unresolved = unresolved
    return plan


@dataclass
class PlanResult:
    plan: QueryPlan
    notes: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class Planner:
    def __init__(self, gateway: Gateway | None, cache: PlanCache | None = None) -> None:
        self._gateway = gateway
        self._cache = cache if cache is not None else MemoryPlanCache()

    async def plan(
        self,
        text: str,
        cameras: Sequence[CameraLike],
        reference: float,
        tz: tzinfo = UTC,
    ) -> PlanResult:
        started = time.monotonic()
        notes: list[str] = []
        timings: dict[str, float] = {}

        plan = fastpath.parse(text, cameras)
        if plan is not None:
            timings["plan_fastpath"] = _ms(started)
        else:
            key = cache_key(text, cameras)
            cached = self._cache.get(key)
            if cached is not None:
                plan = cached.model_copy(update={"source": "cache"})
                timings["plan_cache"] = _ms(started)
            else:
                plan = await self._ask_model(text, cameras, notes)
                self._cache.put(key, plan)
                timings["plan_llm"] = _ms(started)

        window, understood = resolve_window(plan.time, reference, tz)
        if not understood:
            notes.append(f'Could not interpret the time phrase "{plan.time.phrase}"; searched all footage.')
        return PlanResult(plan=plan.model_copy(update={"time": window}), notes=notes, timings_ms=timings)

    async def _ask_model(self, text: str, cameras: Sequence[CameraLike], notes: list[str]) -> QueryPlan:
        if self._gateway is None:
            raise PlanningError("question not understood by the fast path and no language model is configured")
        messages = build_planner_messages([{"id": c.id, "name": c.name} for c in cameras], text)
        try:
            plan, backend = await self._gateway.chat_json_ex("planner", messages, QueryPlan)
        except LLMError as exc:
            raise PlanningError(f"could not plan this question: {exc}") from exc
        source = "llm" if backend == "groq" else "local_llm"
        if source == "local_llm":
            notes.append("Planned with the local model (cloud planner unavailable or off).")
        return sanitize(plan.model_copy(update={"source": source}), cameras)


def _ms(since: float) -> float:
    return round((time.monotonic() - since) * 1000, 2)
