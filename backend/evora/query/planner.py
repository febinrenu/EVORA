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
from functools import lru_cache
from typing import Protocol

from contracts.models import QueryPlan, Referent, Target, TimeWindow
from pydantic import ValidationError

from evora.core.db import Database
from evora.llm.gateway import Gateway
from evora.llm.prompts import build_planner_messages, planner_system_prompt
from evora.llm.schemas import LLMError
from evora.query import fastpath
from evora.query.fuse import CARRIED, GARMENTS
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


def _is_ordinary_time(text: str, reference: float, tz: tzinfo) -> bool:
    if fastpath.is_standard_time(text) or _clock_bounds(text) is not None:
        return True
    resolved, ok = resolve_window(TimeWindow(phrase=text), reference, tz)
    return ok and resolved is not None and resolved.start is not None


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@lru_cache(maxsize=1)
def _prompt_signature() -> str:
    """A plan made under an older prompt must not be served after the prompt changes."""
    return hashlib.sha1(planner_system_prompt().encode()).hexdigest()[:6]


def cache_key(text: str, cameras: Sequence[CameraLike]) -> str:
    """Plans name camera ids, so the key includes the camera set they were made for, and the prompt that made them."""
    sig = hashlib.sha1("|".join(sorted(f"{c.id}={c.name}" for c in cameras)).encode()).hexdigest()[:8]
    return f"{normalize_text(text)}\x1f{sig}\x1f{_prompt_signature()}"


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


# only "my car" style references can be remembered; "something heavy" or "the vehicle" are just descriptions
_REMEMBERED_OBJECT = re.compile(r"^(my|our|mine|that|this|these|those|his|her|their|the same)\b", re.IGNORECASE)


# "when did it happen" asks for a time; it is not a time of day to look up
_ASKS_FOR_TIME = re.compile(
    r"^(?:at |on |in )?(?:what|which)\s+(?:time|hour|day|date|moment)s?\b"
    r"|^when(?:$|\s+(?:did|does|do|was|were|is|are|will|can|could|has|have)\b)",  # a question, not "when it was dark"
    re.IGNORECASE,
)


def _is_time_question(text: str) -> bool:
    return bool(_ASKS_FOR_TIME.match(text.strip()))


_CLOCK_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?(?::\d{2})?\s*(am|pm)?\b", re.IGNORECASE)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
_OPEN_ENDED = re.compile(r"\b(after|before|since|until|till|by|around|about|early|late)\b", re.IGNORECASE)
_BARE_RANGE = re.compile(r"^\D*\d{1,2}\s*(?:and|to|-|–|until|till)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.IGNORECASE)


def _clock_bounds(phrase: str) -> tuple[str, str] | None:
    """Times of day named outright: "at 13:53", "at 00:13:53", "from 4pm to 5pm", "13:24 to 13:27" -> ("HH:MM", "HH:MM").

    One time means the minute it falls in. "between 8 and 9 pm" gives the first number the second one's am or pm.
    Whatever cannot be read exactly (a lone bare number, "after 8 pm", "11 and 1 pm") returns None, so it is asked
    about, rather than being turned into a window nobody said.
    """
    found = []  # (hour, minute, meridiem, written with a colon or am/pm)
    phrase = _DATE.sub(" ", phrase)  # the digits of a date are not times
    for m in _CLOCK_TIME.finditer(phrase):
        meridiem = (m.group(3) or "").lower()
        found.append((int(m.group(1)), int(m.group(2) or 0), meridiem, m.group(2) is not None or bool(meridiem)))
    if any(not explicit for *_, explicit in found):
        # the only bare number allowed is the start of "8 and 9 pm", which takes the end's am or pm
        if len(found) != 2 or found[0][3] or not found[1][3] or not found[1][2] or not _BARE_RANGE.match(phrase):
            return None
        start_hour, end_hour = found[0][0], found[1][0]
        if not (1 <= start_hour <= 12 and 1 <= end_hour <= 12) or start_hour % 12 > end_hour % 12:
            return None  # "11 and 1 pm" may cross noon: ask
        found[0] = (start_hour, found[0][1], found[1][2], True)
    elif len(found) == 1 and _OPEN_ENDED.search(phrase):
        return None  # "after 8 pm" is open-ended, not a minute
    minutes: list[int] = []
    for hour, minute, meridiem, _ in found:
        if meridiem:
            if not 1 <= hour <= 12:
                return None
            hour = hour % 12 + (12 if meridiem == "pm" else 0)
        if hour > 23 or minute > 59:
            return None
        minutes.append(hour * 60 + minute)
    if not minutes or len(minutes) > 2:
        return None
    start, end = minutes[0], minutes[-1] if len(minutes) == 2 else minutes[0] + 1
    return f"{start // 60:02d}:{start % 60:02d}", f"{end // 60 % 24:02d}:{end % 60:02d}"


def _worn_or_carried(noun: str) -> bool:
    words = re.findall(r"[a-z]+", noun.lower())
    return bool(words) and words[-1] in (GARMENTS | CARRIED)


def _repair_targets(targets: list[Target]) -> list[Target]:
    """Small models split "green jacket" into its own target, forget classes and drop attributes."""
    fixed: dict[str, Target] = {}
    # "the brown shirt guy" comes back as a person plus a second target "brown shirt": the shirt describes the person
    anchor = next((t for t in targets if any(c in DETECTOR_CLASSES for c in t.cls) or t.noun.strip().lower()
                   in fastpath.PERSON_GENERIC | fastpath.PERSON_SPECIFIC), None)
    for t in targets:
        noun = t.noun.strip().lower()
        if not noun or noun in fastpath.COLOURS:
            continue  # a colour is an attribute, not an object
        if anchor is not None and t is not anchor and _worn_or_carried(noun):
            words = re.findall(r"[a-z_]+", f"{noun} {t.embed_text}".lower())
            colours = [fastpath.COLOURS[w] for w in words if w in fastpath.COLOURS]
            carried = [fastpath.CARRY[w] for w in words if w in fastpath.CARRY]  # "backpack" is an attribute too
            anchor.attributes = list(dict.fromkeys([*anchor.attributes, *colours, *carried]))
            if noun not in anchor.embed_text.lower():
                verb = "carrying" if noun.split()[-1] in CARRIED else "wearing"
                anchor.embed_text = f"{anchor.embed_text} {verb} {noun}".strip()
            continue
        t.cls = [c for c in t.cls if c in DETECTOR_CLASSES]
        if not t.cls:
            if noun in fastpath.PERSON_GENERIC or noun in fastpath.PERSON_SPECIFIC:
                t.cls = ["person"]
            elif noun in fastpath.VEHICLES:
                t.cls = list(fastpath.VEHICLES[noun][1])
        if not t.attributes:
            words = re.findall(r"[a-z_]+", t.embed_text.lower())
            t.attributes = list(dict.fromkeys(fastpath.COLOURS[w] for w in words if w in fastpath.COLOURS))
        fixed.setdefault(noun, t)
    return list(fixed.values())


def sanitize(plan: QueryPlan, cameras: Sequence[CameraLike], question: str | None = None) -> QueryPlan:
    """Enforce the rules the prompt asks for, whatever the model returned.

    With `question`, a camera id is kept only if the question actually names that camera; models
    sometimes invent one for a place they cannot map.
    """
    known = {c.id for c in cameras}
    by_name = {fastpath.norm_name(c.name): c.id for c in cameras}
    plan = plan.model_copy(deep=True)

    camera_ids = [c for c in plan.camera_ids if c in known]
    if question is not None:
        asked = fastpath.norm_name(question)
        named = {c.id for c in cameras
                 if fastpath.norm_name(c.name) and re.search(r"\b" + re.escape(fastpath.norm_name(c.name)) + r"\b", asked)
                 or c.id.lower() in question.lower()}
        camera_ids = [c for c in camera_ids if c in named]
    plan.targets = _repair_targets(plan.targets)
    plan.limit = max(1, min(plan.limit, MAX_LIMIT))

    # small models list the place under `unresolved` but leave `place` empty
    places = [r for r in plan.unresolved if r.role == "place"]
    if plan.place is None and len(places) == 1:
        plan.place = places[0]

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
    plan.unresolved = [r for r in unresolved if r.role != "object" or _REMEMBERED_OBJECT.match(r.text.strip())]
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
        started = time.perf_counter()
        notes: list[str] = []
        timings: dict[str, float] = {}

        plan = fastpath.parse(text, cameras) or fastpath.parse_action(text, cameras)
        if plan is not None:
            timings["plan_fastpath"] = _ms(started)
        else:
            key = cache_key(text, cameras)
            cached = self._cache.get(key)
            if cached is not None:
                plan = cached.model_copy(deep=True)
                plan.source, plan.targets = "cache", _repair_targets(plan.targets)
                timings["plan_cache"] = _ms(started)
            else:
                plan = await self._ask_model(text, cameras, notes)
                self._cache.put(key, plan)
                timings["plan_llm"] = _ms(started)

        if plan.time is not None and plan.time.phrase and _is_time_question(plan.time.phrase):
            plan = plan.model_copy(update={"time": None})  # "at what time" asks for an answer, it is not a range
        asked = [r for r in plan.unresolved if r.role == "time" and _is_time_question(r.text)]
        plan = plan.model_copy(update={"unresolved": [r for r in plan.unresolved if r not in asked]})
        if plan.time is not None and plan.time.tod_after and plan.time.tod_after == plan.time.tod_before:
            instant = _clock_bounds(plan.time.tod_after)  # "at 13:53" as an empty range matches nothing: use that minute
            if instant is not None:
                bounded = plan.time.model_copy(update={"tod_after": instant[0], "tod_before": instant[1]})
                plan = plan.model_copy(update={"time": bounded})
        window, understood = resolve_window(plan.time, reference, tz)
        if not understood and plan.time is not None and plan.time.phrase and not (plan.time.tod_after or plan.time.tod_before):
            bounds = _clock_bounds(plan.time.phrase)  # a clock time said outright needs no explaining
            if bounds is not None:
                bounded = plan.time.model_copy(update={"tod_after": bounds[0], "tod_before": bounds[1]})
                plan = plan.model_copy(update={"time": bounded})
                window, understood = resolve_window(plan.time, reference, tz)
        # models sometimes list "after 8pm" or "last week" as something to look up; those never need memory
        unresolved = [r for r in plan.unresolved if not (r.role == "time" and _is_ordinary_time(r.text, reference, tz))]
        if not understood and plan.time is not None and plan.time.phrase:
            # a phrase like "after hours" is the user's own word: hand it to memory, which asks once and remembers
            if not any(r.role == "time" and norm_key(r.text) == norm_key(plan.time.phrase) for r in unresolved):
                unresolved.append(Referent(text=plan.time.phrase, role="time"))
        if plan.intent == "standing":
            # a standing rule watches the future: keep the phrase and time-of-day bounds, never fixed dates
            window = window.model_copy(update={"start": None, "end": None}) if window is not None else None
        return PlanResult(plan=plan.model_copy(update={"time": window, "unresolved": unresolved}), notes=notes,
                          timings_ms=timings)

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
        return sanitize(plan.model_copy(update={"source": source}), cameras, text)


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 2)
