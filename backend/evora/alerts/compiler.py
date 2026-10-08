"""Compile "tell me when ..." into a StandingRule using the planner and remembered facts."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from contracts.models import ClarifyRequest, QueryPlan

from evora.alerts.store import StandingRule
from evora.core import cameras as cams
from evora.core import zones
from evora.query.logic import ACTION_EVENTS, LINE_DIRECTION
from evora.query.planner import PlanningError, reference_now, workspace_tz

log = logging.getLogger("evora.alerts.compiler")

# what a bare "tell me when someone is at X" means, per zone kind
DEFAULT_EVENTS: dict[str, tuple[str, ...]] = {"line": ("cross_line",), "polygon": ("enter_zone",), "frame": ("appear",)}
ACTION_WORDS = {
    "pass_through": "passes", "enter": "enters", "exit": "leaves", "dwell": "stays at", "appear": "appears at", "any": "is at",
}


class CompileError(ValueError):
    """The text cannot become a watch. The message is meant for the user."""


@dataclass(frozen=True)
class Compiled:
    rule: StandingRule
    plan: QueryPlan


@dataclass(frozen=True)
class NeedsClarification:
    request: ClarifyRequest


class StandingCompiler:
    def __init__(self, db, planner: Any, memory: Any, default_cooldown_s: float = 30.0) -> None:  # noqa: ANN001
        self.db, self.planner, self.memory, self.default_cooldown_s = db, planner, memory, default_cooldown_s

    async def compile(self, text: str) -> Compiled | NeedsClarification:
        text = text.strip()
        if not text:
            raise CompileError("Say what to watch for, for example: tell me when someone enters the main gate after 8pm.")
        try:
            planned = await self.planner.plan(text, cams.list_cameras(self.db), reference_now(self.db), workspace_tz(self.db))
        except PlanningError as exc:
            raise CompileError(str(exc)) from None
        plan: QueryPlan = planned.plan

        camera_ids: set[str] = set(plan.camera_ids)
        zone_id: str | None = None
        tod_after = plan.time.tod_after if plan.time else None
        tod_before = plan.time.tod_before if plan.time else None
        for ref in plan.unresolved:
            resolution = await self.memory.resolve(ref)
            if resolution.status != "bound":
                query_id = f"sq_{uuid.uuid4().hex[:10]}"
                return NeedsClarification(self.memory.ask(query_id, text, plan, ref, resolution))
            binding = resolution.facts[0].binding
            if ref.role == "place":
                if binding.get("camera_id"):
                    camera_ids.add(binding["camera_id"])
                zone_id = binding.get("zone_id") or zone_id
            elif ref.role == "time":
                tod_after = binding.get("tod_after") or tod_after
                tod_before = binding.get("tod_before") or tod_before

        targets = sorted({c for t in plan.targets for c in t.cls})
        attributes = sorted({a for t in plan.targets for a in t.attributes})
        if not targets:
            raise CompileError("I could not tell what to watch for (a person, a car, ...). Try naming it.")

        zone_kind = "frame"
        if zone_id is not None:
            try:
                zone_kind = zones.get_zone(self.db, zone_id).kind
            except zones.ZoneNotFound:
                zone_id = None  # the fact points at a zone that was deleted: watch the whole camera instead
        events = list(self._events(plan.action, zone_kind))
        if not events:
            raise CompileError(f"Watching for '{plan.action.replace('_', ' ')}' does not apply to a {zone_kind} here.")
        direction = LINE_DIRECTION.get(plan.action) if zone_kind == "line" else None

        tod_after, tod_before = (t[:5] if t else t for t in (tod_after, tod_before))  # models sometimes add seconds
        place = plan.place.text if plan.place else None
        rule = StandingRule(
            targets=targets, attributes=attributes, place=place,
            camera_ids=sorted(camera_ids), zone_id=zone_id, events=events, direction=direction,
            tod_after=tod_after, tod_before=tod_before, cooldown_s=self.default_cooldown_s,
            summary=self._summary(plan, targets, place, camera_ids, tod_after, tod_before, attributes),
        )
        return Compiled(rule, plan)

    @staticmethod
    def _events(action: str, zone_kind: str) -> tuple[str, ...]:
        if action == "any":
            return DEFAULT_EVENTS[zone_kind]
        kinds = ACTION_EVENTS.get(action, {}).get(zone_kind, ())
        if not kinds and zone_kind == "frame" and action == "pass_through":
            return DEFAULT_EVENTS["frame"]  # nothing "passes" a whole view; treat it as appearing in it
        return kinds

    def _summary(self, plan, targets, place, camera_ids, after, before, attributes=()) -> str:  # noqa: ANN001
        names = {c.id: c.name for c in cams.list_cameras(self.db)}
        what = " or ".join(targets)
        if attributes:
            what += f" ({', '.join(attributes)})"
        where = place or (", ".join(names.get(c, c) for c in sorted(camera_ids)) or "any camera")
        if after and before:
            when = f" between {after} and {before}"
        elif after:
            when = f" after {after}"
        elif before:
            when = f" before {before}"
        else:
            when = ""
        return f"Alert when a {what} {ACTION_WORDS.get(plan.action, 'is at')} {where}{when}."
