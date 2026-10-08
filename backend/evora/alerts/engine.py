"""Alert engine: evaluates events against active watches and raises each alert once."""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from contracts.models import Alert, Evidence, StandingQuery

from evora.alerts import store as alert_store
from evora.alerts.notify import Notifier
from evora.alerts.store import StandingRule
from evora.core import cameras as cams
from evora.core import live_sessions, zones
from evora.core.bus import Bus
from evora.core.db import Database
from evora.evidence import store as evidence_store
from evora.memory.kb import KnowledgeBase
from evora.query.logic import EventRec, tod_contains
from evora.query.planner import workspace_tz

log = logging.getLogger("evora.alerts.engine")

ZONE_EVENTS = {"cross_line", "enter_zone", "exit_zone", "dwell"}  # everything else is about the whole camera view
EVIDENCE_PAD_S = 2.0


class AlertEngine:
    def __init__(
        self, db: Database, bus: Bus, kb: KnowledgeBase, notifier: Notifier | None = None, *,
        push_cap: int = 0, push_window_s: float = 60.0, clock: Callable[[], float] = time.monotonic,
        on_alert: Callable[[Alert, bool], None] | None = None,
    ) -> None:
        self.db, self.bus, self.kb, self.notifier = db, bus, kb, notifier
        self._push_cap, self._push_window, self._now = push_cap, push_window_s, clock  # a cap of 0 means no cap
        self.on_alert = on_alert  # told about every alert that is raised: (alert, historical)
        self._pushed: dict[str, deque[float]] = {}
        self._held: dict[str, int] = {}
        self._lock = threading.RLock()
        self._rules_cache: list[tuple[StandingQuery, StandingRule]] | None = None
        self._last_alert: dict[tuple[str, str], float] = {}

    # --- rules ---
    def invalidate(self) -> None:
        with self._lock:
            self._rules_cache = None

    def _rules(self) -> list[tuple[StandingQuery, StandingRule]]:
        with self._lock:
            if self._rules_cache is None:
                self._rules_cache = [
                    (sq, StandingRule.model_validate(sq.rule)) for sq in alert_store.list_standing(self.db, active_only=True)
                ]
            return self._rules_cache

    def _scope(self, rule: StandingRule) -> tuple[str | None, set[str], str]:
        """(zone id, camera ids, the zone's own direction). The place is looked up again so corrections apply."""
        zone_id, cameras = rule.zone_id, set(rule.camera_ids)
        if rule.place:
            facts = self.kb.find_exact("place", rule.place)
            if facts:
                binding = facts[0].binding
                zone_id = binding.get("zone_id")
                cameras = {binding["camera_id"]} if binding.get("camera_id") else cameras
        direction = "any"
        if zone_id:
            try:
                direction = zones.get_zone(self.db, zone_id).direction
            except zones.ZoneNotFound:
                zone_id = None
        return zone_id, cameras, direction

    # --- matching ---
    def _matches(self, rule: StandingRule, ev: EventRec) -> tuple[bool, str | None]:
        """(matches, track identity used for the cooldown)."""
        if ev.kind not in rule.events:
            return False, None
        zone_id, cameras, zone_direction = self._scope(rule)
        if cameras and ev.camera_id not in cameras:
            return False, None
        if ev.kind in ZONE_EVENTS and zone_id and ev.zone_id != zone_id:
            return False, None
        if ev.kind == "cross_line":
            got = ev.payload.get("direction")
            wanted = [d for d in (rule.direction, zone_direction if zone_direction != "any" else None) if d]
            if got is not None and any(got != d for d in wanted):
                return False, None
        with self.db.read() as c:
            track = c.execute("SELECT cls, global_id FROM tracks WHERE id=?", (ev.track_id,)).fetchone()
        if track is None or track["cls"] not in rule.targets:
            return False, None
        if not tod_contains(ev.t, rule.tod_after, rule.tod_before, workspace_tz(self.db)):
            return False, None
        return True, track["global_id"] or ev.track_id

    # --- alerts ---
    def evaluate(self, event: Mapping[str, Any], historical: bool = False, only_rule: str | None = None) -> list[Alert]:
        ev = EventRec.from_row(event)
        raised: list[Alert] = []
        with self._lock:
            for sq, rule in self._rules():
                if only_rule and sq.id != only_rule:
                    continue
                ok, identity = self._matches(rule, ev)
                if not ok or identity is None:
                    continue
                last = self._last_alert.get((sq.id, identity))
                if last is not None and abs(ev.t - last) < rule.cooldown_s:
                    continue
                alert = self._build(sq, rule, ev, historical)
                if not alert_store.insert_alert(self.db, alert, ev.track_id):
                    continue  # this watch already alerted for this event
                self._last_alert[(sq.id, identity)] = ev.t
                raised.append(alert)
        for alert in raised:
            self._announce(alert, rule_summary=self._summary_of(alert), historical=historical)
        return raised

    def _summary_of(self, alert: Alert) -> str:
        for sq, rule in self._rules():
            if sq.id == alert.standing_query_id:
                return rule.summary
        return ""

    def _build(self, sq: StandingQuery, rule: StandingRule, ev: EventRec, historical: bool) -> Alert:
        digest = hashlib.sha1(f"{sq.id}:{ev.id}".encode()).hexdigest()[:12]
        cam = cams.get_camera(self.db, ev.camera_id)
        with self.db.read() as c:
            point = c.execute(
                "SELECT x1,y1,x2,y2 FROM track_points WHERE track_id=? ORDER BY ABS(t-?) LIMIT 1", (ev.track_id, ev.t)
            ).fetchone()
            track = c.execute("SELECT global_id FROM tracks WHERE id=?", (ev.track_id,)).fetchone()
        bbox = tuple(point) if point and None not in tuple(point) else None
        why = [rule.summary, f"{ev.kind.replace('_', ' ')} at {self._clock(ev.t)}"]
        if historical:
            why.append("found in earlier footage")
        evidence = Evidence(
            id=f"ev_al_{digest}", camera_id=cam.id, camera_name=cam.name, t_start=ev.t - EVIDENCE_PAD_S,
            t_end=ev.t + EVIDENCE_PAD_S, t_peak=ev.t, offset_s=self._offset(cam, ev.t), track_id=ev.track_id,
            global_id=track["global_id"] if track else None, bbox=bbox,  # type: ignore[arg-type]
            thumb_url=f"/api/media/thumb/ev_al_{digest}.jpg", clip_url=f"/api/media/clip/ev_al_{digest}.mp4",
            score=1.0, why=why,
        )
        evidence_store.register(self.db, evidence)
        return Alert(id=f"al_{digest}", standing_query_id=sq.id, t=ev.t, camera_id=cam.id, evidence=evidence)

    def _offset(self, cam, t: float) -> float:  # noqa: ANN001
        mapped = live_sessions.file_offset(self.db, cam.id, cam.duration_s, t)
        return mapped if mapped is not None else max(t - cam.t0, 0.0)

    def _clock(self, t: float) -> str:
        return datetime.fromtimestamp(t, workspace_tz(self.db)).strftime("%H:%M")

    def _announce(self, alert: Alert, rule_summary: str, historical: bool) -> None:
        self.bus.publish("alert", {"alert": alert.model_dump(mode="json"), "historical": historical})
        if self.on_alert is not None:
            try:
                self.on_alert(alert, historical)
            except Exception:  # noqa: BLE001 - a side effect of an alert must never stop the alert
                log.exception("alert hook failed for %s", alert.id)
        if self.notifier is not None and not historical:  # a phone is only buzzed for what is happening now
            allowed, held = self._push_allowed(alert.standing_query_id)
            if not allowed:
                self.bus.publish("alert_burst", {"standing_query_id": alert.standing_query_id, "held_back": held})
                return
            text = f"{rule_summary} ({alert.evidence.camera_name}, {self._clock(alert.t)})"
            if held:
                text += f" (+{held} more held back)"
            self.notifier.push("evora alert", text)

    def _push_allowed(self, watch_id: str) -> tuple[bool, int]:
        """(may this alert buzz a phone, how many earlier ones were held back). Every alert is still stored and shown."""
        if self._push_cap <= 0:
            return True, 0
        with self._lock:
            now = self._now()
            sent = self._pushed.setdefault(watch_id, deque())
            while sent and now - sent[0] >= self._push_window:
                sent.popleft()
            if len(sent) >= self._push_cap:
                self._held[watch_id] = self._held.get(watch_id, 0) + 1
                return False, self._held[watch_id]
            sent.append(now)
            return True, self._held.pop(watch_id, 0)

    # --- recorded footage ---
    def backfill(self, camera_id: str | None = None, only_rule: str | None = None, historical: bool = True) -> int:
        """Evaluate stored events in time order. Safe to repeat: an event alerts a watch only once."""
        sql, args = "SELECT * FROM events", []
        if camera_id:
            sql, args = sql + " WHERE camera_id=?", [camera_id]
        with self.db.read() as c:
            rows = [dict(r) for r in c.execute(sql + " ORDER BY t, id", args)]
        total = 0
        for row in rows:
            total += len(self.evaluate(row, historical=historical, only_rule=only_rule))
        if total:
            log.info("backfill raised %d alert(s)", total)
        return total
