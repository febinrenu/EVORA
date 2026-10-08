"""Zone workflow: save a zone, clear its stale events, ask perception to recompute them, tell the UI."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from contracts.models import Zone

from evora.core import perception_adapter, zones
from evora.core.bus import Bus
from evora.core.db import Database
from evora.memory.kb import FactNotFound, KnowledgeBase

log = logging.getLogger("evora.zones")

Recompute = Callable[[str, list[Zone]], int | None]


@dataclass(frozen=True)
class SaveResult:
    zone: Zone
    events: int | None  # None: perception is not installed, so nothing could be computed yet


class ZoneService:
    def __init__(self, db: Database, bus: Bus, kb: KnowledgeBase | None = None, recompute: Recompute | None = None):
        self.db, self.bus, self.kb = db, bus, kb
        self._recompute = recompute or perception_adapter.recompute_events
        self.on_recomputed: Callable[[str], None] | None = None  # e.g. the alert engine re-reads this camera's events

    def _compute(self, camera_id: str, targets: list[Zone]) -> int | None:
        geometric = [z for z in targets if z.kind != "frame"]  # a whole-frame zone has no crossing or entry events
        if not geometric:
            return 0
        for z in geometric:
            zones.clear_events(self.db, z.id)
        count = self._recompute(camera_id, geometric)
        for z in geometric:
            self.bus.publish("zone", {"zone_id": z.id, "camera_id": camera_id, "events": count})
        return count

    def save(self, zone: Zone, fact_id: str | None = None) -> SaveResult:
        saved = zones.save(self.db, zone, fact_id)
        result = SaveResult(saved, self._compute(saved.camera_id, [saved]))
        self._recomputed(saved.camera_id)
        return result

    def _recomputed(self, camera_id: str) -> None:
        if self.on_recomputed is not None:
            try:
                self.on_recomputed(camera_id)
            except Exception:  # noqa: BLE001 - a follow-up step must never fail a zone save
                log.exception("post-recompute hook failed for %s", camera_id)

    def recompute_zone(self, zone_id: str) -> int | None:
        zone = zones.get_zone(self.db, zone_id)
        count = self._compute(zone.camera_id, [zone])
        self._recomputed(zone.camera_id)
        return count

    def recompute_camera(self, camera_id: str) -> int | None:
        """Called when a camera finishes ingesting: zones drawn earlier get their events now."""
        targets = zones.list_zones(self.db, camera_id)
        return self._compute(camera_id, targets) if targets else None

    def delete(self, zone_id: str) -> None:
        fact_id = zones.fact_id_of(self.db, zone_id)
        zones.delete(self.db, zone_id)
        if fact_id and self.kb is not None:  # keep the fact, drop only its pointer to the removed zone
            try:
                fact = self.kb.get(fact_id)
            except FactNotFound:
                return
            if fact.binding.get("zone_id") == zone_id:
                self.kb.update(fact_id, binding={k: v for k, v in fact.binding.items() if k != "zone_id"})
        self.bus.publish("zone", {"zone_id": zone_id, "deleted": True})
