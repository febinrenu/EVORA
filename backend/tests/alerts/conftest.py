import json
from dataclasses import dataclass, field

import pytest
from contracts.models import QueryPlan, Referent, Target, TimeWindow, Zone

from evora.alerts.compiler import StandingCompiler
from evora.alerts.engine import AlertEngine
from evora.core import cameras as cams
from evora.core import workspace as wsmod
from evora.core import zones
from evora.core.bus import Bus
from evora.core.db import close_all, open_db
from evora.core.vectors import open_store
from evora.memory.clarify import Clarifier
from evora.memory.kb import KnowledgeBase
from evora.memory.resolve import Resolver
from evora.memory.service import MemoryService
from tests.memory.conftest import FakeEmbedder

T0 = 1791450000.0  # 2026-10-08 09:00:00 UTC; the workspace tz is UTC unless a test sets meta.tz


@dataclass
class PlanResult:
    plan: QueryPlan
    notes: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class FakePlanner:
    """Returns the plan registered for a text; records what it was asked."""

    def __init__(self):
        self.plans: dict[str, QueryPlan] = {}
        self.asked: list[str] = []

    def add(self, text: str, **kw) -> None:
        defaults = {
            "intent": "standing",
            "targets": [Target(noun="person", cls=["person"], attributes=[], embed_text="a photo of a person")],
        }
        self.plans[text] = QueryPlan(**{**defaults, **kw})

    async def plan(self, text, cameras, reference, tz):
        self.asked.append(text)
        return PlanResult(self.plans[text])


class Env:
    def __init__(self, root):
        self.ws = wsmod.create("alerts", root)
        self.db = open_db(self.ws.db_path)
        self.bus = Bus()
        self.kb = KnowledgeBase(self.db, open_store(self.ws.vectors_dir), FakeEmbedder())
        resolver = Resolver(self.kb, lambda: cams.list_cameras(self.db), None)
        self.memory = MemoryService(self.db, self.kb, resolver, Clarifier(self.db, self.kb))
        self.planner = FakePlanner()
        self.compiler = StandingCompiler(self.db, self.planner, self.memory, default_cooldown_s=30.0)
        self.pushed: list[tuple[str, str]] = []
        self.engine = AlertEngine(self.db, self.bus, self.kb, notifier=None)
        for name in ("Gate", "Lobby"):
            cams.insert_camera(
                self.db, name=name, kind="file", source_uri=f"/x/{name}.mp4", t0=T0, t0_source="manual", duration_s=600.0
            )

    # --- fixtures of the world ---
    def line(self, zone_id="z_gate", camera="cam_01", direction="any"):
        zone = Zone(id=zone_id, camera_id=camera, kind="line", points=[(0.1, 0.7), (0.9, 0.7)], direction=direction)
        return zones.save(self.db, zone)

    def polygon(self, zone_id="z_yard", camera="cam_01"):
        zone = Zone(id=zone_id, camera_id=camera, kind="polygon", points=[(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])
        return zones.save(self.db, zone)

    def remember_gate(self, zone_id="z_gate", camera="cam_01"):
        return self.kb.create("place", "main gate", {"camera_id": camera, "zone_id": zone_id}, "clarification")

    def track(self, track_id="cam_01:t1", camera="cam_01", cls="person", global_id=None):
        with self.db.write() as c:
            c.execute(
                "INSERT OR REPLACE INTO tracks(id,camera_id,cls,t_start,t_end,n_obs,global_id) VALUES(?,?,?,?,?,5,?)",
                (track_id, camera, cls, T0, T0 + 100, global_id),
            )
            c.execute(
                "INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,?,?,?,?,0.9)",
                (track_id, T0 + 10, 0.3, 0.4, 0.4, 0.8),
            )

    def event(
        self, event_id="e1", kind="cross_line", zone="z_gate", track="cam_01:t1", camera="cam_01", t=T0 + 10,
        direction="a_to_b",
    ):
        row = {
            "id": event_id, "camera_id": camera, "track_id": track, "kind": kind, "zone_id": zone, "t": t,
            "payload": json.dumps({"direction": direction} if direction else {}),
        }
        with self.db.write() as c:
            c.execute(
                "INSERT OR REPLACE INTO events(id,camera_id,track_id,kind,zone_id,t,payload) VALUES(?,?,?,?,?,?,?)",
                (row["id"], camera, track, kind, zone, t, row["payload"]),
            )
        return row


def place_plan(**kw):
    return {"place": Referent(text="main gate", role="place"), "unresolved": [Referent(text="main gate", role="place")], **kw}


@pytest.fixture()
def env(tmp_path):
    e = Env(tmp_path / "ws")
    yield e
    close_all()


__all__ = ["Env", "FakePlanner", "T0", "TimeWindow", "place_plan"]
