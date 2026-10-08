import json

import pytest
from contracts.models import Zone

from evora.core import cameras as cams
from evora.core import workspace as wsmod
from evora.core import zones
from evora.core.bus import Bus
from evora.core.db import open_db
from evora.core.vectors import open_store
from evora.core.zone_service import ZoneService
from evora.memory.kb import KnowledgeBase
from tests.memory.conftest import FakeEmbedder


@pytest.fixture()
def db(tmp_path):
    d = open_db(wsmod.create("z", tmp_path / "ws").db_path)
    for name in ("Gate", "Lobby"):
        cams.insert_camera(d, name=name, kind="file", source_uri=f"/x/{name}.mp4", t0=1.0, t0_source="manual")
    return d


def make_kb(db):
    return KnowledgeBase(db, open_store(db.path.parent / "vectors"), FakeEmbedder())


def line(zid="z1", cam="cam_01", direction="any"):
    return Zone(id=zid, camera_id=cam, kind="line", points=[(0.1, 0.7), (0.9, 0.6)], direction=direction)


def poly(zid="p1", cam="cam_01"):
    return Zone(id=zid, camera_id=cam, kind="polygon", points=[(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])


@pytest.mark.parametrize(
    "zone",
    [
        Zone(id="a", camera_id="cam_01", kind="line", points=[(0, 0)]),
        Zone(id="a", camera_id="cam_01", kind="polygon", points=[(0, 0), (1, 1)]),
        Zone(id="a", camera_id="cam_01", kind="line", points=[(0, 0), (1.2, 1)]),
        Zone(id="a", camera_id="cam_01", kind="frame", points=[(0, 0)]),
        Zone(id="../x", camera_id="cam_01", kind="frame"),
        Zone(id="a b", camera_id="cam_01", kind="frame"),
    ],
)
def test_invalid_geometry_is_rejected(db, zone):
    with pytest.raises(zones.ZoneError):
        zones.save(db, zone)
    assert zones.list_zones(db) == []


def test_save_list_get_roundtrip(db):
    zones.save(db, line())
    zones.save(db, poly())
    zones.save(db, Zone(id="f1", camera_id="cam_02", kind="frame"))
    assert [z.id for z in zones.list_zones(db)] == ["z1", "p1", "f1"]
    assert [z.id for z in zones.list_zones(db, "cam_02")] == ["f1"]
    got = zones.get_zone(db, "z1")
    assert got.points == [(0.1, 0.7), (0.9, 0.6)] and got.kind == "line"
    with pytest.raises(zones.ZoneNotFound):
        zones.get_zone(db, "nope")


def test_unknown_camera_and_camera_change_are_refused(db):
    with pytest.raises(cams.CameraNotFound):
        zones.save(db, line(cam="cam_99"))
    zones.save(db, line())
    with pytest.raises(zones.ZoneError):
        zones.save(db, line(cam="cam_02"))


def test_redrawing_replaces_geometry_and_keeps_the_fact_link(db):
    fact = make_kb(db).create("place", "main gate", {})
    zones.save(db, line(), fact.id)
    zones.save(db, Zone(id="z1", camera_id="cam_01", kind="line", points=[(0.2, 0.2), (0.8, 0.8)], direction="a_to_b"))
    z = zones.get_zone(db, "z1")
    assert z.points == [(0.2, 0.2), (0.8, 0.8)] and z.direction == "a_to_b"
    assert zones.fact_id_of(db, "z1") == fact.id


class Recorder:
    def __init__(self, result=3, fail=False):
        self.calls, self.result, self.fail = [], result, fail

    def __call__(self, camera_id, targets):
        self.calls.append((camera_id, [z.id for z in targets]))
        if self.fail:
            raise RuntimeError("model missing")
        return self.result


def service(db, rec, bus=None, kb=None):
    return ZoneService(db, bus or Bus(), kb, rec)


def add_events(db, zone_id, n=2):
    with db.write() as c:
        for i in range(n):
            c.execute(
                "INSERT INTO events(id,camera_id,track_id,kind,zone_id,t,payload) VALUES(?,?,?,?,?,?,?)",
                (f"e_{zone_id}_{i}", "cam_01", "t", "cross_line", zone_id, float(i), json.dumps({"direction": "a_to_b"})),
            )


def count_events(db, zone_id):
    with db.read() as c:
        return c.execute("SELECT count(*) FROM events WHERE zone_id=?", (zone_id,)).fetchone()[0]


def test_saving_clears_stale_events_then_recomputes_once(db):
    rec = Recorder(result=5)
    svc = service(db, rec)
    zones.save(db, line())
    add_events(db, "z1")
    res = svc.save(line())
    assert res.events == 5 and rec.calls == [("cam_01", ["z1"])]
    assert count_events(db, "z1") == 0, "stale events are removed before perception writes the new ones"


def test_frame_zone_never_asks_perception(db):
    rec = Recorder()
    res = service(db, rec).save(Zone(id="f", camera_id="cam_01", kind="frame"))
    assert res.events == 0 and rec.calls == []


def test_missing_perception_means_pending_and_the_zone_is_still_saved(db, monkeypatch):
    from evora.core import perception_adapter

    monkeypatch.setattr(perception_adapter, "MODULES", ())  # M2's real module exists now; simulate its absence
    assert service(db, Recorder(result=None)).save(line()).events is None
    assert perception_adapter.recompute_events("cam_01", [line()]) is None, "no perception installed"
    res = ZoneService(db, Bus(), None, perception_adapter.recompute_events).save(poly())
    assert res.events is None and zones.get_zone(db, "p1").kind == "polygon"


def test_adapter_swallows_a_raising_recompute(monkeypatch):
    from evora.core import perception_adapter

    monkeypatch.setattr(perception_adapter, "_find", lambda name: Recorder(fail=True))
    assert perception_adapter.recompute_events("cam_01", [line()]) is None


def test_recompute_camera_covers_only_that_cameras_geometric_zones(db):
    rec = Recorder()
    svc = service(db, rec)
    for z in (line("a"), poly("b"), Zone(id="c", camera_id="cam_01", kind="frame"), line("d", cam="cam_02")):
        zones.save(db, z)
    svc.recompute_camera("cam_01")
    assert rec.calls == [("cam_01", ["a", "b"])]
    assert svc.recompute_camera("cam_03") is None


def test_the_ui_is_told_how_many_events_were_found(db):
    import asyncio

    bus = Bus()

    async def scenario():
        sub = bus.subscribe()
        service(db, Recorder(result=7), bus).save(line())
        msg = await asyncio.wait_for(sub.queue.get(), 2)
        return json.loads(msg["data"])

    assert asyncio.run(scenario()) == {"kind": "zone", "zone_id": "z1", "camera_id": "cam_01", "events": 7}


def test_delete_removes_events_and_the_facts_pointer_but_not_the_fact(db):
    kb = make_kb(db)
    fact = kb.create("place", "main gate", {"camera_id": "cam_01", "zone_id": "z1"})
    svc = service(db, Recorder(), kb=kb)
    zones.save(db, line(), fact.id)
    add_events(db, "z1")
    svc.delete("z1")
    assert count_events(db, "z1") == 0
    with pytest.raises(zones.ZoneNotFound):
        zones.get_zone(db, "z1")
    assert kb.get(fact.id).binding == {"camera_id": "cam_01"}
    with pytest.raises(zones.ZoneNotFound):
        svc.delete("z1")


def test_zones_saved_in_the_same_clock_tick_keep_their_order(db, monkeypatch):
    monkeypatch.setattr(zones.time, "time", lambda: 1000.0)  # every save gets the same timestamp
    for zid in ("zz", "aa", "mm"):
        zones.save(db, Zone(id=zid, camera_id="cam_01", kind="frame"))
    assert [z.id for z in zones.list_zones(db)] == ["zz", "aa", "mm"], "insertion order, not alphabetical"
    zones.save(db, Zone(id="zz", camera_id="cam_01", kind="frame"))  # redrawing keeps the zone's place
    assert [z.id for z in zones.list_zones(db)] == ["zz", "aa", "mm"]
