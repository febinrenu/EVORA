import json

import pytest
from contracts.models import QueryPlan, Referent, Target, TimeWindow, Zone
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.core import cameras as cams
from evora.core import zones
from tests.alerts.conftest import T0, PlanResult
from tests.memory.conftest import FakeEmbedder

TEXT = "tell me when someone enters the main gate after 8pm"
PLAN = QueryPlan(
    intent="standing", action="enter", time=TimeWindow(tod_after="20:00"),
    targets=[Target(noun="person", cls=["person"], attributes=[], embed_text="a photo of a person")],
    place=Referent(text="main gate", role="place"), unresolved=[Referent(text="main gate", role="place")],
)


class StubPlanner:
    async def plan(self, text, cameras, reference, tz):
        if text == "gibberish":
            from evora.query.planner import PlanningError

            raise PlanningError("could not plan this question")
        return PlanResult(PLAN)


@pytest.fixture()
def client(tmp_path, monkeypatch, sample_mp4):
    monkeypatch.setenv("evora_WORKSPACE", "alerts")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object(), embedder=FakeEmbedder()))
    c.app.state.ctx.compiler.planner = StubPlanner()
    with open(sample_mp4, "rb") as fh:
        c.post("/api/cameras", files=[("files", ("gate.mp4", fh))])
    return c


def world(client):
    """A remembered gate line on cam_01 and one person crossing it at 20:30 UTC (the workspace time zone)."""
    ctx = client.app.state.ctx
    cams.update_camera(ctx.db, "cam_01", t0=T0)
    ctx.zones._recompute = lambda camera_id, zs: 0
    zones.save(ctx.db, Zone(id="z_gate", camera_id="cam_01", kind="line", points=[(0.1, 0.7), (0.9, 0.7)]))
    ctx.memory.kb.create("place", "main gate", {"camera_id": "cam_01", "zone_id": "z_gate"}, "clarification")
    t = T0 + 11.5 * 3600  # 20:30
    with ctx.db.write() as c:
        c.execute(
            "INSERT INTO tracks(id,camera_id,cls,t_start,t_end,n_obs) VALUES('cam_01:t1','cam_01','person',?,?,5)",
            (t - 5, t + 5),
        )
        c.execute(
            "INSERT INTO events(id,camera_id,track_id,kind,zone_id,t,payload) "
            "VALUES('e1','cam_01','cam_01:t1','cross_line','z_gate',?,?)",
            (t, json.dumps({"direction": "a_to_b"})),
        )
    return t


def test_create_a_watch_and_it_reports_what_already_happened(client):
    t = world(client)
    r = client.post("/api/standing", json={"text": TEXT})
    assert r.status_code == 200
    sq = r.json()
    assert sq["active"] and sq["rule"]["events"] == ["cross_line"] and sq["rule"]["tod_after"] == "20:00"
    alerts = client.get("/api/alerts").json()
    assert len(alerts) == 1 and alerts[0]["standing_query_id"] == sq["id"] and alerts[0]["t"] == t
    assert "found in earlier footage" in alerts[0]["evidence"]["why"]
    assert [w["id"] for w in client.get("/api/standing").json()] == [sq["id"]]


def test_pause_and_resume_a_watch(client):
    world(client)
    sq = client.post("/api/standing", json={"text": TEXT}).json()
    off = client.patch(f"/api/standing/{sq['id']}", json={"active": False}).json()
    assert off["active"] is False
    assert client.patch(f"/api/standing/{sq['id']}", json={"active": "no"}).status_code == 422
    assert client.patch("/api/standing/sq_nope", json={"active": True}).status_code == 404


def test_acknowledge_an_alert(client):
    world(client)
    client.post("/api/standing", json={"text": TEXT})
    alert = client.get("/api/alerts").json()[0]
    assert client.post(f"/api/alerts/{alert['id']}/ack").json()["acknowledged"] is True
    assert client.get("/api/alerts", params={"acknowledged": False}).json() == []
    assert len(client.get("/api/alerts", params={"acknowledged": True}).json()) == 1
    assert client.post("/api/alerts/al_nope/ack").status_code == 404


def test_the_alert_thumbnail_route_resolves(client):
    world(client)
    client.post("/api/standing", json={"text": TEXT})
    client.post("/api/settings", json={"blur_faces": False})
    alert = client.get("/api/alerts").json()[0]
    assert client.get(alert["evidence"]["thumb_url"]).status_code == 200


def test_an_unknown_place_returns_the_clarify_card_and_the_retry_works(client):
    client.app.state.ctx.zones._recompute = lambda camera_id, zs: 0
    first = client.post("/api/standing", json={"text": TEXT})
    assert first.status_code == 409
    card = first.json()["clarify"]
    assert card["kind"] == "choose_camera" and card["referent"]["text"] == "main gate"
    assert client.get("/api/standing").json() == [], "nothing is created until the place is known"
    answered = client.post("/api/clarify", json={"query_id": card["query_id"], "camera_id": "cam_01"})
    assert answered.status_code == 200
    second = client.post("/api/standing", json={"text": TEXT})
    assert second.status_code == 200 and second.json()["rule"]["camera_ids"] == ["cam_01"]


@pytest.mark.parametrize(
    "body,status", [({"text": ""}, 422), ({"text": "x" * 501}, 422), ({"text": "gibberish"}, 422)]
)
def test_bad_requests(client, body, status):
    assert client.post("/api/standing", json=body).status_code == status


def test_watches_and_alerts_survive_a_restart(tmp_path, monkeypatch, sample_mp4):
    from evora.core.db import close_all

    monkeypatch.setenv("evora_WORKSPACE", "persist")

    def start():
        c = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object(), embedder=FakeEmbedder()))
        c.app.state.ctx.compiler.planner = StubPlanner()
        return c

    first = start()
    with open(sample_mp4, "rb") as fh:
        first.post("/api/cameras", files=[("files", ("gate.mp4", fh))])
    world(first)
    first.post("/api/standing", json={"text": TEXT})
    first.close()
    close_all()
    second = start()
    assert len(second.get("/api/standing").json()) == 1 and len(second.get("/api/alerts").json()) == 1


def test_mock_mode_serves_fixtures(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "mock")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", mock=True))
    assert c.post("/api/standing", json={"text": "anything"}).json()["text"] == "anything"
    assert c.get("/api/alerts").json()[0]["id"] == "al_001"
    assert c.post("/api/alerts/al_9/ack").json()["acknowledged"] is True
