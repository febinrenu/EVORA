import time

import pytest
from contracts.models import ClarifyResponse, QueryPlan, Referent
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.memory.resolve import Unknown
from tests.memory.conftest import FakeEmbedder

LINE = {"id": "gate_line", "camera_id": "cam_01", "kind": "line", "points": [[0.1, 0.7], [0.9, 0.65]]}


class Recompute:
    def __init__(self):
        self.calls = []
        self.result = 4

    def __call__(self, camera_id, zones):
        self.calls.append((camera_id, [z.id for z in zones]))
        return self.result


@pytest.fixture()
def client(tmp_path, monkeypatch, sample_mp4):
    monkeypatch.setenv("evora_WORKSPACE", "zones")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object(), embedder=FakeEmbedder()))
    c.rec = Recompute()
    c.app.state.ctx.zones._recompute = c.rec
    with open(sample_mp4, "rb") as fh:
        assert c.post("/api/cameras", files=[("files", ("gate.mp4", fh))]).status_code == 200
    return c


def test_draw_a_zone_and_see_its_event_count(client):
    r = client.post("/api/zones", json=LINE)
    assert r.status_code == 200 and r.json()["id"] == "gate_line" and r.headers["x-evora-events"] == "4"
    assert client.rec.calls == [("cam_01", ["gate_line"])]
    assert [z["id"] for z in client.get("/api/zones").json()] == ["gate_line"]
    assert client.get("/api/zones", params={"camera_id": "cam_02"}).json() == []


def test_perception_not_ready_is_reported_as_pending(client):
    client.rec.result = None
    r = client.post("/api/zones", json=LINE)
    assert r.status_code == 200 and r.headers["x-evora-events"] == "pending"


def test_redraw_replaces_the_zone(client):
    client.post("/api/zones", json=LINE)
    redraw = {**LINE, "points": [[0.2, 0.2], [0.8, 0.8]], "direction": "a_to_b"}
    assert client.post("/api/zones", json=redraw).status_code == 200
    zones = client.get("/api/zones").json()
    assert len(zones) == 1 and zones[0]["points"] == [[0.2, 0.2], [0.8, 0.8]] and zones[0]["direction"] == "a_to_b"
    assert len(client.rec.calls) == 2


@pytest.mark.parametrize(
    "patch,status",
    [
        ({"camera_id": "cam_99"}, 404),
        ({"points": [[0.1, 0.1]]}, 422),
        ({"points": [[0.1, 1.4], [0.2, 0.2]]}, 422),
        ({"id": "../etc"}, 422),
        ({"kind": "circle"}, 422),
    ],
)
def test_bad_zones_are_rejected(client, patch, status):
    assert client.post("/api/zones", json={**LINE, **patch}).status_code == status
    assert client.get("/api/zones").json() == []


def test_delete(client):
    client.post("/api/zones", json=LINE)
    assert client.delete("/api/zones/gate_line").json() == {"ok": True}
    assert client.get("/api/zones").json() == []
    assert client.delete("/api/zones/gate_line").status_code == 404


def test_answering_a_clarify_with_a_line_recomputes_its_events_once(client):
    ctx = client.app.state.ctx
    ref = Referent(text="main gate", role="place")
    ctx.memory.ask("q1", "did a car pass the main gate", QueryPlan(intent="exists"), ref, Unknown())
    zone = {"id": "client", "camera_id": "cam_01", "kind": "line", "points": [[0.1, 0.7], [0.9, 0.65]]}
    out = ctx.memory.apply(ClarifyResponse(query_id="q1", camera_id="cam_01", zone=zone))
    assert client.rec.calls == [("cam_01", [out.fact.binding["zone_id"]])]


def test_a_finished_ingest_computes_events_for_zones_drawn_earlier(client):
    client.post("/api/zones", json=LINE)
    client.rec.calls.clear()
    client.post("/api/ingest", json={"camera_ids": ["cam_01"], "layers": ["L0"]})
    end = time.time() + 5
    while time.time() < end and not client.rec.calls:
        time.sleep(0.05)
    assert client.rec.calls == [("cam_01", ["gate_line"])]
