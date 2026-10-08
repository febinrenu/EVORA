import json

import pytest
from fastapi.testclient import TestClient

from evora.api.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "routes")
    return TestClient(create_app(workspaces_root=tmp_path / "ws", mock=True))  # fixture streams, as the UI uses in mock mode


def sse_types(resp) -> list[str]:
    return [line.split(":", 1)[1].strip() for line in resp.text.splitlines() if line.startswith("event:")]


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] and body["workspace"] == "routes"


def test_query_sse_order(client):
    r = client.post("/api/query", json={"text": "person in red", "session_id": "s"})
    types = sse_types(r)
    assert types[0] == "plan" and types[-1] == "done"
    assert types.index("answer") < types.index("verified")
    assert set(types) <= {"plan", "evidence", "answer", "verified", "done"}


def test_query_clarify_stops_after_clarify(client):
    r = client.post("/api/query", json={"text": "who entered the back entrance", "session_id": "s"})
    assert sse_types(r) == ["plan", "clarify"]


def test_empty_query_rejected(client):
    assert client.post("/api/query", json={"text": " ", "session_id": "s"}).status_code == 422


def test_clarify_resumes(client):
    r = client.post("/api/clarify", json={"query_id": "q_002", "camera_id": "cam_01"})
    assert sse_types(r)[-1] == "done"


def test_zone_validation(client):
    ok = client.post("/api/zones", json={"id": "z", "camera_id": "cam_01", "kind": "line"})
    assert ok.status_code == 200
    bad = client.post("/api/zones", json={"id": "z", "camera_id": "cam_01", "kind": "circle"})
    assert bad.status_code == 422


def test_remaining_routes_respond(client):
    assert client.get("/api/globals/g_0007/path").json()[0]["camera_id"] == "cam_01"
    assert len(client.get("/api/tracks/cam_01:t000001/similar", params={"k": 2}).json()) == 2
    assert client.get("/api/alerts").json()[0]["id"] == "al_001"
    assert client.post("/api/alerts/al_001/ack").json()["acknowledged"] is True
    assert client.get("/api/standing").json()[0]["active"] is True
    assert client.post("/api/evidence/ev_001/pack").content[:2] == b"PK"
    assert client.post("/api/settings", json={"onprem": True}).json()["onprem"] is True
    assert json.loads(client.get("/api/report").text) == {"eval": None, "ablations": None}


def test_cors_allows_ui_origin(client):
    r = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
