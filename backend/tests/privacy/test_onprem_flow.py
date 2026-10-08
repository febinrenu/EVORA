"""On-prem mode end to end: which connections does the real app try, and what does it refuse."""
import json
import os
import socket

import pytest
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.core import cameras as cams
from evora.core import privacy_guard
from evora.core.db import close_all
from evora.core.privacy_guard import EgressBlocked, is_local
from evora.evidence import audit
from tests.api.fakes import parse_sse
from tests.memory.conftest import FakeEmbedder

UNPLANNABLE = "zzyzx plugh blorp {n}"  # the fast path cannot parse this, so the planner has to ask a model


def no_real_network():
    """Whatever the guard lets through must still never reach a real network from a test."""
    guard = privacy_guard.guard
    real_connect, real_connect_ex, real_lookup = guard._orig["connect"], guard._orig["connect_ex"], guard._orig["getaddrinfo"]

    def connect(sock, address, *a):
        if isinstance(address, tuple) and is_local(address[0]):
            return real_connect(sock, address, *a)
        raise ConnectionRefusedError("test: no real network")

    def connect_ex(sock, address, *a):
        if isinstance(address, tuple) and is_local(address[0]):
            return real_connect_ex(sock, address, *a)
        return 10061

    def lookup(host, *a, **k):
        if is_local(host):
            return real_lookup(host, *a, **k)
        raise socket.gaierror("test: no real DNS")

    guard._orig.update(connect=connect, connect_ex=connect_ex, getaddrinfo=lookup)


def non_local_attempts():
    return [a for a in privacy_guard.guard.attempts if not a[3]]


@pytest.fixture()
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "privacy")
    monkeypatch.setenv("GROQ_KEYS", "gsk_not_a_real_key")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # a closed local port: no real model is ever used
    monkeypatch.setattr("evora.core.config.REPO_ROOT", tmp_path)  # a developer's real .env must not leak in
    yield tmp_path
    os.environ.pop("evora_ONPREM", None)  # `start` sets it directly; it must not leak into other tests


def start(app_env, onprem: bool, **kw):
    if onprem:
        os.environ["evora_ONPREM"] = "1"
    else:
        os.environ.pop("evora_ONPREM", None)
    client = TestClient(create_app(workspaces_root=app_env / "ws", embedder=FakeEmbedder(), **kw))
    no_real_network()
    return client


def ask(client, n):
    return parse_sse(client.post("/api/query", json={"text": UNPLANNABLE.format(n=n), "session_id": "s"}).text)


def test_onprem_planning_only_ever_touches_loopback(app_env):
    client = start(app_env, onprem=True)
    ask(client, 1)
    attempts = list(privacy_guard.guard.attempts)
    assert attempts, "the planner did try to reach a model"
    assert all(a[3] for a in attempts), f"a non-loopback attempt was made: {non_local_attempts()}"
    assert any(a[1] == "127.0.0.1" and a[2] == 9 for a in attempts), "it went to the local model"
    assert not any("groq" in a[1] for a in attempts) and privacy_guard.guard.blocked == 0


def test_switching_onprem_off_lets_planning_reach_the_cloud_and_back_on_blocks_it(app_env):
    client = start(app_env, onprem=True)
    assert client.post("/api/settings", json={"onprem": False}).json()["onprem"] is False
    ask(client, 2)
    assert any("groq" in a[1] for a in privacy_guard.guard.attempts), "off: the cloud planner is tried (no real network here)"
    assert privacy_guard.guard.blocked == 0

    client.post("/api/settings", json={"onprem": True})
    with pytest.raises(EgressBlocked):
        socket.create_connection(("api.groq.com", 443), timeout=1)
    assert client.get("/api/health").json()["egress_blocked"] >= 1


def test_a_blocked_attempt_is_counted_on_the_health_badge(app_env):
    client = start(app_env, onprem=True)
    health = client.get("/api/health").json()
    assert health["onprem"] is True and health["egress_blocked"] == 0
    with pytest.raises(EgressBlocked):
        socket.create_connection(("93.184.216.34", 443), timeout=1)
    assert client.get("/api/health").json()["egress_blocked"] == 1


def test_footage_work_needs_no_network_at_all(app_env, sample_mp4):
    client = start(app_env, onprem=True)
    with open(sample_mp4, "rb") as fh:
        cam = client.post("/api/cameras", files=[("files", ("gate.mp4", fh))]).json()[0]
    client.post("/api/ingest", json={"camera_ids": [cam["id"]], "layers": ["L0"]})
    assert client.get(f"/api/cameras/{cam['id']}/frame", params={"t": cam["t0"] + 0.5}).status_code == 200
    assert privacy_guard.guard.blocked == 0 and non_local_attempts() == []


def test_voice_is_a_clear_503_in_onprem_mode(app_env):
    client = start(app_env, onprem=True)
    r = client.post("/api/voice", content=b"audio")
    assert r.status_code == 503 and "browser microphone" in r.json()["detail"]


# ---- settings -------------------------------------------------------------------------------------------------

def test_settings_are_validated(app_env):
    client = start(app_env, onprem=False)
    for bad in ({"onprem": "yes"}, {"blur_faces": 1}, {"reference_now": "noon"}, {"reference_now": True}):
        assert client.post("/api/settings", json=bad).status_code == 422, bad
    ok = client.post("/api/settings", json={"reference_now": 1791450000.5, "unknown": 1}).json()
    assert ok["reference_now"] == 1791450000.5 and "unknown" not in ok
    assert client.post("/api/settings", json={"reference_now": None}).json()["reference_now"] is None


def test_toggling_onprem_is_audited_and_announced(app_env):
    import asyncio

    client = start(app_env, onprem=False)
    ctx = client.app.state.ctx

    async def scenario():
        sub = ctx.bus.subscribe()
        await asyncio.to_thread(client.post, "/api/settings", json={"onprem": True})
        return json.loads((await asyncio.wait_for(sub.queue.get(), 2))["data"])

    assert asyncio.run(scenario()) == {"kind": "privacy", "onprem": True}
    assert audit.entries(ctx.db, "onprem_setting")[0]["detail"] == {"onprem": True}
    client.post("/api/settings", json={"onprem": True})  # no change, no new entry
    assert len(audit.entries(ctx.db, "onprem_setting")) == 1


def test_settings_survive_a_restart_and_the_environment_forces_onprem_on(app_env, monkeypatch):
    first = start(app_env, onprem=False)
    first.post("/api/settings", json={"onprem": True, "blur_faces": False})
    first.close()
    close_all()
    second = start(app_env, onprem=False)
    h = second.get("/api/health").json()
    assert h["onprem"] is True and h["blur"] == "off", "what the user chose is still chosen"
    second.post("/api/settings", json={"onprem": False})
    second.close()
    close_all()
    third = start(app_env, onprem=True)  # evora_ONPREM=1 wins over the stored choice
    assert third.get("/api/health").json()["onprem"] is True


# ---- health -------------------------------------------------------------------------------------------------

def add_cam(client, name, layers, status="ready"):
    ctx = client.app.state.ctx
    cam = cams.insert_camera(ctx.db, name=name, kind="file", source_uri=f"/x/{name}.mp4", t0=1.0, t0_source="manual")
    cams.add_layers(ctx.db, cam.id, layers)
    cams.set_status(ctx.db, cam.id, status)


def test_health_reports_layers_ready_on_every_working_camera(app_env):
    client = start(app_env, onprem=False)
    assert client.get("/api/health").json()["layers_ready"] == []
    add_cam(client, "a", ["L0", "L1"])
    add_cam(client, "b", ["L0"])
    add_cam(client, "broken", [], status="error")
    h = client.get("/api/health").json()
    assert h["layers_ready"] == ["L0"] and h["ok"] and h["workspace"] == "privacy" and h["version"] == "0.1.0"
    add_cam(client, "c", ["L0", "L1"])
    assert client.get("/api/health").json()["layers_ready"] == ["L0"]


def test_health_blur_state_is_checked_by_running_the_model(app_env):
    state = {"fn": None}
    client = start(app_env, onprem=False, blur_provider=lambda: state["fn"])
    ctx = client.app.state.ctx
    assert client.get("/api/health").json()["blur"] == "unavailable"

    def broken(_):
        raise RuntimeError("model file missing")

    for fn, expected in ((broken, "unavailable"), (lambda jpeg: jpeg, "applied")):
        state["fn"], ctx.media._probe = fn, None
        assert client.get("/api/health").json()["blur"] == expected
    client.post("/api/settings", json={"blur_faces": False})
    assert client.get("/api/health").json()["blur"] == "off"
