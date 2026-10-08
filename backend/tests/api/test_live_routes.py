import asyncio
import json
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evora.api.app import create_app
from tests.live.test_restream import Spawner, free_port


@pytest.fixture()
def client(tmp_path, monkeypatch, sample_mp4):
    monkeypatch.setenv("evora_WORKSPACE", "livert")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))
    port = free_port()
    binary = tmp_path / "mediamtx.exe"
    binary.write_bytes(b"")
    live = c.app.state.ctx.live
    live.port, live._configured, live._spawn, live._ffmpeg = port, str(binary), Spawner(port), "ffmpeg"
    with open(sample_mp4, "rb") as fh:
        assert c.post("/api/cameras", files=[("files", ("gate.mp4", fh))]).status_code == 200
    c.post("/api/cameras", json={"uri": "rtsp://10.0.0.2/live", "name": "Door"})
    c.live = live
    yield c
    live.shutdown()


def test_status_before_anything_runs(client):
    body = client.get("/api/live").json()
    assert body["streams"] == [] and body["server"]["ready"] is False and body["server"]["port"] == client.live.port


def test_start_and_stop_a_replay(client):
    r = client.post("/api/live/replay", json={"camera_ids": ["cam_01"], "speed": 2})
    assert r.status_code == 200
    stream = r.json()["streams"][0]
    assert stream["url"] == f"rtsp://127.0.0.1:{client.live.port}/cam_01" and stream["speed"] == 2
    assert r.json()["server"]["ready"] is True
    stopped = client.post("/api/live/replay/stop", json={"camera_ids": ["cam_01"]}).json()
    assert stopped["streams"][0]["state"] == "stopped"
    assert client.post("/api/live/replay/stop").json()["server"]["ready"] is False


@pytest.mark.parametrize(
    "body,status",
    [
        ({"camera_ids": ["cam_99"]}, 404),
        ({"camera_ids": ["cam_02"]}, 409),
        ({"camera_ids": ["cam_01"], "speed": 50}, 422),
        ({"camera_ids": []}, 422),
        ({"camera_ids": ["../x"]}, 422),
        ({}, 422),
    ],
)
def test_errors(client, body, status):
    assert client.post("/api/live/replay", json=body).status_code == status


def test_missing_mediamtx_is_a_503_with_the_install_command(client, tmp_path):
    client.live._configured = str(tmp_path / "gone.exe")
    r = client.post("/api/live/replay", json={"camera_ids": ["cam_01"]})
    assert r.status_code == 503 and "winget install bluenviron.mediamtx" in r.json()["detail"]


def test_a_busy_port_is_a_409(client):
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", client.live.port))
    blocker.listen(1)
    try:
        assert client.post("/api/live/replay", json={"camera_ids": ["cam_01"]}).status_code == 409
    finally:
        blocker.close()


def test_state_changes_reach_the_event_stream(client):
    ctx = client.app.state.ctx

    async def scenario():
        sub = ctx.bus.subscribe()
        await asyncio.to_thread(client.post, "/api/live/replay", json={"camera_ids": ["cam_01"]})
        seen = []
        while True:
            note = json.loads((await asyncio.wait_for(sub.queue.get(), 5))["data"])
            if note.get("kind") == "live":
                seen.append((note["camera_id"], note["state"]))
                if seen[-1][1] == "running":
                    return seen

    assert ("cam_01", "running") in asyncio.run(scenario())


def test_mock_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "livemock")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", mock=True))
    assert c.get("/api/live").json()["streams"][0]["state"] == "running"
    assert c.post("/api/live/replay", json={"camera_ids": ["cam_01"]}).status_code == 200
    assert c.post("/api/live/replay/stop").json()["streams"] == []
    assert Path(c.app.state.ctx.ws.root).exists()
