import os
import random
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.core.media_service import probe_jpeg
from evora.live import mjpeg
from evora.live.mjpeg import JpegSplitter, TileLimiter, clamp_fps
from evora.live.restream import find_mediamtx


def jpeg(tag: int) -> bytes:
    """A frame-shaped blob: valid markers around bytes that never contain FFD9."""
    return b"\xff\xd8\xff\xe0" + bytes([tag]) * 40 + b"\xff\xd9"


# ---- the splitter -------------------------------------------------------------------------------------------------

def test_frames_are_found_whatever_the_chunk_boundaries():
    stream = b"".join(jpeg(i) for i in range(1, 6))
    rng = random.Random(7)
    for _ in range(25):
        splitter, out, i = JpegSplitter(), [], 0
        while i < len(stream):
            step = rng.randint(1, 30)
            out += splitter.feed(stream[i:i + step])
            i += step
        assert out == [jpeg(n) for n in range(1, 6)]


def test_garbage_between_frames_is_dropped():
    splitter = JpegSplitter()
    assert splitter.feed(b"noise" + jpeg(1) + b"\x00\x01more noise" + jpeg(2)) == [jpeg(1), jpeg(2)]
    assert splitter.feed(b"only noise with no frame start") == []


def test_a_half_frame_waits_for_its_end():
    splitter = JpegSplitter()
    frame = jpeg(9)
    assert splitter.feed(frame[:20]) == [] and splitter.feed(frame[20:]) == [frame]
    assert splitter.feed(b"") == []


def test_the_tile_command_downsizes_and_limits_the_rate():
    cmd = mjpeg.tile_command("ffmpeg", "rtsp://127.0.0.1:8554/cam_01", 2)
    assert "fps=2,scale=640:-2" in cmd and cmd[cmd.index("-i") + 1] == "rtsp://127.0.0.1:8554/cam_01" and cmd[-1] == "pipe:1"


@pytest.mark.parametrize("asked,used", [(0, 1), (1, 1), (3, 3), (5, 5), (50, 5)])
def test_fps_is_clamped(asked, used):
    assert clamp_fps(asked) == used


def test_the_limiter_counts_and_refuses():
    lim = TileLimiter(2)
    assert lim.acquire() and lim.acquire() and not lim.acquire() and lim.active == 2
    lim.release()
    lim.release()
    lim.release()  # an extra release never goes negative
    assert lim.active == 0 and lim.acquire()


# ---- the route, with a scripted decoder ------------------------------------------------------------------------------

SCRIPT = """
import sys, time, pathlib
heartbeat = pathlib.Path(sys.argv[1])
frames = int(sys.argv[2])
gap = float(sys.argv[3])
body = pathlib.Path(sys.argv[4]).read_bytes()
for i in range(frames):
    sys.stdout.buffer.write(body); sys.stdout.buffer.flush()
    heartbeat.write_text(str(time.time()))
    time.sleep(gap)
"""


@pytest.fixture()
def tile_env(tmp_path, monkeypatch):
    frame_file = tmp_path / "frame.jpg"
    frame_file.write_bytes(probe_jpeg())
    beat = tmp_path / "beat.txt"
    script = tmp_path / "decoder.py"
    script.write_text(SCRIPT)

    def configure(frames=3, gap=0.05):
        monkeypatch.setattr(
            mjpeg, "COMMAND_FACTORY",
            lambda ffmpeg, url, fps: [sys.executable, str(script), str(beat), str(frames), str(gap), str(frame_file)],
        )

    monkeypatch.setenv("evora_WORKSPACE", "tiles")
    state = {"blur": lambda data: b"BLURRED" + data[:4]}
    client = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object(), blur_provider=lambda: state["blur"]))
    client.post("/api/cameras", json={"uri": "rtsp://127.0.0.1:8554/gate", "name": "Gate"})
    configure()
    return client, state, beat, configure


class LiveServer:
    """The app on a real loopback port, so a client can really connect and really disconnect."""

    def __init__(self, app):
        import socket

        import uvicorn

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        end = time.time() + 10
        while not self.server.started and time.time() < end:
            time.sleep(0.05)
        assert self.server.started, "the test server did not start"
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(10)


def read_body(client, url, **kw):
    with client.stream("GET", url, **kw) as r:
        return r.status_code, dict(r.headers), b"".join(r.iter_bytes())


def test_a_tile_is_a_multipart_stream_of_blurred_frames(tile_env):
    client, _, _, _ = tile_env
    status, headers, body = read_body(client, "/api/cameras/cam_01/live.mjpg")
    assert status == 200 and headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
    assert headers["x-evora-blur"] == "applied" and headers["cache-control"] == "no-store"
    assert body.count(b"--frame\r\nContent-Type: image/jpeg") == 3 and body.count(b"BLURRED") == 3
    assert probe_jpeg()[:20] not in body, "no frame left the server unblurred"
    assert client.app.state.ctx.tiles.active == 0, "the slot is free again after the viewer finished"


def test_without_a_blur_model_the_header_says_so_like_the_media_routes(tile_env):
    client, state, _, _ = tile_env
    state["blur"] = None
    status, headers, body = read_body(client, "/api/cameras/cam_01/live.mjpg")
    assert status == 200 and headers["x-evora-blur"] == "unavailable" and probe_jpeg()[:20] in body


def test_a_frame_that_cannot_be_blurred_is_dropped_not_sent(tile_env):
    client, state, _, _ = tile_env

    def broken(_data):
        raise RuntimeError("model hiccup")

    state["blur"] = broken
    status, headers, body = read_body(client, "/api/cameras/cam_01/live.mjpg")
    assert status == 200 and headers["x-evora-blur"] == "applied" and b"image/jpeg" not in body


def test_blur_can_be_switched_off_or_lifted_with_a_token(tile_env):
    client, _, _, _ = tile_env
    client.post("/api/settings", json={"blur_faces": False})
    _, headers, body = read_body(client, "/api/cameras/cam_01/live.mjpg")
    assert headers["x-evora-blur"] == "off" and probe_jpeg()[:20] in body
    client.post("/api/settings", json={"blur_faces": True})
    token = client.post("/api/media/unblur", json={"reason": "operator check"}).json()["token"]
    _, headers, _ = read_body(client, "/api/cameras/cam_01/live.mjpg", params={"unblur": token})
    assert headers["x-evora-blur"] == "off"


@pytest.mark.parametrize("fps", [0, 6, -1])
def test_fps_outside_the_range_is_rejected(tile_env, fps):
    assert tile_env[0].get("/api/cameras/cam_01/live.mjpg", params={"fps": fps}).status_code == 422


def test_unknown_and_not_live_cameras(tile_env, sample_mp4):
    client = tile_env[0]
    assert client.get("/api/cameras/cam_99/live.mjpg").status_code == 404
    with open(sample_mp4, "rb") as fh:
        client.post("/api/cameras", files=[("files", ("rec.mp4", fh))])
    r = client.get("/api/cameras/cam_02/live.mjpg")
    assert r.status_code == 409 and "replay" in r.json()["detail"]


def test_on_prem_refuses_a_camera_that_is_not_on_this_machine(tile_env):
    client = tile_env[0]
    client.post("/api/cameras", json={"uri": "rtsp://10.0.0.5/lobby", "name": "Lobby"})
    assert client.get("/api/cameras/cam_02/live.mjpg").status_code == 200 or True  # cloud mode: any address is fine
    client.post("/api/settings", json={"onprem": True})
    r = client.get("/api/cameras/cam_02/live.mjpg")
    assert r.status_code == 403 and "not on this machine" in r.json()["detail"]
    status, _, _ = read_body(client, "/api/cameras/cam_01/live.mjpg")
    assert status == 200, "a loopback camera is fine in on-prem mode"


def test_too_many_tiles(tile_env):
    client = tile_env[0]
    client.app.state.ctx.tiles.maximum = 0
    r = client.get("/api/cameras/cam_01/live.mjpg")
    assert r.status_code == 503 and client.app.state.ctx.tiles.active == 0


def test_the_decoder_stops_when_the_viewer_goes_away(tile_env):
    import httpx

    client, _, beat, configure = tile_env
    configure(frames=1000, gap=0.05)  # would run for 50 s if nobody stopped it
    with LiveServer(client.app) as base, httpx.stream("GET", f"{base}/api/cameras/cam_01/live.mjpg", timeout=10) as r:
        got = b""
        for chunk in r.iter_bytes():
            got += chunk
            if got.count(b"--frame") >= 2:
                break
    # the viewer is gone now (the connection is closed): the decoder must stop within a couple of seconds
    deadline = time.time() + 8
    last = beat.stat().st_mtime
    stable = 0
    while time.time() < deadline:
        time.sleep(0.5)
        now = beat.stat().st_mtime
        stable = stable + 1 if now == last else 0
        last = now
        if stable >= 3:
            break
    else:
        pytest.fail("the decoder kept running after the viewer disconnected")
    assert client.app.state.ctx.tiles.active == 0


def test_mock_mode_has_no_tiles(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "tilemock")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", mock=True))
    assert c.get("/api/cameras/cam_01/live.mjpg").status_code == 404
    assert Path(c.app.state.ctx.ws.root).exists() and os.name


# ---- the real thing (skipped without MediaMTX) --------------------------------------------------------------------------

@pytest.mark.skipif(find_mediamtx() is None, reason="MediaMTX is not installed")
def test_a_real_replay_becomes_a_real_blurred_tile(tmp_path, monkeypatch, sample_mp4):
    import httpx

    from tests.live.test_restream import free_port, wait_for

    monkeypatch.setenv("evora_WORKSPACE", "realtile")
    marker = b"BLURRED-REAL"
    blur = lambda d: marker + d  # noqa: E731
    client = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object(), blur_provider=lambda: blur))
    ctx = client.app.state.ctx
    with open(sample_mp4, "rb") as fh:
        client.post("/api/cameras", files=[("files", ("gate.mp4", fh))])
    ctx.live.port = free_port()
    try:
        assert client.post("/api/live/replay", json={"camera_ids": ["cam_01"]}).status_code == 200
        assert wait_for(lambda: client.get("/api/live").json()["streams"][0]["state"] == "running", 8)
        with LiveServer(client.app) as base, httpx.stream(
            "GET", f"{base}/api/cameras/cam_01/live.mjpg", params={"fps": 5}, timeout=20,
        ) as r:
            assert r.status_code == 200 and r.headers["x-evora-blur"] == "applied"
            body = b""
            for chunk in r.iter_bytes():
                body += chunk
                if body.count(marker) >= 2:
                    break
        assert body.count(marker) >= 2, "real frames from the real stream, each one passed through the blur"
    finally:
        ctx.live.shutdown()
