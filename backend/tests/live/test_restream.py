import json
import socket
import subprocess
import time
from pathlib import Path

import pytest

from evora.core import cameras as cams
from evora.core import workspace as wsmod
from evora.core.db import open_db
from evora.live import restream
from evora.live.restream import (
    LiveError,
    ReplayManager,
    ffmpeg_command,
    find_mediamtx,
    mediamtx_config,
    port_open,
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeProc:
    def __init__(self, args):
        self.args, self.code, self.killed = args, None, False
        self.on_close = None

    def poll(self):
        return self.code

    def die(self, code=1):
        self.code = code

    def terminate(self):
        self.killed = True
        self.code = 0 if self.code is None else self.code
        if self.on_close:
            self.on_close()

    kill = terminate

    def wait(self, timeout=None):
        return self.code


class Spawner:
    """Pretends to start MediaMTX (really listens on its port) and ffmpeg (records the command line)."""

    def __init__(self, port):
        self.port, self.ffmpeg, self.servers, self._sockets = port, [], [], []

    def __call__(self, args, log_file):
        proc = FakeProc(args)
        if Path(args[0]).name.startswith("mediamtx"):
            sock = socket.socket()
            sock.bind(("127.0.0.1", self.port))
            sock.listen(5)
            proc.on_close = sock.close
            self.servers.append(proc)
        else:
            self.ffmpeg.append(proc)
        return proc


@pytest.fixture()
def world(tmp_path):
    ws = wsmod.create("live", tmp_path / "ws")
    db = open_db(ws.db_path)
    for name in ("Gate", "Lobby"):
        src = tmp_path / f"{name}.mp4"
        src.write_bytes(b"fake video")
        cams.insert_camera(db, name=name, kind="file", source_uri=str(src), t0=1.0, t0_source="manual")
    cams.insert_camera(db, name="Door cam", kind="rtsp", source_uri="rtsp://10.0.0.9/live", t0=1.0, t0_source="live")
    binary = tmp_path / "mediamtx.exe"
    binary.write_bytes(b"")
    port = free_port()
    spawn = Spawner(port)
    events: list[tuple[str, str]] = []
    mgr = ReplayManager(
        db, ws.root / "live", mediamtx_path=str(binary), port=port, spawn=spawn, ffmpeg="ffmpeg",
        on_state=lambda c, s: events.append((c, s)),
    )
    mgr.spawner, mgr.events, mgr.binary = spawn, events, binary
    yield mgr
    mgr.shutdown()


def wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


# ---- command lines ---------------------------------------------------------------------------------------------

def test_copy_command_at_real_time():
    cmd = ffmpeg_command("ffmpeg", "C:/My Videos/it's here.mp4", "rtsp://127.0.0.1:8554/cam_01", 1.0, False)
    assert cmd[0] == "ffmpeg" and "-re" in cmd and "-readrate" not in cmd
    assert cmd[cmd.index("-i") + 1] == "C:/My Videos/it's here.mp4", "a path with spaces and quotes stays one argument"
    assert ["-c:v", "copy"] == cmd[cmd.index("-c:v"):cmd.index("-c:v") + 2] and cmd[-1] == "rtsp://127.0.0.1:8554/cam_01"
    assert cmd.index("-re") < cmd.index("-i") and "-stream_loop" in cmd and "-an" in cmd


@pytest.mark.parametrize("speed,expected", [(0.5, "0.5"), (4.0, "4"), (2.0, "2")])
def test_speed_uses_readrate(speed, expected):
    cmd = ffmpeg_command("ffmpeg", "a.mp4", "rtsp://x/y", speed, False)
    assert cmd[cmd.index("-readrate") + 1] == expected and "-re" not in cmd


def test_transcode_command():
    cmd = ffmpeg_command("ffmpeg", "a.mp4", "rtsp://x/y", 1.0, True)
    assert "libx264" in cmd and "zerolatency" in cmd and "copy" not in cmd


def test_mediamtx_config_is_loopback_only():
    text = mediamtx_config(8554)
    assert "rtspAddress: 127.0.0.1:8554" in text and "rtspTransports: [tcp]" in text
    for protocol in ("api", "metrics", "pprof", "playback", "rtmp", "hls", "webrtc", "srt"):
        assert f"{protocol}: false" in text
    assert "0.0.0.0" not in text and ":8554\n" in text


def test_find_mediamtx(tmp_path):
    assert find_mediamtx(str(tmp_path / "nope.exe")) is None
    present = tmp_path / "mediamtx.exe"
    present.write_bytes(b"")
    assert find_mediamtx(str(present)) == present


# ---- validation ------------------------------------------------------------------------------------------------

def test_everything_is_validated_before_anything_starts(world):
    with pytest.raises(LiveError) as err:
        world.start(["cam_01", "cam_99"])
    assert err.value.status == 404
    assert world.spawner.servers == [] and world.spawner.ffmpeg == [] and world.status()["streams"] == []


@pytest.mark.parametrize(
    "ids,speed,status",
    [(["cam_03"], None, 409), (["../etc"], None, 422), ([], None, 422), (["cam_01"], 0.1, 422), (["cam_01"], 9, 422)],
)
def test_bad_requests(world, ids, speed, status):
    with pytest.raises(LiveError) as err:
        world.start(ids, speed)
    assert err.value.status == status and world.spawner.ffmpeg == []


def test_a_missing_recording_is_reported(world):
    Path(cams.get_camera(world.db, "cam_01").source_uri).unlink()
    with pytest.raises(LiveError) as err:
        world.start(["cam_01"])
    assert err.value.status == 404 and "missing" in err.value.message


def test_missing_mediamtx_gives_the_install_hint(world):
    world._configured = str(world.binary.parent / "gone.exe")
    with pytest.raises(LiveError) as err:
        world.start(["cam_01"])
    assert err.value.status == 503 and "winget install bluenviron.mediamtx" in err.value.message


def test_a_busy_port_is_not_reused(world):
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", world.port))
    blocker.listen(1)
    try:
        with pytest.raises(LiveError) as err:
            world.start(["cam_01"])
        assert err.value.status == 409 and str(world.port) in err.value.message and world.spawner.servers == []
    finally:
        blocker.close()


def test_a_server_that_dies_at_once_reports_its_log(world, monkeypatch):
    log = world.live_dir / "mediamtx.log"

    def dying(args, log_file):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("ERR cannot bind\n")
        proc = FakeProc(args)
        proc.die(1)
        return proc

    world._spawn = dying
    with pytest.raises(LiveError) as err:
        world.start(["cam_01"])
    assert err.value.status == 502 and "cannot bind" in err.value.message


def test_too_many_streams(world):
    world.max_streams = 1
    world.start(["cam_01"])
    with pytest.raises(LiveError) as err:
        world.start(["cam_02"])
    assert err.value.status == 409


# ---- running streams -------------------------------------------------------------------------------------------

def test_start_runs_one_ffmpeg_per_camera_and_reports_urls(world):
    status = world.start(["cam_01", "cam_02"], 2.0)
    assert [s["url"] for s in status["streams"]] == [f"rtsp://127.0.0.1:{world.port}/cam_01", f"rtsp://127.0.0.1:{world.port}/cam_02"]
    assert status["server"]["ready"] is True and len(world.spawner.servers) == 1 and len(world.spawner.ffmpeg) == 2
    assert all("-readrate" in p.args for p in world.spawner.ffmpeg)
    assert wait_for(lambda: all(s["state"] == "running" for s in world.status()["streams"]), 4)


def test_starting_again_at_the_same_speed_changes_nothing_but_a_new_speed_restarts(world):
    world.start(["cam_01"], 1.0)
    world.start(["cam_01"], 1.0)
    assert len(world.spawner.ffmpeg) == 1 and len(world.spawner.servers) == 1
    world.start(["cam_01"], 2.0)
    assert len(world.spawner.ffmpeg) == 2 and world.spawner.ffmpeg[0].killed


def test_an_instant_failure_is_retried_once_as_a_transcode(world):
    world.start(["cam_01"])
    world.spawner.ffmpeg[0].die(1)
    assert wait_for(lambda: len(world.spawner.ffmpeg) == 2)
    assert "libx264" in world.spawner.ffmpeg[1].args and "copy" not in world.spawner.ffmpeg[1].args
    assert wait_for(lambda: world.status()["streams"][0]["transcoding"] is True)


def test_a_stream_that_keeps_crashing_ends_failed_with_the_reason(world, monkeypatch):
    monkeypatch.setattr(restream, "QUICK_EXIT_S", 0.0)  # treat every death as a late crash, not a codec problem
    world.start(["cam_01"])
    (world.live_dir / "cam_01.log").write_text("Connection refused\nbroken pipe\n")
    for expected in range(1, restream.MAX_RESTARTS + 2):
        assert wait_for(lambda n=expected: len(world.spawner.ffmpeg) == n, 5), f"launch {expected}"
        world.spawner.ffmpeg[-1].die(1)
    assert wait_for(lambda: world.status()["streams"][0]["state"] == "failed", 5)
    s = world.status()["streams"][0]
    assert s["restarts"] == restream.MAX_RESTARTS and "broken pipe" in s["error"]
    assert len(world.spawner.ffmpeg) == restream.MAX_RESTARTS + 1


def test_stop_kills_only_the_named_streams(world):
    world.start(["cam_01", "cam_02"])
    status = world.stop(["cam_01"])
    states = {s["camera_id"]: s["state"] for s in status["streams"]}
    assert states["cam_01"] == "stopped" and states["cam_02"] in ("starting", "running")
    assert world.spawner.ffmpeg[0].killed and not world.spawner.ffmpeg[1].killed
    assert not world.spawner.servers[0].killed, "the server stays up while another stream needs it"


def test_stopping_everything_also_stops_the_server_and_frees_the_port(world):
    world.start(["cam_01"])
    assert port_open(world.port)
    world.stop()
    assert world.spawner.servers[0].killed and not port_open(world.port)
    assert world.status()["server"]["ready"] is False


def test_shutdown_is_idempotent_and_kills_everything(world):
    world.start(["cam_01", "cam_02"])
    world.shutdown()
    world.shutdown()
    assert all(p.killed for p in world.spawner.ffmpeg) and world.spawner.servers[0].killed and not port_open(world.port)


def test_state_changes_are_announced(world):
    world.start(["cam_01"])
    assert wait_for(lambda: ("cam_01", "running") in world.events, 4)
    world.stop(["cam_01"])
    assert world.events[-1] == ("cam_01", "stopped")


def test_restarting_after_a_stop_works(world):
    world.start(["cam_01"])
    world.stop(["cam_01"])
    world.start(["cam_01"])
    assert len(world.spawner.ffmpeg) == 2 and world.status()["streams"][0]["state"] in ("starting", "running")


# ---- the real thing (skipped without MediaMTX) -----------------------------------------------------------------

@pytest.mark.skipif(find_mediamtx() is None, reason="MediaMTX is not installed")
def test_a_real_stream_can_be_read_back(tmp_path, sample_mp4):
    ws = wsmod.create("real", tmp_path / "ws")
    db = open_db(ws.db_path)
    cams.insert_camera(db, name="Gate", kind="file", source_uri=str(sample_mp4), t0=1.0, t0_source="manual", duration_s=2.0)
    port = free_port()
    mgr = ReplayManager(db, ws.root / "live", port=port)
    try:
        mgr.start(["cam_01"], 1.0)
        assert wait_for(lambda: mgr.status()["streams"][0]["state"] == "running", 6)
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,width,height", "-of", "json", f"rtsp://127.0.0.1:{port}/cam_01"],
            capture_output=True, text=True, timeout=30,
        )
        stream = json.loads(out.stdout)["streams"][0]
        assert (stream["codec_name"], stream["width"], stream["height"]) == ("h264", 320, 240)
    finally:
        mgr.shutdown()
    assert wait_for(lambda: not port_open(port), 5), "the port is free again after shutdown"
