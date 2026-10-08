import threading
import time

import pytest
from fastapi.testclient import TestClient

from evora.alerts import store
from evora.alerts.store import StandingRule
from evora.api.app import create_app
from evora.core import cameras as cams
from evora.core import live_sessions
from evora.live.live_runner import INSTALL_HINT, LiveRunner
from evora.live.restream import LiveError
from tests.alerts.conftest import T0, Env
from tests.live.test_restream import wait_for


class FakeReplay:
    """Just the part of the replay manager the runner reads."""

    def __init__(self):
        self.streams: dict[str, dict] = {}

    def status(self):
        return {"server": {}, "streams": list(self.streams.values())}

    def run(self, camera_id, state="running", url=None):
        self.streams[camera_id] = {"camera_id": camera_id, "state": state, "url": url or f"rtsp://127.0.0.1:8554/{camera_id}"}


class FakeIngest:
    """Stands in for M2's live_ingest: remembers what it was given, runs until told to stop, feeds scripted events."""

    def __init__(self):
        self.calls: list[dict] = []
        self.script: dict[str, list[dict]] = {}
        self.crash: dict[str, Exception] = {}
        self.gone = threading.Event()

    def __call__(self, cam, profile, stop, on_event, *, ws=None, on_state=None):
        self.calls.append({"cam": cam, "profile": profile, "ws": ws})
        if on_state:
            on_state("running")
        for event in self.script.get(cam.id, []):
            on_event(event)
        if cam.id in self.crash:
            raise self.crash[cam.id]
        stop.wait(10)
        self.gone.set()


class World:
    def __init__(self, root, monkeypatch):
        self.env = Env(root)
        cams.insert_camera(
            self.env.db, name="Door", kind="rtsp", source_uri="rtsp://127.0.0.1:8554/door", t0=T0, t0_source="live",
        )
        self.replay = FakeReplay()
        self.ingest = FakeIngest()
        self.notes: list[tuple[str, dict]] = []
        real = self.env.bus.publish
        monkeypatch.setattr(self.env.bus, "publish", lambda kind, data: (self.notes.append((kind, data)), real(kind, data))[0])
        self.allowed = {"127.0.0.1"}
        self.onprem = False
        self.runner = LiveRunner(
            self.env.db, self.env.ws, self.env.bus, self.env.engine, self.replay, profile="cpu",
            onprem=lambda: self.onprem, allows=lambda host: host in self.allowed,
            find_live_ingest=lambda: self.ingest, join_timeout_s=3.0,
        )

    def states(self, camera_id):
        return [d["state"] for k, d in self.notes if k == "analysis" and d["camera_id"] == camera_id]

    def status(self, camera_id):
        return cams.get_camera(self.env.db, camera_id).status


@pytest.fixture()
def world(tmp_path, monkeypatch):
    w = World(tmp_path / "ws", monkeypatch)
    yield w
    w.runner.shutdown()


def watch(env) -> str:
    rule = StandingRule(
        targets=["person"], place="main gate", camera_ids=["cam_01"], zone_id="z_gate", events=["cross_line"],
        direction="a_to_b", cooldown_s=30.0, summary="Alert when a person enters main gate.",
    )
    sq = store.create_standing(env.db, "watch the gate", rule)
    env.engine.invalidate()
    return sq.id


# ---- where the frames come from, and whether we may read them -----------------------------------------------------

def test_a_recorded_file_needs_its_replay_first(world):
    with pytest.raises(LiveError) as exc:
        world.runner.stream_url("cam_01")
    assert exc.value.status == 409 and "replay" in exc.value.message
    world.replay.run("cam_01", state="stopped")
    with pytest.raises(LiveError):
        world.runner.stream_url("cam_01")
    world.replay.run("cam_01", url="rtsp://127.0.0.1:8554/cam_01")
    assert world.runner.stream_url("cam_01") == "rtsp://127.0.0.1:8554/cam_01"


def test_an_rtsp_camera_streams_from_its_own_address(world):
    assert world.runner.stream_url("cam_03") == "rtsp://127.0.0.1:8554/door"


@pytest.mark.parametrize("cid,status", [("cam_99", 404), ("../etc", 422), ("", 422)])
def test_bad_camera_ids(world, cid, status):
    with pytest.raises(LiveError) as exc:
        world.runner.stream_url(cid)
    assert exc.value.status == status


def test_on_prem_only_reads_this_machine_or_the_allow_list(world):
    cams.insert_camera(world.env.db, name="Lobby cam", kind="rtsp", source_uri="rtsp://10.0.0.5/lobby", t0=T0, t0_source="live")
    assert world.runner.stream_url("cam_04") == "rtsp://10.0.0.5/lobby", "cloud mode: any address"
    world.onprem = True
    with pytest.raises(LiveError) as exc:
        world.runner.stream_url("cam_04")
    assert exc.value.status == 403 and "not on this machine" in exc.value.message
    world.allowed.add("10.0.0.5")
    assert world.runner.stream_url("cam_04").startswith("rtsp://10.0.0.5")
    assert world.runner.stream_url("cam_03"), "loopback stays fine"


# ---- starting -----------------------------------------------------------------------------------------------------------

def test_without_the_perception_stack_start_says_what_to_install(world):
    world.runner._find = lambda: None
    with pytest.raises(LiveError) as exc:
        world.runner.start(["cam_03"])
    assert exc.value.status == 503 and exc.value.message == INSTALL_HINT


def test_everything_is_validated_before_anything_starts(world):
    with pytest.raises(LiveError):
        world.runner.start(["cam_03", "cam_99"])
    assert world.ingest.calls == [] and world.runner.status() == {"analyzers": [], "recordings": []}
    with pytest.raises(LiveError):
        world.runner.start([])


def test_the_ingest_gets_a_transient_live_copy_of_the_same_camera(world):
    world.replay.run("cam_01", url="rtsp://127.0.0.1:8554/cam_01")
    before = time.time()
    world.runner.start(["cam_01"])
    assert wait_for(lambda: world.ingest.calls)
    call = world.ingest.calls[0]
    cam = call["cam"]
    assert cam.id == "cam_01" and cam.kind == "rtsp" and cam.source_uri == "rtsp://127.0.0.1:8554/cam_01"
    assert cam.t0_source == "live" and before <= cam.t0 <= time.time()
    assert call["profile"] == "cpu" and call["ws"] is world.env.ws
    stored = cams.get_camera(world.env.db, "cam_01")
    assert stored.kind == "file" and stored.t0 == T0, "the recorded camera itself is not rewritten"


def test_state_changes_set_the_camera_status_and_are_announced(world):
    world.runner.start(["cam_03"])
    assert wait_for(lambda: world.status("cam_03") == "live")
    assert wait_for(lambda: world.runner.status()["analyzers"][0]["state"] == "running")
    assert "running" in world.states("cam_03")
    world.runner.stop(["cam_03"])
    assert world.status("cam_03") == "ready" and world.states("cam_03")[-1] == "stopped"
    assert world.runner.status()["analyzers"][0]["error"] is None


def test_starting_twice_does_not_start_a_second_analyzer(world):
    world.runner.start(["cam_03"])
    assert wait_for(lambda: world.ingest.calls)
    world.runner.start(["cam_03"])
    time.sleep(0.2)
    assert len(world.ingest.calls) == 1


def test_a_function_without_ws_or_state_hooks_still_runs(world):
    seen = []

    def plain(cam, profile, stop, on_event):
        seen.append(cam.id)
        stop.wait(5)

    world.runner._find = lambda: plain
    world.runner.start(["cam_03"])
    assert wait_for(lambda: seen == ["cam_03"])


def test_cameras_are_independent(world):
    world.replay.run("cam_01")
    world.ingest.crash["cam_03"] = RuntimeError("stream refused")
    world.runner.start(["cam_01", "cam_03"])
    assert wait_for(lambda: world.status("cam_03") == "error")
    assert world.status("cam_01") == "live"
    by_id = {a["camera_id"]: a for a in world.runner.status()["analyzers"]}
    assert by_id["cam_03"]["error"] == "stream refused" and by_id["cam_01"]["state"] == "running"


def test_a_crash_is_reported_with_its_reason(world):
    world.ingest.crash["cam_03"] = RuntimeError("")
    world.runner.start(["cam_03"])
    assert wait_for(lambda: world.status("cam_03") == "error")
    assert world.runner.status()["analyzers"][0]["error"] == "RuntimeError"
    assert world.states("cam_03")[-1] == "error"


def test_stopping_everything_joins_the_threads(world):
    world.replay.run("cam_01")
    world.runner.start(["cam_01", "cam_03"])
    assert wait_for(lambda: len(world.ingest.calls) == 2)
    world.runner.shutdown()
    assert world.ingest.gone.is_set()
    assert all(not a.thread.is_alive() for a in world.runner._running.values())
    assert world.status("cam_01") == "ready" and world.status("cam_03") == "ready"


# ---- events reach the alert engine ---------------------------------------------------------------------------------------

def test_a_live_event_raises_an_alert_with_evidence(world):
    env = world.env
    env.line()
    env.remember_gate()
    env.track()
    sq = watch(env)
    world.replay.run("cam_01")
    world.ingest.script["cam_01"] = [env.event()]
    pushed = []
    env.engine.notifier = type("N", (), {"push": lambda self, title, text: pushed.append((title, text))})()
    world.runner.start(["cam_01"])
    assert wait_for(lambda: any(k == "alert" for k, _ in world.notes))
    alert = next(d for k, d in world.notes if k == "alert")
    assert alert["alert"]["standing_query_id"] == sq and alert["historical"] is False
    assert wait_for(lambda: pushed), "live events may buzz a phone"
    assert "main gate" in pushed[0][1]


def test_a_bad_event_never_stops_the_stream(world):
    env = world.env
    env.line()
    env.remember_gate()
    env.track()
    watch(env)
    world.replay.run("cam_01")
    world.ingest.script["cam_01"] = [{"nonsense": True}, env.event(event_id="e2")]
    world.runner.start(["cam_01"])
    assert wait_for(lambda: any(k == "alert" for k, _ in world.notes)), "the event after the bad one was still handled"
    assert world.runner.status()["analyzers"][0]["state"] == "running"


def test_alert_evidence_points_at_the_right_second_of_the_replayed_file(world):
    env = world.env
    env.line()
    env.remember_gate()
    env.track()
    watch(env)
    started = T0 + 5_000_000.0  # the replay began long after the recording
    live_sessions.begin(env.db, "cam_01", started, 1.0)
    alerts = env.engine.evaluate(env.event(t=started + 7.0), historical=False)
    assert len(alerts) == 1 and alerts[0].evidence.offset_s == pytest.approx(7.0)
    live_sessions.begin(env.db, "cam_01", started + 100.0, 2.0)
    again = env.engine.evaluate(env.event(event_id="e9", t=started + 100.0 + 5.0), historical=False)
    assert again and again[0].evidence.offset_s == pytest.approx(10.0), "a restart at speed 2 maps again from the file start"


def test_recorded_footage_keeps_the_recording_clock(world):
    env = world.env
    env.line()
    env.remember_gate()
    env.track()
    watch(env)
    live_sessions.begin(env.db, "cam_01", T0 + 5_000_000.0, 1.0)
    alerts = env.engine.evaluate(env.event(t=T0 + 10.0), historical=True)
    assert alerts[0].evidence.offset_s == pytest.approx(10.0)


# ---- the routes ------------------------------------------------------------------------------------------------------------

@pytest.fixture()
def api(tmp_path, monkeypatch, sample_mp4):
    monkeypatch.setenv("evora_WORKSPACE", "liveapi")
    client = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))
    with open(sample_mp4, "rb") as fh:
        client.post("/api/cameras", files=[("files", ("gate.mp4", fh))])
    ctx = client.app.state.ctx
    fake = FakeIngest()
    ctx.live_runner._find = lambda: fake
    ctx.live_runner.replay = FakeReplay()
    ctx.live.start = lambda ids, speed=None: ctx.live_runner.replay.run(ids[0])  # no MediaMTX in these tests
    yield client, ctx, fake
    ctx.live_runner.shutdown()


def test_analyze_needs_the_replay_first(api):
    client, _, fake = api
    r = client.post("/api/live/analyze", json={"camera_ids": ["cam_01"]})
    assert r.status_code == 409 and "replay" in r.json()["detail"] and fake.calls == []


def test_analyze_and_stop_through_the_api(api):
    client, ctx, fake = api
    ctx.live_runner.replay.run("cam_01")
    r = client.post("/api/live/analyze", json={"camera_ids": ["cam_01"]})
    assert r.status_code == 200 and r.json()["analyzers"][0]["camera_id"] == "cam_01"
    assert wait_for(lambda: fake.calls)
    assert client.get("/api/live").json()["analyzers"][0]["state"] in ("starting", "running")
    r = client.post("/api/live/analyze/stop", json={"camera_ids": ["cam_01"]})
    assert r.status_code == 200 and r.json()["analyzers"][0]["state"] == "stopped"


def test_replay_with_analyze_starts_both(api):
    client, ctx, fake = api
    r = client.post("/api/live/replay", json={"camera_ids": ["cam_01"], "analyze": True})
    assert r.status_code == 200 and [a["camera_id"] for a in r.json()["analyzers"]] == ["cam_01"]
    assert wait_for(lambda: fake.calls and fake.calls[0]["cam"].id == "cam_01")


def test_replay_without_analyze_only_streams(api):
    client, ctx, fake = api
    r = client.post("/api/live/replay", json={"camera_ids": ["cam_01"]})
    assert r.status_code == 200 and r.json()["analyzers"] == [] and fake.calls == []


def test_analyze_without_the_stack_is_a_503_with_the_hint(api):
    client, ctx, _ = api
    ctx.live_runner._find = lambda: None
    ctx.live_runner.replay.run("cam_01")
    r = client.post("/api/live/analyze", json={"camera_ids": ["cam_01"]})
    assert r.status_code == 503 and "start.bat setup" in r.json()["detail"]


def test_stopping_the_replay_stops_the_analysis_first(api):
    client, ctx, fake = api
    client.post("/api/live/replay", json={"camera_ids": ["cam_01"], "analyze": True})
    assert wait_for(lambda: fake.calls)
    r = client.post("/api/live/replay/stop")
    assert r.status_code == 200 and fake.gone.is_set()


def test_mock_mode_answers_canned_analysis(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "livemock")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", mock=True))
    assert c.post("/api/live/analyze", json={"camera_ids": ["cam_01"]}).json()["analyzers"][0]["state"] == "running"
    assert c.post("/api/live/analyze/stop").json()["analyzers"] == []
