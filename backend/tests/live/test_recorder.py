import json
import os
import shutil
import subprocess
import threading
import time
import types
from datetime import datetime
from pathlib import Path

import pytest
from contracts.models import Evidence
from fastapi.testclient import TestClient

from evora.api import context as context_module
from evora.api.app import create_app
from evora.core import cameras as cams
from evora.core.media_service import NOT_BUFFERED, MediaError
from evora.evidence import pack, store
from evora.live.recorder import STAMP, Recorder, RecordingIndex, segment_command
from evora.live.restream import LiveError, find_mediamtx
from tests.live.test_restream import FakeProc, free_port, wait_for

T = 1_800_000_000.0  # a fixed wall-clock time; segment names are local time, like ffmpeg writes them


def seg_name(start: float) -> str:
    return datetime.fromtimestamp(start).strftime(STAMP) + ".ts"


def put(folder: Path, start: float, end: float, data: bytes = b"x" * 1000) -> Path:
    """A segment file that began at `start` and was last written at `end`."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / seg_name(start)
    path.write_bytes(data)
    os.utime(path, (end, end))
    return path


# ---- the command ----------------------------------------------------------------------------------------------------------

def test_the_command_copies_the_stream_into_segments_named_by_start_time(tmp_path):
    cmd = segment_command("ffmpeg", "rtsp://cam/stream", tmp_path / "cam_01", 10)
    assert cmd[0] == "ffmpeg" and cmd[cmd.index("-i") + 1] == "rtsp://cam/stream"
    assert cmd[cmd.index("-c:v") + 1] == "copy" and "-an" in cmd, "no re-encoding, no audio"
    assert cmd[cmd.index("-segment_time") + 1] == "10" and cmd[cmd.index("-segment_format") + 1] == "mpegts"
    assert cmd[cmd.index("-strftime") + 1] == "1" and cmd[-1] == str(tmp_path / "cam_01" / "%Y%m%d_%H%M%S.ts")
    assert all(isinstance(a, str) for a in cmd)


# ---- the index ------------------------------------------------------------------------------------------------------------

@pytest.fixture()
def index(tmp_path):
    return RecordingIndex(tmp_path / "rec", segment_s=10, keep_s=100, max_bytes=10_000)


def test_segments_come_from_the_folder_sorted_and_clamped(index):
    folder = index.directory("cam_01")
    put(folder, T + 20, T + 30.2)
    put(folder, T, T + 10.4)
    put(folder, T + 10, T + 20.1)
    (folder / "notes.txt").write_text("not a segment")
    (folder / "garbage.ts").write_bytes(b"x")
    (folder / "20269999_999999.ts").write_bytes(b"x")
    segs = index.segments("cam_01")
    assert [round(s.start - T) for s in segs] == [0, 10, 20]
    assert segs[0].end == pytest.approx(T + 10), "a segment ends no later than the next one starts"
    assert segs[2].end == pytest.approx(T + 30.2)
    assert index.segments("cam_02") == [] and index.has_recording("cam_01") and not index.has_recording("cam_02")


def test_an_empty_segment_from_a_failed_start_is_ignored(index):
    folder = index.directory("cam_01")
    put(folder, T, T + 5, b"")
    put(folder, T + 10, T + 15)
    assert [round(s.start - T) for s in index.segments("cam_01")] == [10]
    assert index.at("cam_01", T + 2) is None


def test_lookup_by_time(index):
    folder = index.directory("cam_01")
    put(folder, T, T + 10)
    put(folder, T + 30, T + 40)  # the camera was off in between
    assert index.at("cam_01", T + 5).start == pytest.approx(T)
    assert index.at("cam_01", T + 11).start == pytest.approx(T), "just after the last write still belongs to it"
    assert index.at("cam_01", T + 20) is None, "inside the gap"
    assert index.at("cam_01", T - 1) is None and index.at("cam_01", T + 50) is None
    assert [round(s.start - T) for s in index.window("cam_01", T + 5, T + 35)] == [0, 30]
    assert index.window("cam_01", T + 15, T + 25) == []
    assert index.buffered_s("cam_01") == pytest.approx(20.0)


def test_a_bad_camera_id_is_refused(index):
    with pytest.raises(ValueError):
        index.directory("../x")
    assert index.has_recording("../x") is False


def test_old_segments_are_deleted_but_never_the_newest(index):
    folder = index.directory("cam_01")
    old = put(folder, T - 600, T - 590)
    mid = put(folder, T - 60, T - 50)
    newest = put(folder, T - 10, T)
    assert index.prune("cam_01", now=T) == 1
    assert not old.exists() and mid.exists() and newest.exists()
    assert index.prune("cam_01", now=T + 10_000) == 1, "everything is old, the newest is still kept"
    assert newest.exists() and not mid.exists()


def test_the_disk_cap_removes_the_oldest_first(tmp_path):
    index = RecordingIndex(tmp_path / "rec", segment_s=10, keep_s=10_000, max_bytes=2500)
    folder = index.directory("cam_01")
    paths = [put(folder, T + 10 * i, T + 10 * i + 10, b"x" * 1000) for i in range(5)]
    assert index.prune("cam_01", now=T + 50) == 3
    assert [p.exists() for p in paths] == [False, False, False, True, True]
    assert index.stats("cam_01")["bytes"] == 2000


def test_pruning_an_empty_or_missing_folder_is_fine(index):
    assert index.prune("cam_01") == 0 and index.stats("cam_01")["segments"] == 0


# ---- the supervisor ----------------------------------------------------------------------------------------------------------

class Spawn:
    def __init__(self):
        self.procs: list[FakeProc] = []

    def __call__(self, args, log_file):
        proc = FakeProc(args)
        self.procs.append(proc)
        return proc


@pytest.fixture()
def recorder(tmp_path):
    states: list[tuple[str, str]] = []
    spawn = Spawn()
    idx = RecordingIndex(tmp_path / "rec", segment_s=1.0, keep_s=100, max_bytes=10_000)
    rec = Recorder(idx, tmp_path / "live", ffmpeg="ffmpeg", spawn=spawn, on_state=lambda c, s: states.append((c, s)),
                   backoff_s=0.05, join_timeout_s=2.0)
    rec.spawn, rec.states, rec.idx = spawn, states, idx
    yield rec
    rec.shutdown()


def test_start_launches_one_ffmpeg_and_reports_recording(recorder):
    recorder.start("cam_01", "rtsp://cam/a")
    assert len(recorder.spawn.procs) == 1 and recorder.spawn.procs[0].args[recorder.spawn.procs[0].args.index("-i") + 1] == "rtsp://cam/a"
    assert recorder.idx.directory("cam_01").is_dir() and recorder.is_recording("cam_01")
    assert wait_for(lambda: ("cam_01", "recording") in recorder.states, 4)
    status = recorder.status()[0]
    assert status["camera_id"] == "cam_01" and status["state"] == "recording" and status["segments"] == 0


def test_starting_the_same_address_again_changes_nothing_a_new_one_restarts(recorder):
    recorder.start("cam_01", "rtsp://cam/a")
    recorder.start("cam_01", "rtsp://cam/a")
    assert len(recorder.spawn.procs) == 1
    recorder.start("cam_01", "rtsp://cam/b")
    assert len(recorder.spawn.procs) == 2 and recorder.spawn.procs[0].killed


def test_a_dropped_camera_comes_back_by_itself(recorder):
    recorder.start("cam_01", "rtsp://cam/a")
    assert wait_for(lambda: ("cam_01", "recording") in recorder.states, 4)
    recorder.spawn.procs[0].die(1)
    assert wait_for(lambda: len(recorder.spawn.procs) == 2, 4), "ffmpeg was started again"
    assert ("cam_01", "retrying") in recorder.states
    assert recorder.status()[0]["restarts"] == 1
    assert wait_for(lambda: recorder.status()[0]["state"] == "recording", 4)
    assert recorder.status()[0]["error"] is None, "the error clears once it records again"


def test_stop_kills_ffmpeg_and_a_stopped_recorder_does_not_restart(recorder):
    recorder.start("cam_01", "rtsp://cam/a")
    recorder.stop(["cam_01"])
    assert recorder.spawn.procs[0].killed and not recorder.is_recording("cam_01")
    assert recorder.states[-1] == ("cam_01", "stopped")
    time.sleep(0.3)
    assert len(recorder.spawn.procs) == 1
    recorder.stop(["nobody"])
    recorder.shutdown()
    recorder.shutdown()


def test_old_segments_are_pruned_while_recording(recorder):
    folder = recorder.idx.directory("cam_01")
    old = put(folder, time.time() - 1000, time.time() - 990)
    newest = put(folder, time.time() - 2, time.time())
    recorder.start("cam_01", "rtsp://cam/a")
    assert wait_for(lambda: not old.exists(), 4) and newest.exists()


def test_without_ffmpeg_start_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    rec = Recorder(RecordingIndex(tmp_path / "rec"), tmp_path / "live", spawn=Spawn())
    with pytest.raises(LiveError) as exc:
        rec.start("cam_01", "rtsp://cam/a")
    assert exc.value.status == 500


# ---- footage from the buffer: frames, clips, evidence packs ------------------------------------------------------------------

def make_ts(path: Path, colour: str, seconds: int = 3) -> bytes:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:size=160x120:rate=10:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "10", "-f", "mpegts", str(path)],
        check=True,
    )
    return path.read_bytes()


@pytest.fixture(scope="session")
def colours(tmp_path_factory) -> dict[str, bytes]:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    root = tmp_path_factory.mktemp("ts")
    return {name: make_ts(root / f"{name}.ts", name) for name in ("red", "blue")}


@pytest.fixture()
def live(tmp_path, monkeypatch, colours):
    """A real camera whose buffer holds a red segment [T, T+3] and a blue one [T+3, T+6]."""
    monkeypatch.setenv("evora_WORKSPACE", "livebuf")
    client = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))
    ctx = client.app.state.ctx
    cams.insert_camera(ctx.db, name="Door", kind="rtsp", source_uri="rtsp://user:secret@10.0.0.9/door", t0=T, t0_source="live")
    folder = ctx.recordings.directory("cam_01")
    put(folder, T, T + 3, colours["red"])
    put(folder, T + 3, T + 6, colours["blue"])
    client.post("/api/settings", json={"blur_faces": False})
    client.ctx = ctx
    return client


def dominant(jpeg: bytes) -> str:
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    b, _, r = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
    return "red" if r > b else "blue"


def register(live, eid, t_start, t_end, t_peak):
    store.register(live.ctx.db, Evidence(
        id=eid, camera_id="cam_01", camera_name="Door", t_start=t_start, t_end=t_end, t_peak=t_peak, offset_s=0.0,
        thumb_url=f"/api/media/thumb/{eid}.jpg", clip_url=f"/api/media/clip/{eid}.mp4", score=1.0, why=["test"],
    ))
    return eid


def duration(content: bytes, tmp_path: Path) -> float:
    path = tmp_path / "probe.mp4"
    path.write_bytes(content)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def test_a_frame_comes_from_the_segment_holding_that_second(live):
    ctx, cam = live.ctx, cams.get_camera(live.ctx.db, "cam_01")
    assert dominant(ctx.media.frame(cam, T + 1.0, False)[0]) == "red"
    assert dominant(ctx.media.frame(cam, T + 4.5, False)[0]) == "blue"
    assert dominant(ctx.media.frame(cam, T + 2.9, False)[0]) == "red", "the last moments of a segment are not lost"
    assert dominant(ctx.media.frame(cam, T + 5.95, False)[0]) == "blue"


def test_a_moment_outside_the_buffer_is_a_404_with_the_reason(live):
    ctx, cam = live.ctx, cams.get_camera(live.ctx.db, "cam_01")
    for t in (T - 30, T + 60):
        with pytest.raises(MediaError) as exc:
            ctx.media.frame(cam, t, False)
        assert exc.value.status == 404 and exc.value.message == NOT_BUFFERED
    eid = register(live, "ev_gone", T + 100, T + 104, T + 102)
    r = live.get(f"/api/media/clip/{eid}.mp4")
    assert r.status_code == 404 and NOT_BUFFERED in r.json()["detail"]
    assert live.get(f"/api/media/thumb/{eid}.jpg").status_code == 404


def test_the_thumbnail_route_serves_a_buffered_moment(live):
    eid = register(live, "ev_thumb", T + 3.5, T + 4.5, T + 4.0)
    r = live.get(f"/api/media/thumb/{eid}.jpg")
    assert r.status_code == 200 and dominant(r.content) == "blue"


def test_a_clip_inside_one_segment_renders(live, tmp_path):
    ctx = live.ctx
    ctx.media.pre_roll = ctx.media.post_roll = 0.0
    eid = register(live, "ev_one", T + 0.5, T + 2.0, T + 1.0)
    r = live.get(f"/api/media/clip/{eid}.mp4")
    assert r.status_code == 200 and duration(r.content, tmp_path) == pytest.approx(1.5, abs=0.4)


def test_a_clip_across_two_segments_joins_them(live, tmp_path):
    ctx = live.ctx
    ctx.media.pre_roll = ctx.media.post_roll = 0.0
    eid = register(live, "ev_two", T + 1.5, T + 4.5, T + 3.0)
    r = live.get(f"/api/media/clip/{eid}.mp4")
    assert r.status_code == 200 and duration(r.content, tmp_path) == pytest.approx(3.0, abs=0.5)


def test_a_clip_is_cut_to_what_was_recorded(live, tmp_path):
    ctx = live.ctx
    ctx.media.pre_roll = ctx.media.post_roll = 10.0  # asks for far more than the buffer holds
    eid = register(live, "ev_wide", T + 2.0, T + 4.0, T + 3.0)
    r = live.get(f"/api/media/clip/{eid}.mp4")
    assert r.status_code == 200 and duration(r.content, tmp_path) == pytest.approx(6.0, abs=0.6)


def test_a_cached_clip_survives_the_buffer_rolling_over(live):
    ctx = live.ctx
    ctx.media.pre_roll = ctx.media.post_roll = 0.0
    eid = register(live, "ev_keep", T + 0.5, T + 2.0, T + 1.0)
    assert live.get(f"/api/media/clip/{eid}.mp4").status_code == 200
    for seg in ctx.recordings.segments("cam_01"):
        seg.path.unlink()
    assert live.get(f"/api/media/clip/{eid}.mp4").status_code == 200, "the clip rendered while the buffer existed is still served"
    fresh = register(live, "ev_fresh", T + 1.0, T + 2.0, T + 1.5)
    assert live.get(f"/api/media/clip/{fresh}.mp4").status_code == 409, "nothing cached and no recording left"


def test_a_pack_of_live_footage_lists_segment_hashes_and_never_the_camera_address(live, tmp_path):
    import hashlib

    ctx = live.ctx
    ctx.media.pre_roll = ctx.media.post_roll = 0.0
    eid = register(live, "ev_pack", T + 1.5, T + 4.5, T + 3.0)
    r = live.post(f"/api/evidence/{eid}/pack")
    assert r.status_code == 200 and pack.verify_pack(r.content) == []
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        members = {i.filename: zf.read(i) for i in zf.infolist()}
    manifest = json.loads(members["manifest.json"])
    source = manifest["source"]
    assert source["kind"] == "live_recording" and len(source["segments"]) == 2
    for entry, seg in zip(source["segments"], ctx.recordings.segments("cam_01"), strict=True):
        assert entry["name"] == seg.path.name and entry["sha256"] == hashlib.sha256(seg.path.read_bytes()).hexdigest()
    evidence = json.loads(members["evidence.json"])
    assert evidence["times"]["offset_in_file_s"] is None
    blob = b"".join(members.values())
    assert b"secret" not in blob and b"rtsp://" not in blob and b"10.0.0.9" not in blob


def test_a_live_camera_with_no_recording_still_cannot_be_packed(live):
    cams.insert_camera(live.ctx.db, name="Other", kind="rtsp", source_uri="rtsp://10.0.0.8/x", t0=T, t0_source="live")
    store.register(live.ctx.db, Evidence(
        id="ev_none", camera_id="cam_02", camera_name="Other", t_start=T, t_end=T + 1, t_peak=T + 0.5, offset_s=0.0,
        thumb_url="/t", clip_url="/c", score=1.0,
    ))
    assert live.post("/api/evidence/ev_none/pack").status_code == 409


# ---- the alert hook and the live runner ----------------------------------------------------------------------------------------

def test_a_live_alert_on_a_real_camera_renders_its_clip_after_the_post_roll(live, monkeypatch):
    ctx = live.ctx
    scheduled, timers = [], []

    class FakeTimer:
        def __init__(self, delay, fn):
            self.delay, self.fn, self.daemon = delay, fn, False
            timers.append(self)

        def start(self):
            self.fn()

    monkeypatch.setattr(context_module.threading, "Timer", FakeTimer)
    monkeypatch.setattr(ctx.prerender, "schedule", lambda ids: scheduled.append(list(ids)))
    alert = types.SimpleNamespace(camera_id="cam_01", evidence=types.SimpleNamespace(id="ev_al_1"))
    ctx.alerts.on_alert(alert, False)
    assert scheduled == [["ev_al_1"]]
    assert timers[0].delay == pytest.approx(ctx.media.post_roll + ctx.recordings.segment_s + 2.0) and timers[0].daemon
    ctx.alerts.on_alert(alert, True)
    assert len(scheduled) == 1, "alerts about earlier footage need no clip cut now"


def test_a_recorded_file_camera_needs_no_prerender_timer(live, sample_mp4, monkeypatch):
    ctx = live.ctx
    with open(sample_mp4, "rb") as fh:
        live.post("/api/cameras", files=[("files", ("gate.mp4", fh))])
    scheduled = []
    monkeypatch.setattr(ctx.prerender, "schedule", lambda ids: scheduled.append(ids))
    monkeypatch.setattr(context_module.threading, "Timer", lambda *a, **k: pytest.fail("no timer for a file camera"))
    ctx.alerts.on_alert(types.SimpleNamespace(camera_id="cam_02", evidence=types.SimpleNamespace(id="x")), False)
    ctx.alerts.on_alert(types.SimpleNamespace(camera_id="cam_77", evidence=types.SimpleNamespace(id="x")), False)
    assert scheduled == []


def test_a_failing_alert_hook_never_stops_the_alert(tmp_path):
    from tests.alerts.conftest import Env

    env = Env(tmp_path / "ws")
    env.line()
    env.remember_gate()
    env.track()
    from evora.alerts import store as alert_store
    from evora.alerts.store import StandingRule

    alert_store.create_standing(env.db, "watch", StandingRule(
        targets=["person"], place="main gate", camera_ids=["cam_01"], zone_id="z_gate", events=["cross_line"],
        direction="a_to_b", cooldown_s=30.0, summary="Alert when a person enters main gate."))
    env.engine.invalidate()
    seen = []

    def hook(alert, historical):
        seen.append((alert.id, historical))
        raise RuntimeError("boom")

    env.engine.on_alert = hook
    assert len(env.engine.evaluate(env.event(), historical=False)) == 1 and len(seen) == 1 and seen[0][1] is False


class FakeRecorder:
    def __init__(self, fail=False):
        self.started, self.stopped, self.fail = [], [], fail

    def start(self, camera_id, url):
        if self.fail:
            raise LiveError(500, "ffmpeg is not installed")
        self.started.append((camera_id, url))

    def stop(self, camera_ids=None):
        self.stopped.append(camera_ids)

    def status(self):
        return [{"camera_id": c, "state": "recording"} for c, _ in self.started]


def runner_world(tmp_path, monkeypatch, recorder):
    from tests.live.test_live_runner import World

    w = World(tmp_path / "ws", monkeypatch)
    w.runner.recorder = recorder
    return w


def test_analysing_a_real_camera_records_it_and_stopping_stops_both(tmp_path, monkeypatch):
    rec = FakeRecorder()
    w = runner_world(tmp_path, monkeypatch, rec)
    try:
        w.runner.start(["cam_03"])
        assert rec.started == [("cam_03", "rtsp://127.0.0.1:8554/door")]
        assert w.runner.status()["recordings"] == [{"camera_id": "cam_03", "state": "recording"}]
        w.runner.stop(["cam_03"])
        assert rec.stopped == [["cam_03"]]
    finally:
        w.runner.shutdown()


def test_a_replayed_file_is_not_recorded(tmp_path, monkeypatch):
    rec = FakeRecorder()
    w = runner_world(tmp_path, monkeypatch, rec)
    try:
        w.replay.run("cam_01")
        w.runner.start(["cam_01"])
        assert rec.started == []
    finally:
        w.runner.shutdown()


def test_a_recorder_that_cannot_start_does_not_stop_the_analysis(tmp_path, monkeypatch):
    w = runner_world(tmp_path, monkeypatch, FakeRecorder(fail=True))
    try:
        w.runner.start(["cam_03"])
        assert wait_for(lambda: w.ingest.calls, 4), "the analysis still started"
    finally:
        w.runner.shutdown()


def test_recording_can_be_switched_off(tmp_path, monkeypatch):
    w = runner_world(tmp_path, monkeypatch, None)
    try:
        w.runner.start(["cam_03"])
        assert wait_for(lambda: w.ingest.calls, 4) and w.runner.status()["recordings"] == []
    finally:
        w.runner.shutdown()


def test_the_app_wires_the_recorder_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "recwire")
    ctx = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object())).app.state.ctx
    assert ctx.live_runner.recorder is ctx.recorder and ctx.media._recordings is ctx.recordings
    assert (ctx.recordings.segment_s, ctx.recordings.keep_s, ctx.recordings.max_bytes) == (10.0, 1800.0, 2_000_000_000)
    assert ctx.recordings.root == ctx.ws.root / "live" / "rec"


# ---- the real thing: MediaMTX, ffmpeg and a replayed file --------------------------------------------------------------

@pytest.mark.skipif(find_mediamtx() is None or shutil.which("ffmpeg") is None, reason="MediaMTX or ffmpeg is not installed")
def test_a_real_stream_is_recorded_and_a_clip_is_cut_from_it(tmp_path, monkeypatch, sample_mp4):
    monkeypatch.setenv("evora_WORKSPACE", "realrec")
    client = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))
    ctx = client.app.state.ctx
    with open(sample_mp4, "rb") as fh:
        client.post("/api/cameras", files=[("files", ("gate.mp4", fh))])
    client.post("/api/settings", json={"blur_faces": False})
    ctx.live.port = free_port()
    ctx.recordings.segment_s = 2.0
    try:
        assert client.post("/api/live/replay", json={"camera_ids": ["cam_01"]}).status_code == 200
        assert wait_for(lambda: client.get("/api/live").json()["streams"][0]["state"] == "running", 10)
        cams.insert_camera(
            ctx.db, name="Replay as camera", kind="rtsp", source_uri=f"rtsp://127.0.0.1:{ctx.live.port}/cam_01",
            t0=time.time(), t0_source="live",
        )
        ctx.recorder.start("cam_02", f"rtsp://127.0.0.1:{ctx.live.port}/cam_01")
        assert wait_for(lambda: len(ctx.recordings.segments("cam_02")) >= 3, 25), ctx.recorder.status()
        first = ctx.recordings.segments("cam_02")[0]
        ctx.media.pre_roll = ctx.media.post_roll = 0.0
        eid = register_for(ctx, "cam_02", first.start + 0.5, first.start + 3.0)
        clip = client.get(f"/api/media/clip/{eid}.mp4")
        thumb = client.get(f"/api/media/thumb/{eid}.jpg")
        assert clip.status_code == 200 and thumb.status_code == 200, (clip.text, thumb.text)
        assert duration(clip.content, tmp_path) == pytest.approx(2.5, abs=0.8)
    finally:
        ctx.recorder.shutdown()
        ctx.live.shutdown()


def register_for(ctx, camera_id, t_start, t_end):
    eid = "ev_real"
    store.register(ctx.db, Evidence(
        id=eid, camera_id=camera_id, camera_name="x", t_start=t_start, t_end=t_end, t_peak=(t_start + t_end) / 2, offset_s=0.0,
        thumb_url=f"/api/media/thumb/{eid}.jpg", clip_url=f"/api/media/clip/{eid}.mp4", score=1.0,
    ))
    return eid



# ---- recorder health ---------------------------------------------------------------------------------------------------------

def test_health_reports_the_buffer_of_every_camera(live):
    body = live.get("/api/health").json()
    assert body["recordings"] == [], "nothing is recording until a camera is analysed"
    ctx = live.ctx
    ctx.recorder._recs["cam_01"] = types.SimpleNamespace(
        camera_id="cam_01", state="recording", restarts=0, error=None, url="rtsp://x", stop=threading.Event(), proc=None,
    )
    row = live.get("/api/health").json()["recordings"][0]
    assert (row["camera_id"], row["state"], row["segments"]) == ("cam_01", "recording", 2)
    assert row["buffered_s"] == pytest.approx(6.0) and row["bytes"] > 0
    ctx.recorder._recs.clear()


def test_a_recording_that_goes_quiet_is_flagged_stalled(recorder):
    folder = recorder.idx.directory("cam_01")  # segment_s = 1.0: stalled after 3 s of silence
    put(folder, time.time() - 20, time.time() - 10)
    rec = types.SimpleNamespace(
        camera_id="cam_01", state="recording", restarts=0, error=None, url="rtsp://x", stop=threading.Event(), proc=None,
    )
    recorder._recs["cam_01"] = rec
    row = recorder.status()[0]
    assert row["stalled"] is True and row["last_segment_age_s"] == pytest.approx(10, abs=1.5)
    put(folder, time.time() - 1, time.time())
    row = recorder.status()[0]
    assert row["stalled"] is False and row["last_segment_age_s"] < 2
    rec.state = "retrying"
    put(folder, time.time() - 40, time.time() - 30)
    assert recorder.status()[0]["stalled"] is False, "a camera that is reconnecting is not 'stalled', it says retrying"
    recorder._recs.clear()


def test_a_camera_without_segments_has_no_age(recorder):
    recorder._recs["cam_01"] = types.SimpleNamespace(
        camera_id="cam_01", state="starting", restarts=0, error=None, url="rtsp://x", stop=threading.Event(), proc=None,
    )
    row = recorder.status()[0]
    assert row["last_segment_age_s"] is None and row["stalled"] is False
    recorder._recs.clear()
