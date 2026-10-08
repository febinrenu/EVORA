import pytest
from contracts.models import Evidence
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.core import cameras as cams
from evora.core import live_sessions as ls
from evora.core import workspace as wsmod
from evora.core.db import open_db
from evora.evidence import store

T0 = 1791450000.0  # creation_time of the sample clip (2 s long)


@pytest.fixture()
def db(tmp_path):
    return open_db(wsmod.create("ls", tmp_path / "ws").db_path)


def test_a_replay_maps_wall_clock_to_the_file_and_loops(db):
    ls.begin(db, "cam_01", 100.0, 1.0)
    assert ls.file_offset(db, "cam_01", 10.0, 103.0) == pytest.approx(3.0)
    assert ls.file_offset(db, "cam_01", 10.0, 112.0) == pytest.approx(2.0), "the file has looped once"
    assert ls.file_offset(db, "cam_01", 10.0, 99.9) is None, "before the replay"
    assert ls.file_offset(db, "cam_02", 10.0, 103.0) is None, "another camera"


def test_speed_scales_the_position(db):
    ls.begin(db, "cam_01", 100.0, 2.0)
    assert ls.file_offset(db, "cam_01", 10.0, 103.0) == pytest.approx(6.0)


def test_without_a_duration_there_is_no_mapping(db):
    ls.begin(db, "cam_01", 100.0)
    assert ls.file_offset(db, "cam_01", None, 103.0) is None and ls.file_offset(db, "cam_01", 0, 103.0) is None


def test_a_restart_starts_a_new_session_and_closes_the_old_one(db):
    ls.begin(db, "cam_01", 100.0)
    ls.begin(db, "cam_01", 200.0)
    first, second = ls.load(db, "cam_01")
    assert first.ended_at == 200.0 and second.ended_at is None
    assert ls.file_offset(db, "cam_01", 50.0, 105.0) == pytest.approx(5.0), "a time in the first session keeps its position"
    assert ls.file_offset(db, "cam_01", 50.0, 203.0) == pytest.approx(3.0), "after the restart the loop began again"


def test_ending_a_session_stops_the_mapping(db):
    ls.begin(db, "cam_01", 100.0)
    ls.end(db, "cam_01", 150.0)
    assert ls.file_offset(db, "cam_01", 10.0, 149.0) is not None and ls.file_offset(db, "cam_01", 10.0, 151.0) is None
    ls.end(db, "cam_01", 160.0)  # ending twice changes nothing
    assert ls.load(db, "cam_01")[0].ended_at == 150.0


def test_a_damaged_value_means_no_sessions(db):
    db.set_meta("live_sessions.cam_01", "not json")
    assert ls.load(db, "cam_01") == [] and ls.file_offset(db, "cam_01", 10.0, 1.0) is None


def test_the_session_list_stays_small(db):
    for i in range(ls.MAX_SESSIONS + 20):
        ls.begin(db, "cam_01", float(i))
    assert len(ls.load(db, "cam_01")) == ls.MAX_SESSIONS


# ---- evidence from replayed-as-live footage can be rendered ---------------------------------------------------------

@pytest.fixture()
def env(tmp_path, monkeypatch, sample_mp4):
    monkeypatch.setenv("evora_WORKSPACE", "livemedia")
    client = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))
    with open(sample_mp4, "rb") as fh:
        client.post("/api/cameras", files=[("files", ("gate.mp4", fh))])
    client.post("/api/settings", json={"blur_faces": False})
    return client


def register(env, eid, t_start, t_end, t_peak):
    store.register(env.app.state.ctx.db, Evidence(
        id=eid, camera_id="cam_01", camera_name="gate", t_start=t_start, t_end=t_end, t_peak=t_peak, offset_s=0.0,
        thumb_url="/t", clip_url="/c", score=1.0,
    ))


def test_a_live_thumbnail_is_the_frame_the_replay_was_showing(env):
    ctx = env.app.state.ctx
    started = T0 + 5_000_000.0  # the replay started "now", long after the recording
    ls.begin(ctx.db, "cam_01", started, 1.0)
    cam = cams.get_camera(ctx.db, "cam_01")
    recorded, _ = ctx.media.frame(cam, T0 + 1.0, want_blur=False)
    live, _ = ctx.media.frame(cam, started + 1.0, want_blur=False)
    assert live == recorded, "wall-clock 1 s into the replay is 1 s into the file"
    looped, _ = ctx.media.frame(cam, started + 2.4, want_blur=False)
    again, _ = ctx.media.frame(cam, T0 + 0.4, want_blur=False)
    assert looped == again, "the loop restarted after 2 s of video"


def test_recorded_footage_still_uses_the_recording_clock(env):
    ctx = env.app.state.ctx
    ls.begin(ctx.db, "cam_01", T0 + 5_000_000.0, 1.0)
    register(env, "ev_rec", T0 + 0.5, T0 + 1.0, T0 + 0.8)
    assert env.post("/api/evidence/ev_rec/pack").status_code == 200


def test_a_live_clip_that_crosses_the_loop_seam_is_cut_at_the_end(env):
    ctx = env.app.state.ctx
    started = T0 + 5_000_000.0
    ls.begin(ctx.db, "cam_01", started, 1.0)
    register(env, "ev_seam", started + 1.7, started + 2.3, started + 2.0)  # 2 s file: spans the seam
    r = env.get("/api/media/clip/ev_seam.mp4")
    assert r.status_code == 200 and len(r.content) > 1000


def test_a_live_clip_inside_the_loop_renders(env):
    ctx = env.app.state.ctx
    started = T0 + 5_000_000.0
    ls.begin(ctx.db, "cam_01", started, 1.0)
    register(env, "ev_in", started + 0.5, started + 1.2, started + 0.8)
    assert env.get("/api/media/clip/ev_in.mp4").status_code == 200
    assert env.get("/api/media/thumb/ev_in.jpg").status_code == 200


def test_the_restream_records_its_sessions(tmp_path, monkeypatch):
    from evora.live.restream import ReplayManager
    from tests.live.test_restream import Spawner, free_port

    monkeypatch.setenv("evora_WORKSPACE", "restreamsessions")
    ws = wsmod.create("rs", tmp_path / "ws")
    db = open_db(ws.db_path)
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    cams.insert_camera(db, name="A", kind="file", source_uri=str(src), t0=1.0, t0_source="manual", duration_s=10.0)
    binary = tmp_path / "mediamtx.exe"
    binary.write_bytes(b"")
    port = free_port()
    ended = []
    mgr = ReplayManager(
        db, ws.root / "live", mediamtx_path=str(binary), port=port, spawn=Spawner(port), ffmpeg="ffmpeg",
        on_launch=lambda cid, started, speed: ls.begin(db, cid, started, speed), on_end=lambda cid: ended.append(cid),
    )
    try:
        mgr.start(["cam_01"], 2.0)
        sessions = ls.load(db, "cam_01")
        assert len(sessions) == 1 and sessions[0].speed == 2.0 and sessions[0].ended_at is None
        mgr.stop(["cam_01"])
        assert ended == ["cam_01"]
    finally:
        mgr.shutdown()


def test_the_router_maps_replayed_live_times_to_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "routerwire")
    ctx = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object())).app.state.ctx
    hook = ctx.router._file_offset
    assert hook is not None and hook("cam_01", 10.0, 103.0) is None, "no session yet: the router keeps t - t0"
    ls.begin(ctx.db, "cam_01", 100.0, 1.0)
    assert hook("cam_01", 10.0, 103.0) == pytest.approx(3.0)
    assert hook("cam_01", 10.0, 112.0) == pytest.approx(2.0), "looped once"
