"""Live ingest with a scripted frame source, detector and embedder: no camera, GPU or weights needed."""
import json
import threading
import time

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")
pytest.importorskip("lancedb")

from evora.core import cameras as cams  # noqa: E402
from evora.core import workspace as wsmod  # noqa: E402
from evora.core.db import open_db  # noqa: E402
from evora.perception import live  # noqa: E402
from evora.perception.decode import Frame  # noqa: E402
from evora.perception.settings import IngestSettings  # noqa: E402
from evora.perception.track import TrackedBox  # noqa: E402

DIM = 16
ROW_KEYS = {"id", "camera_id", "track_id", "kind", "zone_id", "t", "payload"}


class FakeEmbedder:
    dim = DIM

    def embed_images(self, images):
        out = np.zeros((len(images), DIM), dtype=np.float32)
        for i, im in enumerate(images):
            out[i, int(np.mean(im)) % DIM] = 1.0
        return out


class WalkingPerson:
    """One person walking left to right across a 320x240 frame, 8 px per processed frame."""

    def __init__(self, det, cfg):
        self.n = 0

    def update(self, bgr):
        self.n += 1
        x = 20 + self.n * 8
        return [TrackedBox(1, "person", 0.9, (float(x), 70.0, float(x + 40), 200.0))]


def _stream(n_frames=40, fps=20.0):
    """Yield n frames in real time, then stay silent until stopped (a camera whose scene went quiet)."""
    def source(stop):
        rng = np.random.default_rng(0)
        for i in range(n_frames):
            if stop.is_set():
                return
            yield time.time(), Frame.from_image(i, 0.0, rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
            time.sleep(1.0 / fps)
        stop.wait()
    return source


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "get_embedder", lambda cfg: FakeEmbedder())
    monkeypatch.setattr(live, "load_detector", lambda cfg: object())
    monkeypatch.setattr(live, "FrameTracker", WalkingPerson)
    ws = wsmod.create("live-test", tmp_path / "w")
    db = open_db(ws.db_path)
    cam = cams.insert_camera(db, name="Gate", kind="rtsp", source_uri="rtsp://127.0.0.1:8554/cam_01", t0=time.time(),
                             t0_source="live")
    with db.write() as c:
        c.execute("INSERT INTO zones(id,camera_id,kind,points,direction,created_at) VALUES(?,?,?,?,?,?)",
                  ("z_mid", cam.id, "line", "[[0.5,0.0],[0.5,1.0]]", "any", time.time()))
    return ws, db, cam


def _settings():
    return IngestSettings(min_track_obs=2, motion_gate=False, fixed_fps=20, track_lost_s=1.0, point_hz=10.0)


def run(cam, ws, source, on_event, until: threading.Event | None = None, timeout=12.0, on_state=None):
    stop = threading.Event()
    t = threading.Thread(target=live.live_ingest, args=(cam, "cpu", stop, on_event),
                         kwargs={"ws": ws, "settings": _settings(), "frame_source": source, "on_state": on_state})
    t.start()
    (until or threading.Event()).wait(timeout)
    started = time.monotonic()
    stop.set()
    t.join(timeout=10)
    assert not t.is_alive()
    return time.monotonic() - started


def test_events_arrive_live_in_row_shape_and_rows_exist_before_the_track_ends(env):
    ws, db, cam = env
    events, seen_in_db = [], {}
    done = threading.Event()

    def on_event(ev):
        events.append(ev)
        with db.read() as c:
            seen_in_db[ev["id"]] = c.execute("SELECT cls FROM tracks WHERE id=?", (ev["track_id"],)).fetchone()
        if ev["kind"] == "disappear":
            done.set()

    stop_after = run(cam, ws, _stream(), on_event, until=done)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "appear" and "cross_line" in kinds and kinds[-1] == "disappear"
    assert all(set(e) == ROW_KEYS for e in events)
    assert all(isinstance(e["payload"], str) for e in events)
    assert len({e["id"] for e in events}) == len(events)                       # every event exactly once
    crossing = next(e for e in events if e["kind"] == "cross_line")
    assert crossing["zone_id"] == "z_mid" and json.loads(crossing["payload"])["direction"] in ("a_to_b", "b_to_a")
    # whenever an event arrives, the alert engine can look its track up (class and trajectory)
    assert all(row is not None and row["cls"] == "person" for row in seen_in_db.values())
    assert stop_after < 3.0

    with db.read() as c:
        stored = {r["id"] for r in c.execute("SELECT id FROM events")}
        track = c.execute("SELECT * FROM tracks").fetchone()
        points = c.execute("SELECT count(*) FROM track_points").fetchone()[0]
    assert stored == {e["id"] for e in events}
    assert track["id"] == f"{cam.id}:t000001" and track["best_crop"].startswith(f"crops/{cam.id}/") and points > 3
    assert (ws.media_dir / track["best_crop"]).is_file()


def test_numbering_continues_after_a_restart(env):
    ws, db, cam = env
    for _ in range(2):
        done = threading.Event()
        run(cam, ws, _stream(), lambda e, d=done: d.set() if e["kind"] == "disappear" else None, until=done)
    with db.read() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM tracks ORDER BY id")]
    assert ids == [f"{cam.id}:t000001", f"{cam.id}:t000002"]


def test_a_failing_event_handler_does_not_stop_ingestion(env):
    ws, db, cam = env
    calls, done = [], threading.Event()

    def bad(ev):
        calls.append(ev["kind"])
        if ev["kind"] == "disappear":
            done.set()
        raise RuntimeError("alert engine exploded")

    run(cam, ws, _stream(), bad, until=done)
    assert "appear" in calls and calls[-1] == "disappear"


def test_a_broken_stream_is_retried_and_reported(env, monkeypatch):
    monkeypatch.setattr(live, "BACKOFF_START_S", 0.2)
    ws, db, cam = env
    attempts, states, done = [], [], threading.Event()
    good = _stream()

    def flaky(stop):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("camera unplugged")
        yield from good(stop)

    run(cam, ws, flaky, lambda e: done.set() if e["kind"] == "disappear" else None, until=done, on_state=states.append)
    assert len(attempts) >= 2
    assert states[0] == "running" and "retrying" in states and states[-1] == "running"


def test_slot_keeps_only_the_newest_frame():
    slot = live.FrameSlot()
    f = lambda i: (float(i), Frame.from_image(i, 0.0, np.zeros((4, 4, 3), dtype=np.uint8)))  # noqa: E731
    slot.put(f(1))
    slot.put(f(2))
    slot.put(f(3))
    item = slot.take(0.1)
    assert item is not None and item[0] == 3.0 and slot.dropped == 2
    assert slot.take(0.05) is None


def test_stopping_with_no_frames_returns_quickly(env):
    ws, db, cam = env

    def silent(stop):
        stop.wait()
        return
        yield

    elapsed = run(cam, ws, silent, lambda e: None, timeout=0.5)
    assert elapsed < 3.0
    with db.read() as c:
        assert c.execute("SELECT count(*) FROM tracks").fetchone()[0] == 0
