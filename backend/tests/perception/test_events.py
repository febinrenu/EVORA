import json
import time

import pytest

pytest.importorskip("cv2")

from contracts.models import Zone  # noqa: E402

from evora.core import cameras as cams  # noqa: E402
from evora.core import workspace as wsmod  # noqa: E402
from evora.core.db import open_db  # noqa: E402
from evora.perception import events as ev  # noqa: E402
from evora.perception.settings import IngestSettings  # noqa: E402

A, B = (0.0, 0.5), (1.0, 0.5)   # horizontal line through the middle: left of a->b is "below" (y larger)


def _path(points, dt=0.25):
    return [ev.Sample(i * dt, x, y) for i, (x, y) in enumerate(points)]


def test_side_sign_convention():
    assert ev.side((0.5, 0.8), A, B) > 0   # below the line = left of a->b in image coordinates
    assert ev.side((0.5, 0.2), A, B) < 0
    assert ev.side((0.5, 0.5), A, B) == pytest.approx(0)


def test_crossing_direction_and_interpolated_time():
    down_to_up = _path([(0.5, 0.8), (0.5, 0.6), (0.5, 0.4), (0.5, 0.2)])      # left -> right
    [(t, d)] = ev.line_crossings(down_to_up, A, B)
    assert d == "a_to_b" and 0.25 < t < 0.5 and t == pytest.approx(0.375)    # crosses halfway between samples 1 and 2
    up_to_down = _path([(0.5, 0.2), (0.5, 0.4), (0.5, 0.6), (0.5, 0.8)])
    assert [d for _, d in ev.line_crossings(up_to_down, A, B)] == ["b_to_a"]


def test_jitter_along_the_line_is_not_a_crossing():
    wobble = _path([(0.5, 0.8), (0.5, 0.505), (0.5, 0.495), (0.5, 0.503), (0.5, 0.7)])
    assert ev.line_crossings(wobble, A, B, hysteresis=0.02) == []


def test_crossing_the_extension_of_a_short_line_is_ignored():
    short_a, short_b = (0.4, 0.5), (0.6, 0.5)
    outside = _path([(0.9, 0.8), (0.9, 0.2)])
    assert ev.line_crossings(outside, short_a, short_b) == []
    inside = _path([(0.5, 0.8), (0.5, 0.2)])
    assert len(ev.line_crossings(inside, short_a, short_b)) == 1


def test_there_and_back_gives_two_crossings():
    p = _path([(0.5, 0.8), (0.5, 0.2), (0.5, 0.8)])
    assert [d for _, d in ev.line_crossings(p, A, B)] == ["a_to_b", "b_to_a"]


SQUARE = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]


def test_point_in_polygon():
    assert ev.point_in_polygon((0.5, 0.5), SQUARE) and not ev.point_in_polygon((0.7, 0.5), SQUARE)


def test_inside_spans_with_debounce():
    p = _path([(0.1, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.9, 0.5), (0.9, 0.5)])
    [span] = ev.inside_spans(p, SQUARE, debounce=2)
    assert (span.t_in, span.t_out) == (0.25, 1.0)
    flicker = _path([(0.1, 0.5), (0.5, 0.5), (0.1, 0.5), (0.1, 0.5)])        # one stray sample inside
    assert ev.inside_spans(flicker, SQUARE, debounce=2) == []


def test_track_ending_inside_has_no_exit():
    p = _path([(0.1, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
    [span] = ev.inside_spans(p, SQUARE, debounce=2)
    assert span.t_out is None


@pytest.fixture
def db_with_tracks(tmp_path):
    ws = wsmod.create("events-test", tmp_path / "w")
    db = open_db(ws.db_path)
    cam = cams.insert_camera(db, name="Gate", kind="file", source_uri="x.mp4", t0=1000.0, t0_source="manual")

    def add_track(tid, pts, dt=0.25):
        with db.write() as c:
            c.execute("INSERT INTO tracks(id,camera_id,cls,t_start,t_end,n_obs) VALUES(?,?,?,?,?,?)",
                      (tid, cam.id, "person", 1000.0, 1000.0 + dt * (len(pts) - 1), len(pts)))
            c.executemany("INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,?,?,?,?,?)",
                          [(tid, 1000.0 + i * dt, x - 0.02, y - 0.2, x + 0.02, y, 0.9) for i, (x, y) in enumerate(pts)])
    return db, cam, add_track


def _kinds(db, cam_id):
    with db.read() as c:
        rows = c.execute("SELECT * FROM events WHERE camera_id=?", (cam_id,))
        return sorted((r["kind"], r["zone_id"], r["track_id"]) for r in rows)


def test_recompute_events_from_stored_points(db_with_tracks):
    db, cam, add_track = db_with_tracks
    add_track("t1", [(0.5, 0.8), (0.5, 0.6), (0.5, 0.4), (0.5, 0.2)])
    zone = Zone(id="z_gate", camera_id=cam.id, kind="line", points=[A, B], direction="any")
    started = time.perf_counter()
    n = ev.recompute_events(cam.id, [zone], db=db)
    assert time.perf_counter() - started < 2.0
    assert n == 3                                              # appear, disappear, one crossing
    with db.read() as c:
        row = c.execute("SELECT * FROM events WHERE kind='cross_line'").fetchone()
    assert row["zone_id"] == "z_gate" and json.loads(row["payload"]) == {"direction": "a_to_b"}
    assert 1000.25 < row["t"] < 1000.75
    assert [k for k, _, _ in _kinds(db, cam.id)] == ["appear", "cross_line", "disappear"]


def test_recompute_replaces_instead_of_duplicating_and_new_zone_is_retroactive(db_with_tracks):
    db, cam, add_track = db_with_tracks
    add_track("t1", [(0.1, 0.5)] + [(0.5, 0.5)] * 6 + [(0.9, 0.5)] * 2)
    line = Zone(id="z1", camera_id=cam.id, kind="line", points=[(0.3, 0.0), (0.3, 1.0)])
    ev.recompute_events(cam.id, [line], db=db)
    ev.recompute_events(cam.id, [line], db=db)
    assert sum(1 for k, *_ in _kinds(db, cam.id) if k == "cross_line") == 1
    poly = Zone(id="z2", camera_id=cam.id, kind="polygon", points=SQUARE)
    ev.recompute_events(cam.id, [poly], db=db, settings=IngestSettings(dwell_s=1.0))
    kinds = [k for k, z, _ in _kinds(db, cam.id) if z == "z2"]
    assert kinds == ["dwell", "enter_zone", "exit_zone"]
    assert sum(1 for k, *_ in _kinds(db, cam.id) if k == "cross_line") == 1  # the earlier zone is untouched


def test_zones_of_reads_stored_zones(db_with_tracks):
    db, cam, _ = db_with_tracks
    with db.write() as c:
        c.execute("INSERT INTO zones(id,camera_id,kind,points,direction,created_at) VALUES(?,?,?,?,?,?)",
                  ("z9", cam.id, "line", "[[0.1,0.2],[0.8,0.2]]", "a_to_b", 1.0))
    [z] = ev.zones_of(db, cam.id)
    assert z.id == "z9" and z.points == [(0.1, 0.2), (0.8, 0.2)] and z.direction == "a_to_b"
