import pytest

from evora.core import cameras as cams
from evora.core import workspace as wsmod
from evora.core.db import open_db
from evora.evidence import builder, store

T0 = 1000.0


@pytest.fixture()
def db(tmp_path):
    d = open_db(wsmod.create("b", tmp_path / "ws").db_path)
    cams.insert_camera(d, name="Gate", kind="file", source_uri="/x.mp4", t0=T0, t0_source="manual", duration_s=600.0)
    return d


def add_track(
    db, track_id="cam_01:t000001", start=T0 + 100, end=T0 + 130, best_t=None, best_bbox=None, global_id=None, points=(),
):
    with db.write() as c:
        c.execute(
            "INSERT INTO tracks(id,camera_id,cls,t_start,t_end,n_obs,best_t,best_bbox,global_id) VALUES(?,?,?,?,?,?,?,?,?)",
            (track_id, "cam_01", "person", start, end, 10, best_t, best_bbox, global_id),
        )
        for t, box in points:
            c.execute("INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,?,?,?,?,0.9)", (track_id, t, *box))


def test_peak_window_offset_box_and_registration(db):
    add_track(db, best_t=T0 + 110, global_id="g_1", points=[(T0 + 100, (0.1, 0.1, 0.2, 0.3)), (T0 + 111, (0.3, 0.4, 0.5, 0.9))])
    ev = builder.evidence_for_track(db, "cam_01:t000001", score=0.83, why=["a reason"])
    assert (ev.id, ev.camera_name, ev.track_id, ev.global_id) == ("cam_01_t000001", "Gate", "cam_01:t000001", "g_1")
    assert (ev.t_peak, ev.t_start, ev.t_end, ev.offset_s) == (T0 + 110, T0 + 105, T0 + 115, 110.0)
    assert ev.bbox == (0.3, 0.4, 0.5, 0.9), "the box nearest the peak"
    assert ev.thumb_url == "/api/media/thumb/cam_01_t000001.jpg" and ev.score == 0.83 and ev.why == ["a reason"]
    assert store.get(db, "cam_01_t000001").t_peak == T0 + 110, "registered, so the media routes can render it"


def test_without_a_best_moment_the_middle_of_the_span_is_used_and_the_window_is_clamped(db):
    add_track(db, start=T0 + 100, end=T0 + 103)
    ev = builder.evidence_for_track(db, "cam_01:t000001")
    assert ev.t_peak == T0 + 101.5 and (ev.t_start, ev.t_end) == (T0 + 100, T0 + 103)


def test_a_best_moment_outside_the_span_is_pulled_inside(db):
    add_track(db, best_t=T0 + 999)
    assert builder.evidence_for_track(db, "cam_01:t000001").t_peak == T0 + 130


def test_the_stored_best_box_is_the_fallback(db):
    add_track(db, best_t=T0 + 110, best_bbox="[0.1, 0.2, 0.3, 0.4]")
    assert builder.evidence_for_track(db, "cam_01:t000001").bbox == (0.1, 0.2, 0.3, 0.4)
    add_track(db, "cam_01:t000002", best_bbox="not json")
    assert builder.evidence_for_track(db, "cam_01:t000002").bbox is None


def test_a_custom_id_and_unknown_track(db):
    add_track(db)
    assert builder.evidence_for_track(db, "cam_01:t000001", evidence_id="ev_custom").id == "ev_custom"
    with pytest.raises(builder.TrackNotFound):
        builder.evidence_for_track(db, "cam_01:t999999")
    assert builder.evidence_id_for("cam_01:t000012") == "cam_01_t000012"
