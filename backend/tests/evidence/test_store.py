import json

import pytest
from contracts.models import Evidence

from evora.core import cameras as cams
from evora.core import workspace as wsmod
from evora.core.db import open_db
from evora.evidence import audit, store


@pytest.fixture()
def db(tmp_path):
    d = open_db(wsmod.create("t", tmp_path / "ws").db_path)
    cams.insert_camera(d, name="Gate", kind="file", source_uri="/x.mp4", t0=100.0, t0_source="manual")
    return d


def ev(eid="ev_001", cam="cam_01", bbox=(0.1, 0.2, 0.3, 0.4)):
    return Evidence(
        id=eid, camera_id=cam, camera_name="Gate", t_start=101.0, t_end=105.0, t_peak=103.0, offset_s=3.0,
        bbox=bbox, thumb_url="/t", clip_url="/c", score=0.9,
    )


def test_register_and_get_roundtrip(db):
    store.register(db, ev())
    rec = store.get(db, "ev_001")
    assert (rec.camera_id, rec.t_peak, rec.bbox) == ("cam_01", 103.0, (0.1, 0.2, 0.3, 0.4))


def test_register_twice_refreshes(db):
    store.register(db, ev())
    store.register(db, ev(bbox=None))
    assert store.get(db, "ev_001").bbox is None


@pytest.mark.parametrize("bad", ["../x", "a;b", "a b", "", "x..y", ".hidden", "a/b"])
def test_invalid_ids_are_rejected(db, bad):
    with pytest.raises(store.EvidenceError):
        store.get(db, bad)


def test_register_requires_known_camera(db):
    with pytest.raises(store.EvidenceError):
        store.register(db, ev(cam="cam_99"))


def test_unknown_id(db):
    with pytest.raises(store.EvidenceNotFound):
        store.get(db, "ev_404")


def test_fallback_finds_evidence_inside_a_stored_answer(db):
    answer = {"query_id": "q", "evidence": [ev("ev_a").model_dump()], "nearest_miss": ev("ev_miss").model_dump()}
    with db.write() as c:
        c.execute("INSERT INTO query_log(id,text,answer,created_at) VALUES('q','t',?,1)", (json.dumps(answer),))
    assert store.get(db, "ev_a").t_peak == 103.0
    assert store.get(db, "ev_miss").camera_id == "cam_01"
    with db.read() as c:  # found ids are registered for next time
        assert c.execute("SELECT count(*) FROM evidence").fetchone()[0] == 2


def test_fallback_finds_evidence_inside_an_alert(db):
    with db.write() as c:
        c.execute(
            "INSERT INTO alerts(id,sq_id,t,camera_id,evidence) VALUES('a','s',1,'cam_01',?)",
            (json.dumps(ev("ev_alert").model_dump()),),
        )
    assert store.get(db, "ev_alert").track_id is None


def test_audit_log_roundtrip(db):
    audit.record(db, "unblur_token", {"reason": "review"})
    audit.record(db, "other", {})
    rows = audit.entries(db, "unblur_token")
    assert len(rows) == 1 and rows[0]["detail"] == {"reason": "review"} and rows[0]["actor"] == "local"
