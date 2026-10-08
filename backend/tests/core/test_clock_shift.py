import json

import pytest
from contracts.models import Alert, Evidence
from fastapi.testclient import TestClient

from evora.alerts import store as alert_store
from evora.api.app import create_app
from evora.core import cameras as cams
from evora.core import clock_shift
from evora.core import workspace as wsmod
from evora.core.db import open_db
from evora.core.vectors import open_store
from evora.evidence import audit
from evora.evidence import store as evidence_store

T0 = 1791450000.0


def evidence(eid: str, cam: str, t: float) -> Evidence:
    return Evidence(
        id=eid, camera_id=cam, camera_name=cam, t_start=t - 1, t_end=t + 1, t_peak=t, offset_s=t - T0,
        thumb_url=f"/api/media/thumb/{eid}.jpg", clip_url=f"/api/media/clip/{eid}.mp4", score=1.0,
    )


class World:
    """Two indexed cameras with rows in every table that holds a time."""

    def __init__(self, root):
        self.ws = wsmod.create("clock", root)
        self.db = open_db(self.ws.db_path)
        for name in ("Gate", "Lobby"):
            cams.insert_camera(self.db, name=name, kind="file", source_uri=f"/x/{name}.mp4", t0=T0, t0_source="metadata")
        with self.db.write() as c:
            for cid in ("cam_01", "cam_02"):
                c.execute("INSERT INTO tracks(id,camera_id,cls,t_start,t_end,n_obs,best_t) VALUES(?,?,?,?,?,5,?)",
                          (f"{cid}:t1", cid, "person", T0 + 10, T0 + 20, T0 + 15))
                c.execute("INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,0,0,1,1,1)", (f"{cid}:t1", T0 + 12))
                c.execute("INSERT INTO events(id,camera_id,track_id,kind,zone_id,t,payload) VALUES(?,?,?,?,?,?,'{}')",
                          (f"e_{cid}", cid, f"{cid}:t1", "appear", None, T0 + 10))
        for cid in ("cam_01", "cam_02"):
            evidence_store.register(self.db, evidence(f"ev_{cid}", cid, T0 + 15))
            alert = Alert(id=f"al_{cid}", standing_query_id="sq_1", t=T0 + 15, camera_id=cid,
                          evidence=evidence(f"ev_al_{cid}", cid, T0 + 15))
            alert_store.insert_alert(self.db, alert, f"{cid}:t1")
        store = open_store(self.ws.vectors_dir)
        rows = [{"vector": [0.1, 0.2], "camera_id": cid, "t": T0 + 12} for cid in ("cam_01", "cam_02")]
        for name in ("crops", "scenes", "captions"):
            store.create_table(name, data=rows)
        store.create_table("reid", data=[{"vector": [0.1, 0.2], "camera_id": cid, "t_start": T0 + 10, "t_end": T0 + 20}
                                          for cid in ("cam_01", "cam_02")])
        self.store = store

    def times(self, cid: str) -> dict:
        with self.db.read() as c:
            track = c.execute("SELECT t_start, t_end, best_t FROM tracks WHERE camera_id=?", (cid,)).fetchone()
            point = c.execute("SELECT t FROM track_points WHERE track_id=?", (f"{cid}:t1",)).fetchone()["t"]
            event = c.execute("SELECT t FROM events WHERE camera_id=?", (cid,)).fetchone()["t"]
            ev = c.execute("SELECT t_peak FROM evidence WHERE id=?", (f"ev_{cid}",)).fetchone()["t_peak"]
            alert = c.execute("SELECT t, evidence FROM alerts WHERE camera_id=?", (cid,)).fetchone()
        vec = {name: next(r for r in self.store.open_table(name).to_arrow().to_pylist() if r["camera_id"] == cid)
               for name in ("crops", "scenes", "captions", "reid")}
        return {
            "track": tuple(track), "point": point, "event": event, "evidence": ev, "alert": alert["t"],
            "alert_peak": json.loads(alert["evidence"])["t_peak"], "crops": vec["crops"]["t"], "scenes": vec["scenes"]["t"],
            "captions": vec["captions"]["t"], "reid": (vec["reid"]["t_start"], vec["reid"]["t_end"]),
            "t0": cams.get_camera(self.db, cid).t0,
        }


@pytest.fixture()
def world(tmp_path):
    return World(tmp_path / "ws")


def shifted(before: dict, delta: float) -> dict:
    out = {}
    for k, v in before.items():
        out[k] = tuple(x + delta for x in v) if isinstance(v, tuple) else v + delta
    return out


def test_every_stored_time_moves_with_the_clock_and_only_for_that_camera(world):
    before, other = world.times("cam_01"), world.times("cam_02")
    result = clock_shift.set_camera_clock(world.db, world.ws.vectors_dir, "cam_01", T0 + 3600.5)
    assert result["shift_s"] == 3600.5 and result["rows"]["tracks"] == 1 and result["rows"]["alerts"] == 1
    assert sorted(result["vector_tables"]) == ["captions", "crops", "reid", "scenes"]
    after = world.times("cam_01")
    for key, value in shifted(before, 3600.5).items():
        assert after[key] == pytest.approx(value), key
    assert world.times("cam_02") == other, "the other camera is untouched"
    assert cams.get_camera(world.db, "cam_01").t0_source == "manual"


def test_moving_back_restores_the_original_times(world):
    before = world.times("cam_01")
    clock_shift.set_camera_clock(world.db, world.ws.vectors_dir, "cam_01", T0 - 7200)
    clock_shift.set_camera_clock(world.db, world.ws.vectors_dir, "cam_01", T0)
    after = world.times("cam_01")
    assert all(after[k] == pytest.approx(v) for k, v in before.items())


def test_the_change_is_audited(world):
    clock_shift.set_camera_clock(world.db, world.ws.vectors_dir, "cam_01", T0 + 60, source="osd", actor="clock reader")
    entry = audit.entries(world.db, "clock_change")[0]
    assert entry["actor"] == "clock reader"
    detail = entry["detail"]
    assert detail["shift_s"] == 60 and detail["from_source"] == "metadata" and detail["to_source"] == "osd"


def test_the_same_clock_only_records_the_source(world):
    before = world.times("cam_01")
    result = clock_shift.set_camera_clock(world.db, world.ws.vectors_dir, "cam_01", T0, source="manual")
    assert result == {"shift_s": 0.0, "rows": {}, "vector_tables": []} and world.times("cam_01") == before


def test_a_camera_being_indexed_keeps_its_clock(world):
    with world.db.write() as c:
        c.execute("INSERT INTO ingest_jobs(id,camera_id,state,progress,updated_at) VALUES('j1','cam_01','running',0.4,1)")
    with pytest.raises(clock_shift.ClockBusy):
        clock_shift.set_camera_clock(world.db, world.ws.vectors_dir, "cam_01", T0 + 60)
    assert cams.get_camera(world.db, "cam_01").t0 == T0


def test_a_failure_in_the_database_puts_the_vectors_back(world, monkeypatch):
    before = world.times("cam_01")

    def broken(db, camera_id, delta):
        raise RuntimeError("disk full")

    monkeypatch.setattr(clock_shift, "_shift_rows", broken)
    with pytest.raises(RuntimeError):
        clock_shift.set_camera_clock(world.db, world.ws.vectors_dir, "cam_01", T0 + 60)
    assert world.times("cam_01") == before


def test_a_workspace_without_vector_tables_still_works(tmp_path):
    ws = wsmod.create("bare", tmp_path / "ws")
    db = open_db(ws.db_path)
    cams.insert_camera(db, name="Gate", kind="file", source_uri="/x/g.mp4", t0=T0, t0_source="manual")
    assert clock_shift.set_camera_clock(db, ws.vectors_dir, "cam_01", T0 + 5)["vector_tables"] == []


@pytest.mark.parametrize("cid", ["../x", "cam_01' OR '1'='1", ""])
def test_bad_camera_ids_are_refused(world, cid):
    with pytest.raises(ValueError):
        clock_shift.set_camera_clock(world.db, world.ws.vectors_dir, cid, T0)


# ---- through the API --------------------------------------------------------------------------------------------------------

@pytest.fixture()
def api(tmp_path, monkeypatch, sample_mp4):
    monkeypatch.setenv("evora_WORKSPACE", "clockapi")
    client = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))
    with open(sample_mp4, "rb") as fh:
        client.post("/api/cameras", files=[("files", ("gate.mp4", fh))])
    client.post("/api/settings", json={"blur_faces": False})
    return client


def test_after_a_clock_correction_the_evidence_still_shows_the_same_moment(api):
    ctx = api.app.state.ctx
    cam = cams.get_camera(ctx.db, "cam_01")
    ev = evidence("ev_x", "cam_01", cam.t0 + 1.2).model_copy(update={"t_start": cam.t0 + 0.8, "t_end": cam.t0 + 1.6})
    evidence_store.register(ctx.db, ev)
    frame_before = api.get("/api/media/thumb/ev_x.jpg").content
    for f in ctx.ws.media_dir.glob("thumbs/*"):
        f.unlink()  # render again after the change instead of serving the cached file
    r = api.patch("/api/cameras/cam_01", json={"t0": cam.t0 + 86400})
    assert r.status_code == 200 and r.json()["t0"] == cam.t0 + 86400 and r.json()["t0_source"] == "manual"
    assert evidence_store.get(ctx.db, "ev_x").t_peak == pytest.approx(cam.t0 + 86400 + 1.2)
    assert api.get("/api/media/thumb/ev_x.jpg").content == frame_before, "the same frame of the file, on the new clock"


def test_the_api_refuses_a_clock_change_during_indexing(api):
    ctx = api.app.state.ctx
    with ctx.db.write() as c:
        c.execute("INSERT INTO ingest_jobs(id,camera_id,state,progress,updated_at) VALUES('j1','cam_01','queued',0,1)")
    r = api.patch("/api/cameras/cam_01", json={"t0": 1.0})
    assert r.status_code == 409 and "being indexed" in r.json()["detail"]


@pytest.mark.parametrize("bad", ["noon", True, [1]])
def test_the_api_validates_the_clock(api, bad):
    assert api.patch("/api/cameras/cam_01", json={"t0": bad}).status_code == 422
