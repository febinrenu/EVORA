import json

import pytest
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.core import cameras as cams
from evora.core import perception_adapter as adapter
from evora.evidence import store

T0 = 1791450000.0


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "tracks")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))
    ctx = c.app.state.ctx
    for name in ("Gate", "Lobby", "Rear"):
        cams.insert_camera(
            ctx.db, name=name, kind="file", source_uri=f"/x/{name}.mp4", t0=T0, t0_source="manual", duration_s=600.0,
        )
    return c


def add_track(client, track_id, camera, start, end, global_id=None, attrs=None, quality=0.5):
    with client.app.state.ctx.db.write() as c:
        c.execute(
            "INSERT INTO tracks(id,camera_id,cls,t_start,t_end,n_obs,attrs,global_id,best_t,quality,direction) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (track_id, camera, "person", T0 + start, T0 + end, 8, json.dumps(attrs or {}), global_id,
             T0 + (start + end) / 2, quality, "left_to_right"),
        )
        for i in range(3):
            c.execute(
                "INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,?,?,?,?,0.9)",
                (track_id, T0 + start + i, 0.3, 0.4, 0.5, 0.9),
            )


def test_a_track_with_its_attributes_and_points(client):
    add_track(client, "cam_01:t000001", "cam_01", 10, 20, global_id="g_1", attrs={"color": "red", "carrying": ["backpack"]})
    body = client.get("/api/tracks/cam_01:t000001").json()
    assert (body["camera_name"], body["cls"], body["global_id"], body["direction"]) == ("Gate", "person", "g_1", "left_to_right")
    assert body["attrs"] == {"color": "red", "carrying": ["backpack"]}
    assert len(body["points"]) == 3 and body["points"][0]["bbox"] == [0.3, 0.4, 0.5, 0.9]
    assert body["t_start"] == T0 + 10 and body["n_obs"] == 8


@pytest.mark.parametrize("track_id,status", [("cam_01:t999", 404), ("a;b", 422), ("cam_01:tabc", 422)])
def test_track_errors(client, track_id, status):
    assert client.get(f"/api/tracks/{track_id}").status_code == status


def fake_similar(monkeypatch, result):
    seen = {}

    def fn(track_id, k, workspace=None, store=None):
        seen.update(track_id=track_id, k=k, workspace=workspace, store=store)
        return result

    monkeypatch.setattr(adapter, "similar_tracks", fn)
    return seen


def test_similar_tracks_become_registered_evidence_with_their_scores(client, monkeypatch):
    for i, cam in enumerate(("cam_01", "cam_02", "cam_03"), start=1):
        add_track(client, f"{cam}:t00000{i}", cam, 10 * i, 10 * i + 8)
    seen = fake_similar(monkeypatch, [("cam_02:t000002", 0.91), ("cam_03:t000003", 0.77)])
    r = client.get("/api/tracks/cam_01:t000001/similar", params={"k": 5})
    assert r.status_code == 200
    items = r.json()
    assert [e["id"] for e in items] == ["cam_02_t000002", "cam_03_t000003"] and [e["score"] for e in items] == [0.91, 0.77]
    assert items[0]["camera_name"] == "Lobby" and "similarity 0.91" in items[0]["why"][0]
    assert seen["track_id"] == "cam_01:t000001" and seen["k"] == 5 and seen["store"] is not None and seen["workspace"] is not None
    ctx = client.app.state.ctx
    assert store.get(ctx.db, "cam_02_t000002").camera_id == "cam_02", "playable: the media routes can resolve it"


def test_similar_skips_tracks_that_no_longer_exist(client, monkeypatch):
    add_track(client, "cam_02:t000002", "cam_02", 10, 18)
    fake_similar(monkeypatch, [("cam_09:t000009", 0.99), ("cam_02:t000002", 0.8)])
    assert [e["id"] for e in client.get("/api/tracks/cam_01:t000001/similar").json()] == ["cam_02_t000002"]


def test_similar_without_reid_is_empty_and_pending(client, monkeypatch):
    fake_similar(monkeypatch, None)
    r = client.get("/api/tracks/cam_01:t000001/similar")
    assert r.status_code == 200 and r.json() == [] and r.headers["x-evora-reid"] == "pending"


@pytest.mark.parametrize("k", [0, 101, -1])
def test_similar_bounds(client, k):
    assert client.get("/api/tracks/cam_01:t000001/similar", params={"k": k}).status_code == 422


def test_similar_with_an_unsafe_id(client):
    assert client.get("/api/tracks/a;b/similar").status_code == 422


def test_a_path_merges_runs_orders_hops_and_registers_every_hop(client):
    add_track(client, "cam_01:t000001", "cam_01", 10, 20, global_id="g_7", quality=0.4)
    add_track(client, "cam_01:t000002", "cam_01", 22, 30, global_id="g_7", quality=0.9)
    add_track(client, "cam_02:t000003", "cam_02", 60, 75, global_id="g_7")
    add_track(client, "cam_03:t000004", "cam_03", 120, 140, global_id="g_7")
    add_track(client, "cam_03:t000005", "cam_03", 20, 30, global_id="g_other")
    hops = client.get("/api/globals/g_7/path").json()
    assert [h["camera_name"] for h in hops] == ["Gate", "Lobby", "Rear"], "two tracks on one camera are one hop"
    assert hops[0]["t_in"] == T0 + 10 and hops[0]["t_out"] == T0 + 30
    assert hops[0]["evidence_id"] == "cam_01_t000002", "the best-quality track represents the hop"
    ctx = client.app.state.ctx
    for h in hops:
        assert store.get(ctx.db, h["evidence_id"]).camera_id == h["camera_id"], "every hop is playable"


def test_path_errors_and_pending(client, monkeypatch):
    assert client.get("/api/globals/g_none/path").status_code == 404
    assert client.get("/api/globals/a;b/path").status_code == 422
    add_track(client, "cam_01:t000001", "cam_01", 10, 20, global_id="g_1")
    monkeypatch.setattr(adapter, "path_for", lambda gid, workspace=None, db=None: None)
    r = client.get("/api/globals/g_1/path")
    assert r.status_code == 200 and r.json() == [] and r.headers["x-evora-reid"] == "pending"


def test_mock_mode_keeps_the_fixtures(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "trackmock")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", mock=True))
    assert c.get("/api/tracks/anything").json()["cls"] == "person"
    assert len(c.get("/api/tracks/x/similar", params={"k": 2}).json()) == 2
    assert c.get("/api/globals/g/path").json()[0]["camera_id"] == "cam_01"
