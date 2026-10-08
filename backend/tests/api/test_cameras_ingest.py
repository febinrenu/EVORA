import json
import time

import pytest
from fastapi.testclient import TestClient

from evora.api.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "test")
    return TestClient(create_app(workspaces_root=tmp_path / "ws"))


def upload(client, path, name="clip.mp4"):
    with open(path, "rb") as fh:
        return client.post("/api/cameras", files=[("files", (name, fh, "video/mp4"))])


def test_upload_registers_camera(client, sample_mp4):
    r = upload(client, sample_mp4)
    assert r.status_code == 200
    cam = r.json()[0]
    assert cam["id"] == "cam_01" and cam["status"] == "pending" and cam["layers"] == []
    assert (cam["width"], cam["height"]) == (320, 240)
    assert cam["t0_source"] == "metadata" and cam["t0"] == pytest.approx(1791450000.0, abs=1)
    assert client.get("/api/cameras").json()[0]["id"] == "cam_01"


def test_upload_keeps_file_inside_workspace(client, sample_mp4, tmp_path):
    cam = upload(client, sample_mp4, "../../evil.mp4").json()[0]
    stored = (tmp_path / "ws" / "test" / "uploads")
    assert [p.name for p in stored.iterdir()] == [cam["source_uri"].replace("\\", "/").rsplit("/", 1)[-1]]
    assert "evil" in cam["name"]


@pytest.mark.parametrize("name,status", [("notes.txt", 422), ("a.exe", 422)])
def test_bad_extension_rejected(client, tmp_path, name, status):
    f = tmp_path / "f"
    f.write_bytes(b"x" * 50)
    assert upload(client, f, name).status_code == status


def test_corrupt_video_rejected_and_not_left_on_disk(client, tmp_path):
    f = tmp_path / "bad.mp4"
    f.write_bytes(b"not a video" * 50)
    r = upload(client, f, "bad.mp4")
    assert r.status_code == 422
    assert list((tmp_path / "ws" / "test" / "uploads").iterdir()) == []
    assert client.get("/api/cameras").json() == []


def test_oversized_upload_rejected(tmp_path, monkeypatch, sample_mp4):
    monkeypatch.setenv("evora_WORKSPACE", "small")
    import evora.api.app as appmod

    real = appmod.load_config

    def small():
        cfg = real()
        cfg["uploads"]["max_bytes"] = 100
        return cfg

    monkeypatch.setattr(appmod, "load_config", small)
    c = TestClient(create_app(workspaces_root=tmp_path / "ws"))
    assert upload(c, sample_mp4).status_code == 413
    assert list((tmp_path / "ws" / "small" / "uploads").iterdir()) == []


def test_partial_batch_returns_good_files_and_reports_rejects(client, sample_mp4, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"junk" * 50)
    with open(sample_mp4, "rb") as good, open(bad, "rb") as junk:
        r = client.post("/api/cameras", files=[("files", ("good.mp4", good)), ("files", ("bad.mp4", junk))])
    assert r.status_code == 200 and len(r.json()) == 1
    rejected = json.loads(r.headers["x-evora-rejected"])
    assert rejected[0]["filename"] == "bad.mp4"


def test_rtsp_camera(client):
    r = client.post("/api/cameras", json={"uri": "rtsp://10.0.0.2:8554/gate", "name": "Gate"})
    cam = r.json()[0]
    assert (cam["kind"], cam["t0_source"], cam["name"]) == ("rtsp", "live", "Gate")
    assert client.post("/api/cameras", json={"uri": "http://nope"}).status_code == 422


def test_no_files_is_an_error(client):
    assert client.post("/api/cameras", data={"x": "y"}).status_code == 422


def test_patch_camera(client, sample_mp4):
    cid = upload(client, sample_mp4).json()[0]["id"]
    r = client.patch(f"/api/cameras/{cid}", json={"name": "Main gate", "t0": 1700000000, "site_xy": [0.2, 0.5]})
    body = r.json()
    assert (body["name"], body["t0"], body["t0_source"], body["site_xy"]) == ("Main gate", 1700000000, "manual", [0.2, 0.5])
    assert client.patch(f"/api/cameras/{cid}", json={"name": " "}).status_code == 422
    assert client.patch(f"/api/cameras/{cid}", json={"site_xy": [1]}).status_code == 422
    assert client.patch("/api/cameras/cam_99", json={"name": "x"}).status_code == 404


def test_frame_requires_known_camera(client, sample_mp4):
    cid = upload(client, sample_mp4).json()[0]["id"]
    assert client.get(f"/api/cameras/{cid}/frame", params={"t": 1}).status_code == 200
    assert client.get("/api/cameras/cam_99/frame", params={"t": 1}).status_code == 404


def test_ingest_end_to_end_with_stub(client, sample_mp4):
    cid = upload(client, sample_mp4).json()[0]["id"]
    jobs = client.post("/api/ingest", json={"camera_ids": [cid], "layers": ["L0", "L1"]}).json()
    assert jobs[0]["camera_id"] == cid
    end = time.time() + 5
    while time.time() < end and client.get("/api/cameras").json()[0]["status"] != "ready":
        time.sleep(0.05)
    cam = client.get("/api/cameras").json()[0]
    assert cam["status"] == "ready" and cam["layers"] == ["L0", "L1"]


def test_ingest_validation(client):
    assert client.post("/api/ingest", json={"camera_ids": []}).status_code == 422
    assert client.post("/api/ingest", json={"camera_ids": ["cam_99"]}).status_code == 404
    assert client.post("/api/ingest", json={"camera_ids": ["cam_99"], "layers": "L0"}).status_code == 422
