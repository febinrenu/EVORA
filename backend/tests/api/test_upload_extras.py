import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.core import media
from evora.evidence import audit


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "extras")
    return TestClient(create_app(workspaces_root=tmp_path / "ws"))


def up(client, path, name):
    with open(path, "rb") as fh:
        return client.post("/api/cameras", files=[("files", (name, fh))])


def stored_files(client):
    return sorted(p.name for p in client.app.state.ctx.ws.uploads_dir.iterdir())


def test_identical_upload_reuses_the_camera(client, sample_mp4):
    first = up(client, sample_mp4, "a.mp4").json()[0]
    again = up(client, sample_mp4, "renamed.mp4")
    assert again.json()[0]["id"] == first["id"]
    assert json.loads(again.headers["x-evora-duplicate"]) == [first["id"]]
    assert len(client.get("/api/cameras").json()) == 1
    assert len(stored_files(client)) == 1


def test_native_codec_is_not_converted(client, sample_mp4):
    cam = up(client, sample_mp4, "a.mp4").json()[0]
    assert cam["source_uri"].endswith("_a.mp4")
    assert audit.entries(client.app.state.ctx.db, "transcode") == []


def test_unusual_codec_is_converted_to_h264(client, mpeg2_avi):
    r = up(client, mpeg2_avi, "legacy.avi")
    assert r.status_code == 200
    cam = r.json()[0]
    assert cam["source_uri"].endswith(".h264.mp4") and (cam["width"], cam["height"]) == (320, 240)
    assert media.probe(Path(cam["source_uri"])).codec == "h264"
    assert len(stored_files(client)) == 1, "the original upload is removed after conversion"
    # AVI keeps no creation time, so the clock falls back to the file time of the original upload
    assert cam["t0_source"] in {"metadata", "manual"} and cam["t0"] > 1_000_000_000
    entry = audit.entries(client.app.state.ctx.db, "transcode")[0]["detail"]
    assert entry["from"] == "mpeg2video" and len(entry["original_sha256"]) == 64


def test_converted_upload_is_deduplicated_by_original_hash(client, mpeg2_avi):
    first = up(client, mpeg2_avi, "legacy.avi").json()[0]
    again = up(client, mpeg2_avi, "legacy-copy.avi")
    assert again.json()[0]["id"] == first["id"] and "x-evora-duplicate" in again.headers
    assert len(stored_files(client)) == 1


def test_converted_clip_is_playable_end_to_end(client, mpeg2_avi):
    cid = up(client, mpeg2_avi, "legacy.avi").json()[0]["id"]
    client.post("/api/settings", json={"blur_faces": False})
    assert client.get(f"/api/cameras/{cid}/frame", params={"t": 1791450000.5}).status_code == 200
