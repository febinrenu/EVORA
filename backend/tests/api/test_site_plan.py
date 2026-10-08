import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from evora.api.app import create_app


def picture(kind="PNG", size=(40, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 220, 230)).save(buf, kind)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "siteplan")
    return TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))


def test_put_get_replace_delete(client):
    assert client.get("/api/site/plan").status_code == 404
    r = client.put("/api/site/plan", content=picture("PNG"))
    assert r.status_code == 200 and r.json()["type"] == "image/png" and (r.json()["width"], r.json()["height"]) == (40, 30)
    got = client.get("/api/site/plan")
    assert got.status_code == 200 and got.headers["content-type"] == "image/png" and got.content == picture("PNG")
    jpeg = picture("JPEG", (64, 48))
    assert client.put("/api/site/plan", content=jpeg).json()["type"] == "image/jpeg"
    got = client.get("/api/site/plan")
    assert got.headers["content-type"] == "image/jpeg" and got.content == jpeg
    root = client.app.state.ctx.ws.root
    assert [p.name for p in root.glob("site_plan.*")] == ["site_plan.jpg"], "the replaced picture is gone"
    assert client.delete("/api/site/plan").status_code == 204 and client.get("/api/site/plan").status_code == 404
    assert client.delete("/api/site/plan").status_code == 204, "deleting twice is fine"


def test_it_is_kept_per_workspace_across_a_restart(client, tmp_path):
    client.put("/api/site/plan", content=picture())
    again = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))
    assert again.get("/api/site/plan").status_code == 200


@pytest.mark.parametrize("body,status", [
    (b"", 422), (b"not a picture at all", 415), (b"GIF89a" + b"\x00" * 20, 415),
])
def test_bad_pictures_are_refused(client, body, status):
    assert client.put("/api/site/plan", content=body).status_code == status
    assert client.get("/api/site/plan").status_code == 404


def test_a_gif_or_huge_picture_is_refused(client, monkeypatch):
    from evora.api import routes_site

    assert client.put("/api/site/plan", content=picture("GIF")).status_code == 415
    monkeypatch.setattr(routes_site, "MAX_BYTES", 100)
    assert client.put("/api/site/plan", content=picture()).status_code == 413
