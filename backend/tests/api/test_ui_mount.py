import pytest
from fastapi.testclient import TestClient

from evora.api.app import create_app


@pytest.fixture()
def dist(tmp_path):
    d = tmp_path / "site" / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text("<html><body>evora ui</body></html>")
    (d / "assets" / "app.js").write_text("console.log('app')")
    (tmp_path / "site" / "secret.txt").write_text("SECRET")
    return d


def client(tmp_path, monkeypatch, ui_dir, **kw):
    monkeypatch.setenv("evora_WORKSPACE", "uimount")
    return TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object(), ui_dir=ui_dir, **kw))


def test_the_page_and_its_assets_are_served(tmp_path, monkeypatch, dist):
    c = client(tmp_path, monkeypatch, dist)
    assert "evora ui" in c.get("/").text
    assert c.get("/assets/app.js").text == "console.log('app')"


def test_client_side_routes_fall_back_to_the_page(tmp_path, monkeypatch, dist):
    c = client(tmp_path, monkeypatch, dist)
    r = c.get("/report")
    assert r.status_code == 200 and "evora ui" in r.text


def test_the_api_is_untouched_and_unknown_api_paths_stay_json_404(tmp_path, monkeypatch, dist):
    c = client(tmp_path, monkeypatch, dist)
    assert c.get("/api/health").json()["ok"] is True
    miss = c.get("/api/nope")
    assert miss.status_code == 404 and miss.json() == {"detail": "Not Found"}
    assert c.get("/api/cameras").json() == []


@pytest.mark.parametrize("path", ["/..%2fsecret.txt", "/%2e%2e/secret.txt", "/assets/..%2f..%2fsecret.txt", "/../secret.txt"])
def test_files_outside_the_build_are_never_served(tmp_path, monkeypatch, dist, path):
    c = client(tmp_path, monkeypatch, dist)
    assert "SECRET" not in c.get(path).text


def test_no_mount_without_a_build(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch, tmp_path / "nothing")
    assert c.get("/").status_code == 404 and c.get("/api/health").status_code == 200


def test_mock_mode_never_mounts(tmp_path, monkeypatch, dist):
    c = client(tmp_path, monkeypatch, dist, mock=True)
    assert c.get("/").status_code == 404 and c.get("/api/health").status_code == 200


def test_the_ui_can_be_switched_off_in_config(tmp_path, monkeypatch, dist):
    import evora.api.app as appmod

    real = appmod.load_config

    def off(profile=None):
        cfg = real(profile)
        cfg["server"]["serve_ui"] = False
        return cfg

    monkeypatch.setattr(appmod, "load_config", off)
    assert client(tmp_path, monkeypatch, dist).get("/").status_code == 404
