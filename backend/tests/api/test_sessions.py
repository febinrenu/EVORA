"""Sessions are workspace folders: listing, creating and switching, with the launcher's restart hook faked."""
from fastapi.testclient import TestClient

from evora.api.app import create_app


def client(tmp_path):
    app = create_app(workspaces_root=tmp_path / "ws", gateway=object())
    asked: list[str] = []
    app.state.switch = asked.append          # the launcher sets this; here it only records the request
    return TestClient(app), app, asked


def test_the_session_list_is_the_folders_on_disk_with_the_open_one_marked(tmp_path):
    c, app, _ = client(tmp_path)
    current = app.state.ctx.ws.slug
    (tmp_path / "ws" / "older-site").mkdir()
    rows = c.get("/api/workspaces").json()
    assert {r["slug"] for r in rows} == {current, "older-site"}
    assert [r["slug"] for r in rows if r["active"]] == [current]


def test_a_new_session_is_an_empty_folder_and_switching_is_only_asked_for_when_wanted(tmp_path):
    c, app, asked = client(tmp_path)
    made = c.post("/api/workspaces", json={"name": "Judge day 1"}).json()
    assert made["slug"] == "judge-day-1" and made["switching"] is False and asked == []
    assert (tmp_path / "ws" / "judge-day-1" / "uploads").is_dir()
    auto = c.post("/api/workspaces", json={"activate": True}).json()
    assert auto["slug"].startswith("session-") and auto["switching"] is True and asked == [auto["slug"]]


def test_activating_a_session_asks_the_launcher_to_switch(tmp_path):
    c, app, asked = client(tmp_path)
    c.post("/api/workspaces", json={"name": "other"})
    r = c.post("/api/workspaces/other/activate")
    assert r.status_code == 202 and r.json() == {"switching_to": "other"} and asked == ["other"]
    assert c.post("/api/workspaces/nowhere/activate").status_code == 404
    c.post(f"/api/workspaces/{app.state.ctx.ws.slug}/activate")
    assert asked == ["other"]                # the session that is already open needs no restart


def test_without_the_launcher_the_routes_keep_their_stub_behaviour(tmp_path):
    app = create_app(workspaces_root=tmp_path / "ws", gateway=object())
    c = TestClient(app)
    assert c.post("/api/workspaces", json={"name": "x"}).json()["slug"] == "x"
    assert c.post("/api/workspaces", json={}).status_code == 422
