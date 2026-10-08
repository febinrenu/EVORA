import json

import pytest
from contracts.models import Zone
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.core import cameras as cams
from evora.core import doctor, zones
from evora.evidence import audit

T0 = 1791450000.0


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "admin")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))
    db = c.app.state.ctx.db
    for name in ("Gate", "Lobby"):
        cams.insert_camera(db, name=name, kind="file", source_uri=f"/x/{name}.mp4", t0=T0, t0_source="manual", duration_s=600.0)
    return c


# ---- audit -----------------------------------------------------------------------------------------------------------

def test_the_audit_log_is_newest_first_and_filterable(client):
    db = client.app.state.ctx.db
    for i in range(5):
        audit.record(db, "export" if i % 2 == 0 else "unblur", {"n": i})
    rows = client.get("/api/audit").json()
    assert [r["detail"]["n"] for r in rows] == [4, 3, 2, 1, 0]
    assert set(rows[0]) == {"id", "actor", "action", "detail", "t"}
    assert [r["detail"]["n"] for r in client.get("/api/audit", params={"action": "export"}).json()] == [4, 2, 0]
    assert len(client.get("/api/audit", params={"limit": 2}).json()) == 2


@pytest.mark.parametrize("limit", [0, -1, 5000])
def test_audit_limit_is_bounded(client, limit):
    assert client.get("/api/audit", params={"limit": limit}).status_code == 422


# ---- past queries ---------------------------------------------------------------------------------------------------------

def log_query(client, qid, text, t, plan=None, answer=None, timings="{}"):
    with client.app.state.ctx.db.write() as c:
        c.execute(
            "INSERT INTO query_log(id,text,plan,answer,timings,created_at) VALUES(?,?,?,?,?,?)",
            (qid, text, plan if plan is not None else json.dumps({"intent": "exists"}),
             answer if answer is not None else json.dumps(
                 {"verdict": "yes", "count": None, "confidence": 0.8, "evidence": [{}, {}]}),
             timings, t),
        )


def test_past_queries_are_summarised_newest_first(client):
    log_query(client, "q1", "first", 1.0)
    log_query(client, "q2", "second", 2.0, timings=json.dumps({"total": 120.5}))
    rows = client.get("/api/queries").json()
    assert [r["id"] for r in rows] == ["q2", "q1"]
    assert rows[0] == {"id": "q2", "text": "second", "intent": "exists", "verdict": "yes", "count": None, "n_results": 2,
                       "confidence": 0.8, "timings_ms": {"total": 120.5}, "created_at": 2.0}
    assert "plan" not in rows[0] and "answer" not in rows[0]
    full = client.get("/api/queries", params={"full": True, "limit": 1}).json()
    assert len(full) == 1 and full[0]["plan"] == {"intent": "exists"} and full[0]["answer"]["verdict"] == "yes"


def test_a_damaged_row_still_lists(client):
    log_query(client, "q1", "broken", 1.0, plan="not json", answer="[1,2]", timings="nope")
    row = client.get("/api/queries").json()[0]
    assert row["text"] == "broken" and row["intent"] is None and row["verdict"] is None
    assert row["n_results"] == 0 and row["timings_ms"] == {}


# ---- zone events ----------------------------------------------------------------------------------------------------------

def add_event(client, event_id, camera, zone, kind, t, payload="{}"):
    with client.app.state.ctx.db.write() as c:
        c.execute(
            "INSERT INTO events(id,camera_id,track_id,kind,zone_id,t,payload) VALUES(?,?,?,?,?,?,?)",
            (event_id, camera, "cam_01:t1", kind, zone, t, payload),
        )


@pytest.fixture()
def gate(client):
    zones.save(client.app.state.ctx.db, Zone(id="z_gate", camera_id="cam_01", kind="line", points=[(0.1, 0.7), (0.9, 0.7)]))
    add_event(client, "e1", "cam_01", "z_gate", "cross_line", T0 + 10, json.dumps({"direction": "a_to_b"}))
    add_event(client, "e2", "cam_01", "z_gate", "dwell", T0 + 20)
    add_event(client, "e3", "cam_01", "z_gate", "cross_line", T0 + 30, "garbage")
    add_event(client, "e4", "cam_01", "z_other", "cross_line", T0 + 40)
    add_event(client, "e5", "cam_02", "z_gate", "cross_line", T0 + 50)
    return client


def test_a_zones_events_are_newest_first_and_only_its_own(gate):
    rows = gate.get("/api/zones/z_gate/events").json()
    assert [r["id"] for r in rows] == ["e3", "e2", "e1"]
    assert rows[2]["payload"] == {"direction": "a_to_b"} and rows[0]["payload"] == {}, "a damaged payload becomes empty"


def test_zone_events_filters(gate):
    ids = lambda **kw: [r["id"] for r in gate.get("/api/zones/z_gate/events", params=kw).json()]  # noqa: E731
    assert ids(kind="cross_line") == ["e3", "e1"]
    assert ids(since=T0 + 15) == ["e3", "e2"]
    assert ids(until=T0 + 15) == ["e1"]
    assert ids(limit=1) == ["e3"]


def test_zone_events_errors(gate):
    assert gate.get("/api/zones/nope/events").status_code == 404
    assert gate.get("/api/zones/z_gate/events", params={"limit": 0}).status_code == 422


# ---- identities -----------------------------------------------------------------------------------------------------------

def add_track(client, track_id, camera, start, end, gid, cls="person"):
    with client.app.state.ctx.db.write() as c:
        c.execute(
            "INSERT INTO tracks(id,camera_id,cls,t_start,t_end,n_obs,global_id) VALUES(?,?,?,?,?,5,?)",
            (track_id, camera, cls, T0 + start, T0 + end, gid),
        )


@pytest.fixture()
def people(client):
    with client.app.state.ctx.db.write() as c:
        c.execute("INSERT INTO global_ids(id,cls,label,created_at) VALUES('g_1','person','delivery driver',1.0)")
    add_track(client, "cam_01:t1", "cam_01", 10, 20, "g_1")
    add_track(client, "cam_02:t1", "cam_02", 100, 130, "g_1")
    add_track(client, "cam_01:t2", "cam_01", 50, 60, "g_2", cls="car")
    add_track(client, "cam_01:t3", "cam_01", 70, 80, None)
    return client


def test_identities_are_summarised_from_their_tracks(people):
    rows = people.get("/api/globals").json()
    assert [r["id"] for r in rows] == ["g_1", "g_2"], "most recently seen first; tracks without an identity are not listed"
    first = rows[0]
    assert first == {"id": "g_1", "cls": "person", "label": "delivery driver", "n_tracks": 2, "cameras": ["cam_01", "cam_02"],
                     "t_first": T0 + 10, "t_last": T0 + 130}
    assert rows[1]["cls"] == "car" and rows[1]["label"] is None


def test_identities_filter_and_detail(people):
    assert [r["id"] for r in people.get("/api/globals", params={"cls": "car"}).json()] == ["g_2"]
    assert people.get("/api/globals/g_1").json()["n_tracks"] == 2
    assert people.get("/api/globals/g_9").status_code == 404
    assert people.get("/api/globals/a b").status_code == 422
    assert people.get("/api/globals", params={"limit": 1}).json()[0]["id"] == "g_1"


def test_the_path_route_still_works_next_to_the_detail_route(people):
    assert people.get("/api/globals/g_1/path").status_code == 200


# ---- doctor ---------------------------------------------------------------------------------------------------------------

def test_the_doctor_route_runs_the_quick_checks_without_blocking(client, monkeypatch):
    seen = {}

    def fake(env, skip=frozenset()):
        seen["quick"], seen["on_prem"] = env.quick, env.on_prem
        return [
            doctor.Check("python", "Python", doctor.OK, "3.12"),
            doctor.Check("groq", "Groq keys", doctor.WARN, "none", "add one"),
        ]

    monkeypatch.setattr(doctor, "run_checks", fake)
    body = client.get("/api/doctor").json()
    assert seen == {"quick": True, "on_prem": False}
    assert body["ok"] is True and body["verdict"] == "1 warning, nothing blocking" and len(body["checks"]) == 2
    assert body["checks"][1]["fix"] == "add one" and body["took_s"] >= 0


def test_a_failing_check_is_reported_not_raised(client, monkeypatch):
    failing = [doctor.Check("ffmpeg", "ffmpeg", doctor.FAIL, "missing")]
    monkeypatch.setattr(doctor, "run_checks", lambda env, skip=frozenset(): failing)
    body = client.get("/api/doctor").json()
    assert body["ok"] is False and "1 problem" in body["verdict"]


def test_on_prem_is_passed_to_the_doctor(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(doctor, "run_checks", lambda env, skip=frozenset(): seen.update(on_prem=env.on_prem) or [])
    client.post("/api/settings", json={"onprem": True})
    client.get("/api/doctor")
    assert seen["on_prem"] is True


# ---- mock mode ------------------------------------------------------------------------------------------------------------

def test_mock_mode_returns_canned_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "adminmock")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", mock=True))
    assert c.get("/api/audit").json()[0]["action"] == "export"
    assert c.get("/api/queries").json()[0]["verdict"] == "yes"
    assert c.get("/api/doctor").json()["ok"] is True
    assert c.get("/api/globals").json()[0]["id"] == "g_000001" and c.get("/api/globals/g_7").json()["id"] == "g_7"
