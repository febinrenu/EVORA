import pytest
from contracts.models import Evidence, StreamEvent
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.api.query_wiring import CropSource
from evora.evidence import audit
from tests.api.fakes import FakeGateway, parse_sse
from tests.memory.conftest import FakeEmbedder

LINE = {"id": "c", "camera_id": "cam_01", "kind": "line", "points": [[0.1, 0.7], [0.9, 0.65]]}


def build(tmp_path, monkeypatch, gateway=None, **kw):
    monkeypatch.setenv("evora_WORKSPACE", "q")
    app = create_app(workspaces_root=tmp_path / "ws", gateway=gateway or FakeGateway(), embedder=FakeEmbedder(), **kw)
    return TestClient(app)


@pytest.fixture()
def client(tmp_path, monkeypatch, sample_mp4, sample_mp4_b):
    c = build(tmp_path, monkeypatch)
    for path, name in ((sample_mp4, "north.mp4"), (sample_mp4_b, "south.mp4")):
        with open(path, "rb") as fh:
            assert c.post("/api/cameras", files=[("files", (name, fh))]).status_code == 200
    return c


def ask(client, text="did a car pass the main gate?"):
    return parse_sse(client.post("/api/query", json={"text": text, "session_id": "s"}).text)


def types(events):
    return [name for name, _ in events]


def test_unknown_place_asks_once_with_every_camera(client):
    events = ask(client)
    assert types(events) == ["plan", "clarify"]
    clarify = events[-1][1]
    assert clarify["kind"] == "choose_camera" and clarify["referent"]["text"] == "main gate"
    assert [o["camera_id"] for o in clarify["options"]] == ["cam_01", "cam_02"]


def test_answering_resumes_the_question_and_it_is_not_asked_again(client):
    qid = ask(client)[-1][1]["query_id"]
    resumed = parse_sse(client.post("/api/clarify", json={"query_id": qid, "camera_id": "cam_01", "zone": LINE}).text)
    assert types(resumed)[0] == "plan" and types(resumed)[-1] == "done" and "clarify" not in types(resumed)
    answer = dict(resumed)["answer"]
    assert answer["verdict"] in {"no", "not_found"} and any("indexed" in n for n in answer["notes"])
    assert "clarify" not in types(ask(client))  # same question again: remembered
    facts = client.get("/api/memory").json()
    assert facts[0]["canonical"] == "main gate" and facts[0]["binding"]["camera_id"] == "cam_01"


def test_bad_typed_answer_is_422_and_the_question_stays_open(client):
    qid = ask(client)[-1][1]["query_id"]
    bad = client.post("/api/clarify", json={"query_id": qid, "text": "the one"})
    assert bad.status_code == 422 and "camera" in bad.json()["detail"]
    ok = client.post("/api/clarify", json={"query_id": qid, "text": "camera 2"})
    assert ok.status_code == 200 and types(parse_sse(ok.text))[-1] == "done"
    assert client.get("/api/memory").json()[0]["binding"]["camera_id"] == "cam_02"


def test_answering_twice_or_an_unknown_question_is_410(client):
    qid = ask(client)[-1][1]["query_id"]
    assert client.post("/api/clarify", json={"query_id": qid, "camera_id": "cam_01"}).status_code == 200
    assert client.post("/api/clarify", json={"query_id": qid, "camera_id": "cam_01"}).status_code == 410
    assert client.post("/api/clarify", json={"query_id": "nope", "camera_id": "cam_01"}).status_code == 410


def test_query_validation(client):
    for text in ("", "   ", "x" * 2001):
        assert client.post("/api/query", json={"text": text, "session_id": "s"}).status_code == 422


def test_a_planner_failure_is_an_error_event_then_done(tmp_path, monkeypatch):
    c = build(tmp_path, monkeypatch, gateway=FakeGateway(planner_error=True))
    events = ask(c, "something only a model can plan")
    assert types(events) == ["error", "done"] and "could not plan" in events[0][1]["message"]


def test_a_crash_inside_the_stream_never_drops_the_connection(client, monkeypatch):
    async def boom(text, session_id):
        yield StreamEvent(type="plan", data={})
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(client.app.state.ctx.router, "answer", boom)
    events = ask(client)
    assert types(events) == ["plan", "error", "done"]
    assert "secret" not in events[1][1]["message"]


def test_the_answer_event_pre_renders_the_top_evidence(client, monkeypatch):
    seen = []
    ctx = client.app.state.ctx
    monkeypatch.setattr(ctx.prerender, "schedule", lambda ids: seen.append(list(ids)))

    async def fixed(text, session_id):
        yield StreamEvent(type="plan", data={})
        yield StreamEvent(type="answer", data={"evidence": [{"id": "e1"}, {"id": "e2"}]})
        yield StreamEvent(type="done", data={})

    monkeypatch.setattr(ctx.router, "answer", fixed)
    ask(client)
    assert seen == [["e1", "e2"]]


def test_queries_and_clarifications_are_audited(client):
    qid = ask(client)[-1][1]["query_id"]
    client.post("/api/clarify", json={"query_id": qid, "camera_id": "cam_01"})
    db = client.app.state.ctx.db
    assert audit.entries(db, "query")[0]["detail"]["text"] == "did a car pass the main gate?"
    assert audit.entries(db, "clarify")[0]["detail"]["kind"] == "place"


def test_voice(client):
    assert client.post("/api/voice", content=b"ok-audio").json() == {"text": "red car at the main gate"}
    assert client.post("/api/voice", content=b"").status_code == 422
    assert client.post("/api/voice", content=b"bad").status_code == 502
    assert client.post("/api/voice", content=b"ok" + b"0" * (10 * 1024 * 1024)).status_code == 413


def test_mock_mode_serves_fixture_streams(tmp_path, monkeypatch):
    c = build(tmp_path, monkeypatch, mock=True)
    assert types(ask(c)) == ["plan", "evidence", "evidence", "evidence", "answer", "verified", "done"]
    assert types(ask(c, "who used the back entrance")) == ["plan", "clarify"]
    assert c.post("/api/voice", content=b"x").json()["text"]


# ---- what the verifier may send to a vision model ----------------------------------------------------------------

def evidence(track_id=None):
    return Evidence(
        id="e", camera_id="cam_01", camera_name="north", t_start=1791450000.0, t_end=1791450001.0, t_peak=1791450000.5,
        offset_s=0.5, track_id=track_id, thumb_url="/t", clip_url="/c", score=0.9,
    )


def with_track(ctx, best_crop):
    with ctx.db.write() as c:
        c.execute(
            "INSERT INTO tracks(id,camera_id,cls,t_start,t_end,n_obs,best_crop) VALUES('cam_01:t1','cam_01','car',1,2,5,?)",
            (best_crop,),
        )


def test_crop_source_blurs_before_anything_leaves(client):
    ctx = client.app.state.ctx
    (ctx.ws.media_dir / "crops").mkdir(parents=True, exist_ok=True)
    (ctx.ws.media_dir / "crops" / "c.jpg").write_bytes(b"RAW-CROP")
    with_track(ctx, "crops/c.jpg")
    src = CropSource(ctx)
    ctx.settings["blur_faces"] = True
    ctx.media._blur_provider = lambda: (lambda data: b"BLURRED:" + data)
    assert src.crop_for(evidence("cam_01:t1")) == b"BLURRED:RAW-CROP"
    ctx.media._blur_provider = lambda: None
    assert src.crop_for(evidence("cam_01:t1")) is None, "no blur available: nothing is sent"
    ctx.settings["blur_faces"] = False
    assert src.crop_for(evidence("cam_01:t1")) == b"RAW-CROP"


def test_crop_path_cannot_escape_the_media_folder(client, tmp_path):
    ctx = client.app.state.ctx
    secret = tmp_path / "ws" / "q" / "secret.jpg"
    secret.write_bytes(b"SECRET")
    with_track(ctx, "../secret.jpg")
    ctx.settings["blur_faces"] = False
    got = CropSource(ctx).crop_for(evidence("cam_01:t1"))
    assert got != b"SECRET" and (got is None or got[:2] == b"\xff\xd8"), "falls back to a frame, never the outside file"


def test_crop_source_falls_back_to_a_video_frame(client):
    client.app.state.ctx.settings["blur_faces"] = False
    got = CropSource(client.app.state.ctx).crop_for(evidence())
    assert got is not None and got[:2] == b"\xff\xd8"
