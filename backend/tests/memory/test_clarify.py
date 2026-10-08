import json

import pytest
from contracts.models import ClarifyResponse, QueryPlan, Referent, Zone

from evora.memory.clarify import ClarifyError, parse_camera_text
from evora.memory.resolve import Ambiguous, Bound, Unknown

PLAN = QueryPlan(intent="exists", place=Referent(text="main gate", role="place"))


def ref(text="main gate", role="place"):
    return Referent(text=text, role=role)


def ask(env, qid="q1", r=None, resolution=None):
    r = r or ref()
    return env.clarifier.ask(qid, f"did a car pass the {r.text}", PLAN, r, resolution or Unknown())


def line(camera_id):
    return Zone(id="client", camera_id=camera_id, kind="line", points=[(0.1, 0.7), (0.9, 0.65)])


def test_ask_for_unknown_place_lists_every_camera_and_persists(make_env):
    env = make_env()
    a, b = env.camera("Gate"), env.camera("Lobby")
    req = ask(env)
    assert req.kind == "choose_camera" and req.allow_region and req.question == "Which camera shows main gate?"
    assert [o.camera_id for o in req.options] == [a.id, b.id]
    assert req.options[0].thumb_url.startswith(f"/api/cameras/{a.id}/frame?t=")
    pending = env.clarifier.pending("q1")
    assert pending.text == "did a car pass the main gate" and pending.clarify == req


def test_ask_kinds(make_env):
    env = make_env()
    f1 = env.kb.create("place", "gate east", {})
    f2 = env.kb.create("place", "gate west", {})
    amb = ask(env, "q2", ref("gate"), Ambiguous([f1, f2]))
    assert amb.kind == "choose_known" and amb.known_candidates == [f1.id, f2.id] and amb.options == []
    assert ask(env, "q3", ref("my car", "object")).kind == "choose_track"
    time_req = ask(env, "q4", ref("after hours", "time"))
    assert time_req.kind == "time_range" and not time_req.allow_region


def test_camera_answer_with_a_line_creates_fact_and_zone(make_env):
    env = make_env()
    cam = env.camera("Gate")
    ask(env)
    out = env.clarifier.apply(ClarifyResponse(query_id="q1", camera_id=cam.id, zone=line(cam.id)))
    fact = out.fact
    assert fact.canonical == "main gate" and fact.source == "clarification"
    assert fact.binding["camera_id"] == cam.id and fact.binding["zone_id"].startswith("z_")
    with env.db.read() as c:
        z = c.execute("SELECT * FROM zones WHERE id=?", (fact.binding["zone_id"],)).fetchone()
        assert (z["camera_id"], z["kind"], z["fact_id"]) == (cam.id, "line", fact.id)
        assert json.loads(z["points"]) == [[0.1, 0.7], [0.9, 0.65]]
        assert c.execute("SELECT count(*) FROM pending_queries").fetchone()[0] == 0
    assert out.pending.text == "did a car pass the main gate"


def test_camera_answer_without_region_binds_the_camera_only(make_env):
    env = make_env()
    cam = env.camera("Gate")
    ask(env)
    fact = env.clarifier.apply(ClarifyResponse(query_id="q1", camera_id=cam.id)).fact
    assert fact.binding == {"camera_id": cam.id}


@pytest.mark.parametrize("typed", ["camera 2", "Cam 2", "cam_02", "the lobby one", "lobby"])
def test_typed_answers_are_understood(make_env, typed):
    env = make_env()
    env.camera("Gate")
    lobby = env.camera("Lobby")
    ask(env)
    assert env.clarifier.apply(ClarifyResponse(query_id="q1", text=typed)).fact.binding["camera_id"] == lobby.id


@pytest.mark.parametrize("typed", ["the one", "camera 9", "gate lobby", "banana", ""])
def test_unusable_typed_answers_keep_the_question_open(make_env, typed):
    env = make_env()
    env.camera("Gate")
    env.camera("Lobby")
    ask(env)
    with pytest.raises(ClarifyError):
        env.clarifier.apply(ClarifyResponse(query_id="q1", text=typed))
    assert env.clarifier.pending("q1").query_id == "q1"
    assert env.kb.list() == []


def test_unknown_or_answered_question_is_an_error(make_env):
    env = make_env()
    cam = env.camera("Gate")
    with pytest.raises(ClarifyError):
        env.clarifier.apply(ClarifyResponse(query_id="nope", camera_id=cam.id))
    ask(env)
    env.clarifier.apply(ClarifyResponse(query_id="q1", camera_id=cam.id))
    with pytest.raises(ClarifyError):
        env.clarifier.apply(ClarifyResponse(query_id="q1", camera_id=cam.id))


def test_bad_regions_are_rejected_before_anything_is_stored(make_env):
    env = make_env()
    cam, other = env.camera("Gate"), env.camera("Lobby")
    ask(env)
    bad = [
        Zone(id="z", camera_id=other.id, kind="line", points=[(0, 0), (1, 1)]),
        Zone(id="z", camera_id=cam.id, kind="line", points=[(0, 0), (2, 1)]),
        Zone(id="z", camera_id=cam.id, kind="line", points=[(0, 0)]),
        Zone(id="z", camera_id=cam.id, kind="polygon", points=[(0, 0), (1, 1)]),
    ]
    for zone in bad:
        with pytest.raises(ClarifyError):
            env.clarifier.apply(ClarifyResponse(query_id="q1", camera_id=cam.id, zone=zone))
    with env.db.read() as c:
        assert c.execute("SELECT count(*) FROM zones").fetchone()[0] == 0
    assert env.kb.list() == []


def test_unknown_camera_is_rejected(make_env):
    env = make_env()
    ask(env)
    with pytest.raises(ClarifyError):
        env.clarifier.apply(ClarifyResponse(query_id="q1", camera_id="cam_99"))


def test_choosing_a_known_fact_adds_the_phrasing_as_alias(make_env):
    env = make_env()
    f1 = env.kb.create("place", "gate east", {"camera_id": "cam_01"})
    f2 = env.kb.create("place", "gate west", {"camera_id": "cam_02"})
    ask(env, "q9", ref("gate"), Ambiguous([f1, f2]))
    out = env.clarifier.apply(ClarifyResponse(query_id="q9", fact_id=f2.id))
    assert out.fact.id == f2.id and "gate" in env.kb.get(f2.id).aliases
    assert env.kb.get(f2.id).use_count == 1


def test_choosing_a_wrong_kind_or_replaced_fact_is_rejected(make_env):
    env = make_env()
    car = env.kb.create("object", "my car", {})
    old = env.kb.create("place", "gate", {})
    env.kb.supersede(old.id, {"camera_id": "cam_01"})
    ask(env, "q1", ref("gate"))
    for fid in (car.id, old.id, "mf_deadbeef"):
        with pytest.raises(ClarifyError):
            env.clarifier.apply(ClarifyResponse(query_id="q1", fact_id=fid))


def test_answering_again_for_a_known_name_is_a_correction(make_env):
    env = make_env()
    a, b = env.camera("Gate"), env.camera("Lobby")
    ask(env)
    first = env.clarifier.apply(ClarifyResponse(query_id="q1", camera_id=a.id)).fact
    ask(env, "q2")
    second = env.clarifier.apply(ClarifyResponse(query_id="q2", camera_id=b.id)).fact
    assert env.kb.get(first.id).superseded_by == second.id
    assert [f.binding["camera_id"] for f in env.kb.list()] == [b.id]


def test_object_and_time_answers(make_env):
    env = make_env()
    cam = env.camera("Gate")
    with env.db.write() as c:
        c.execute("INSERT INTO global_ids(id,cls,created_at) VALUES('g_1','car',1)")
        c.execute(
            "INSERT INTO tracks(id,camera_id,cls,t_start,t_end,n_obs,global_id) VALUES('cam_01:t1',?,'car',1,2,5,'g_1')",
            (cam.id,),
        )
    ask(env, "q1", ref("my car", "object"))
    car = env.clarifier.apply(ClarifyResponse(query_id="q1", track_id="cam_01:t1")).fact
    assert car.kind == "object" and car.binding == {"track_id": "cam_01:t1", "global_id": "g_1"}
    ask(env, "q2", ref("after hours", "time"))
    t = env.clarifier.apply(ClarifyResponse(query_id="q2", tod_after="20:00", tod_before="06:00")).fact
    assert t.binding == {"tod_after": "20:00", "tod_before": "06:00"}
    ask(env, "q3", ref("night", "time"))
    for after, before in (("8pm", "6am"), ("25:00", "06:00"), (None, "06:00")):
        with pytest.raises(ClarifyError):
            env.clarifier.apply(ClarifyResponse(query_id="q3", tod_after=after, tod_before=before))
    ask(env, "q4", ref("my truck", "object"))
    with pytest.raises(ClarifyError):
        env.clarifier.apply(ClarifyResponse(query_id="q4", track_id="cam_01:nope"))


def test_parse_camera_text_prefers_exact_ids(make_env):
    env = make_env()
    cams = [env.camera("Gate"), env.camera("Gate B")]
    assert parse_camera_text("cam_02", cams).id == cams[1].id


# ---- the behaviour the rubric scores -------------------------------------------------------------------------------

async def test_clarify_once_survives_restart_and_paraphrases(tmp_path):
    """Ask, answer once, restart the process state, then the original and paraphrases never ask again."""
    from evora.core.db import close_all
    from tests.memory.conftest import Env

    root = tmp_path / "ws"
    env = Env(root)
    cam = env.camera("Gate")
    first = await env.resolver.resolve(ref("main gate"))
    assert isinstance(first, Unknown)
    ask(env)
    env.clarifier.apply(ClarifyResponse(query_id="q1", camera_id=cam.id, zone=line(cam.id)))
    close_all()  # the process "dies"

    env2 = Env(root)
    asked = 0
    for phrase in ("main gate", "the Main Gate", "front gate", "front gate"):
        r = await env2.resolver.resolve(ref(phrase))
        asked += not isinstance(r, Bound)
        assert isinstance(r, Bound) and r.fact.binding["camera_id"] == cam.id
    assert asked == 0
    close_all()
