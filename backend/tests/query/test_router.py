from dataclasses import dataclass

import pytest
from contracts.models import ClarifyRequest, ClarifyResponse, MemoryFact, QueryPlan, Referent, Target, TimeWindow

from evora.llm.schemas import LLMError
from evora.query.fuse import Calibration
from evora.query.planner import Planner
from evora.query.retrieve import RetrievalConfig, Retriever
from evora.query.router import Router, RouterConfig
from tests.query.ws_helpers import E, ToyEmbedder, Workspace

RED_CAR = {"color": "red", "color_conf": 1.0}
BLUE_CAR = {"color": "blue", "color_conf": 1.0}
GATE_Q = "did a red car pass through the main gate"


@dataclass
class Res:
    status: str
    facts: tuple = ()


def place_fact(cam="cam_01", zone="z1", fid="mf_1"):
    return MemoryFact(id=fid, kind="place", canonical="main gate", binding={"camera_id": cam, "zone_id": zone},
                      source="clarification", created_at=0.0)


class FakeResolver:
    def __init__(self, known=None):
        self.known = dict(known or {})

    def resolve(self, ref):
        fact = self.known.get(ref.text)
        return Res("bound", (fact,)) if fact else Res("unknown")


class FakeClarifier:
    def __init__(self, resolver, bind_on_resume=None):
        self.resolver, self.bind, self.pending, self.asked = resolver, bind_on_resume, {}, []

    def ask(self, query_id, text, plan, ref, resolution, options):
        self.pending[query_id] = (text, plan)
        self.asked.append((ref.text, [o.camera_id for o in options]))
        return ClarifyRequest(query_id=query_id, referent=ref, question=f"Which camera shows {ref.text}?",
                              kind="choose_camera", options=options)

    def resume(self, resp):
        pending = self.pending.pop(resp.query_id, None)
        if pending and self.bind:
            self.resolver.known.update(self.bind)
        return pending


class FakeVerifier:
    def __init__(self, results=None, error=None):
        self.results, self.error = results or {}, error

    async def verify(self, plan, evidence):
        if self.error:
            raise self.error
        for ev in evidence:
            yield ev.id, self.results.get(ev.id, True)


class FakeGateway:
    def __init__(self, plan=None, error=None):
        self.plan, self.error = plan, error

    async def chat_json_ex(self, task, messages, schema):
        if self.error:
            raise self.error
        return self.plan, "groq"


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.camera("cam_01", "Gate", source="/data/gate.mp4")
    w.zone("z1", "cam_01")
    yield w
    w.close()


def make_router(ws, resolver=None, clarifier=None, verifier=None, gateway=None, accept=0.4, narrator=None,
                path_for=None, **retrieval):
    resolver = resolver or FakeResolver({"main gate": place_fact()})
    retriever = Retriever(ws.db, ws.store, ToyEmbedder(),
                          RetrievalConfig(calibration=Calibration(0.5, 0.2), **retrieval))
    return Router(ws.db, Planner(gateway), retriever, resolver, clarifier or FakeClarifier(resolver), verifier,
                  RouterConfig(accept=accept), gateway=narrator, path_for=path_for)


async def collect(stream):
    return [ev async for ev in stream]


def types(events):
    return [e.type for e in events]


def of(events, kind):
    return [e.data for e in events if e.type == kind]


def red_car_crossing(ws, tid="t_red", t=1105.0, cam="cam_01", eid="e1", **kw):
    ws.track(tid, cam, attrs=RED_CAR, crops=[E[0]], bbox=(0.3, 0.2, 0.45, 0.7), **kw)
    ws.event(eid, cam, tid, "cross_line", t, direction="a_to_b")


@pytest.mark.asyncio
async def test_happy_path_streams_in_contract_order_with_the_crossing_time(ws):
    red_car_crossing(ws)
    events = await collect(make_router(ws).answer(GATE_Q, "s1"))
    assert types(events) == ["plan", "evidence", "answer", "done"]
    ev = of(events, "evidence")[0]
    assert ev["t_peak"] == 1105.0 and ev["camera_name"] == "Gate" and ev["track_id"] == "t_red"
    assert ev["offset_s"] == 105.0 and ev["bbox"] == [0.3, 0.2, 0.45, 0.7]
    assert ev["thumb_url"].endswith(".jpg") and ev["clip_url"].endswith(".mp4")
    assert any(w.startswith("cross_line") for w in ev["why"])
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "yes" and ans["evidence"][0]["id"] == ev["id"]
    assert ans["text"].startswith("Yes. A red car passed through the main gate:")
    assert "01:45 into gate.mp4" in ans["text"] and "on Gate" in ans["text"]  # both timestamps
    assert ans["timings_ms"]["ttfa"] >= 0 and ans["confidence"] > 0.9
    assert of(events, "plan")[0]["source"] == "fastpath"
    assert of(events, "done")[0]["query_id"] == ans["query_id"]


@pytest.mark.asyncio
async def test_evidence_is_registered_and_the_query_is_logged(ws):
    red_car_crossing(ws)
    events = await collect(make_router(ws).answer(GATE_Q, "s1"))
    eid = of(events, "evidence")[0]["id"]
    with ws.db.read() as c:
        assert c.execute("SELECT camera_id, t_peak FROM evidence WHERE id=?", (eid,)).fetchone()["t_peak"] == 1105.0
        row = c.execute("SELECT text, answer FROM query_log").fetchone()
    assert row["text"] == GATE_Q and eid in row["answer"]


@pytest.mark.asyncio
async def test_unknown_referent_asks_once_and_stops(ws):
    red_car_crossing(ws)
    resolver = FakeResolver()
    clarifier = FakeClarifier(resolver)
    events = await collect(make_router(ws, resolver, clarifier).answer(GATE_Q, "s1"))
    assert types(events) == ["plan", "clarify"]  # no evidence, answer or done
    req = of(events, "clarify")[0]
    assert req["referent"]["text"] == "main gate" and req["kind"] == "choose_camera"
    assert [o["camera_id"] for o in req["options"]] == ["cam_01"]
    assert clarifier.asked == [("main gate", ["cam_01"])]


@pytest.mark.asyncio
async def test_resume_continues_with_the_original_question(ws):
    red_car_crossing(ws)
    resolver = FakeResolver()
    clarifier = FakeClarifier(resolver, bind_on_resume={"main gate": place_fact()})
    router = make_router(ws, resolver, clarifier)
    asked = await collect(router.answer(GATE_Q, "s1"))
    qid = of(asked, "clarify")[0]["query_id"]
    events = await collect(router.resume(ClarifyResponse(query_id=qid, camera_id="cam_01")))
    assert types(events) == ["plan", "evidence", "answer", "done"]
    assert of(events, "answer")[0]["query_id"] == qid and of(events, "answer")[0]["verdict"] == "yes"
    assert len(clarifier.asked) == 1
    again = await collect(router.answer(GATE_Q, "s1"))  # memory now knows the place: no second question
    assert "clarify" not in types(again)


@pytest.mark.asyncio
async def test_resume_of_an_unknown_query_is_an_error_not_a_crash(ws):
    events = await collect(make_router(ws).resume(ClarifyResponse(query_id="q_gone")))
    assert types(events) == ["error", "done"] and "no longer waiting" in of(events, "error")[0]["message"]


@pytest.mark.asyncio
async def test_no_match_gives_a_grounded_negative_with_the_nearest_miss(ws):
    ws.track("t_blue", "cam_01", attrs=BLUE_CAR, crops=[E[1]])
    ws.event("e1", "cam_01", "t_blue", "cross_line", 1105.0, direction="a_to_b")
    events = await collect(make_router(ws).answer(GATE_Q, "s1"))
    assert types(events) == ["plan", "answer", "done"]
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "no" and ans["evidence"] == [] and ans["nearest_miss"]["track_id"] == "t_blue"
    assert ans["text"].startswith("No red car passed through the main gate.") and "Closest:" in ans["text"]
    with ws.db.read() as c:  # the near miss can be shown, so its media must be renderable
        assert c.execute("SELECT 1 FROM evidence WHERE id=?", (ans["nearest_miss"]["id"],)).fetchone()


@pytest.mark.asyncio
async def test_a_good_track_that_never_crossed_is_not_a_match(ws):
    ws.track("t_red", "cam_01", attrs=RED_CAR, crops=[E[0]])  # looks right but no crossing event
    events = await collect(make_router(ws).answer(GATE_Q, "s1"))
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "no" and ans["nearest_miss"] is None


@pytest.mark.asyncio
async def test_count_uses_distinct_identities(ws):
    for i, gid in enumerate(["g1", "g1", "g2"]):
        red_car_crossing(ws, tid=f"t{i}", t=1100.0 + i * 20, eid=f"e{i}", gid=gid, t0=1090.0 + i * 20,
                         t1=1110.0 + i * 20)
    events = await collect(make_router(ws).answer("how many red cars passed through the main gate", "s1"))
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "count" and ans["count"] == 2  # three tracks, two identities
    assert ans["text"].startswith("Counted 2 matching cars passing through the main gate.")


@pytest.mark.asyncio
async def test_first_and_last_pick_by_time_not_by_score(ws):
    red_car_crossing(ws, tid="early", t=1102.0, eid="e1", t0=1100.0, t1=1110.0)
    red_car_crossing(ws, tid="late", t=1202.0, eid="e2", t0=1200.0, t1=1210.0)

    def plan(intent):
        return QueryPlan(intent=intent, targets=[Target(noun="car", cls=["car"], attributes=["red"],
                                                        embed_text="a photo of a red car")],
                         place=Referent(text="main gate", role="place"), action="pass_through",
                         unresolved=[Referent(text="main gate", role="place")], limit=1)

    for intent, want in (("first", 1102.0), ("last", 1202.0)):
        router = make_router(ws, gateway=FakeGateway(plan(intent)))
        events = await collect(router.answer(f"an unusual {intent} question", "s1"))
        ev = of(events, "evidence")
        assert len(ev) == 1 and ev[0]["t_peak"] == want, intent
        assert of(events, "answer")[0]["text"].startswith(f"The {intent} match")


@pytest.mark.asyncio
async def test_verifier_streams_after_the_answer(ws):
    red_car_crossing(ws)
    router = make_router(ws, verifier=FakeVerifier({}))
    events = await collect(router.answer(GATE_Q, "s1"))
    assert types(events) == ["plan", "evidence", "answer", "verified", "done"]
    ev_id = of(events, "evidence")[0]["id"]
    assert of(events, "verified") == [{"evidence_id": ev_id, "verified": True}]


@pytest.mark.asyncio
async def test_a_failing_verifier_never_loses_the_answer(ws):
    red_car_crossing(ws)
    events = await collect(make_router(ws, verifier=FakeVerifier(error=RuntimeError("vlm down"))).answer(GATE_Q, "s1"))
    assert types(events) == ["plan", "evidence", "answer", "note", "done"]
    assert "unavailable" in of(events, "note")[0]["text"]


@pytest.mark.asyncio
async def test_planning_failure_is_an_error_event(ws):
    router = make_router(ws, gateway=FakeGateway(error=LLMError("all down")))
    events = await collect(router.answer("something the fast path cannot parse", "s1"))
    assert types(events) == ["error", "done"] and "could not plan" in of(events, "error")[0]["message"]


@pytest.mark.asyncio
async def test_standing_questions_are_redirected(ws):
    plan = QueryPlan(intent="standing", targets=[Target(noun="person", embed_text="a photo of a person")])
    events = await collect(make_router(ws, gateway=FakeGateway(plan)).answer("alert me when someone arrives", "s1"))
    assert types(events) == ["plan", "note", "done"] and "Watch panel" in of(events, "note")[0]["text"]


@pytest.mark.asyncio
async def test_unindexed_camera_makes_the_answer_partial_and_says_so(ws):
    ws.camera("cam_02", "Lobby", layers=("L0",))
    red_car_crossing(ws)
    router = make_router(ws, resolver=FakeResolver({"main gate": place_fact()}))
    events = await collect(router.answer(GATE_Q, "s1"))
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "yes"  # the gate camera is fully indexed; the lobby is not part of this question
    ws.camera("cam_03", "Roof", layers=())
    red_car_crossing(ws, tid="t2", eid="e2", cam="cam_01", t=1106.0)
    partial = await collect(make_router(ws, resolver=FakeResolver({"main gate": place_fact()})).answer(GATE_Q, "s1"))
    assert of(partial, "answer")[0]["verdict"] == "yes"
    unfinished = await collect(make_router(ws).answer("did a red car pass through the Lobby", "s1"))
    ans = of(unfinished, "answer")[0]
    assert ans["verdict"] == "no" and any("Lobby is still being indexed" in n for n in ans["notes"])


@pytest.mark.asyncio
async def test_coarse_scene_matches_cannot_prove_a_crossing(ws):
    ws.camera("cam_02", "Lobby", layers=("L0",))
    for t in (1050.0, 1051.0):
        ws.scene("cam_02", t, E[0])
    router = make_router(ws, resolver=FakeResolver({"main gate": place_fact("cam_02", "z_none")}))
    events = await collect(router.answer(GATE_Q, "s1"))
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "no" and any("Coarse scene matches" in n for n in ans["notes"])
    plain = await collect(make_router(ws).answer("was there a red car at the Lobby", "s1"))
    found = of(plain, "answer")[0]
    # a scene tile can show presence, but the camera is still indexing, so the answer says it is partial
    assert found["verdict"] == "partial" and found["evidence"][0]["track_id"] is None


@pytest.mark.asyncio
async def test_remembered_object_filters_by_identity(ws):
    red_car_crossing(ws, tid="mine", eid="e1", gid="g_mine")
    red_car_crossing(ws, tid="other", eid="e2", gid="g_other", t=1107.0)
    fact = MemoryFact(id="mf_car", kind="object", canonical="my car", binding={"global_id": "g_mine"},
                      source="statement", created_at=0.0)
    resolver = FakeResolver({"main gate": place_fact(), "my car": fact})
    plan = QueryPlan(intent="list", targets=[Target(noun="car", cls=["car"], attributes=["red"],
                                                    embed_text="a photo of a red car")],
                     place=Referent(text="main gate", role="place"), action="pass_through",
                     unresolved=[Referent(text="main gate", role="place"), Referent(text="my car", role="object")])
    events = await collect(make_router(ws, resolver, gateway=FakeGateway(plan)).answer("show me my car at the gate",
                                                                                    "s1"))
    assert [e["track_id"] for e in of(events, "evidence")] == ["mine"]


@pytest.mark.asyncio
async def test_remembered_time_is_merged_into_the_window(ws):
    red_car_crossing(ws, tid="inside", eid="e1", t=1100.0 + 3600 * 5)  # 05:00 local
    ws.camera("cam_x", "X")
    fact = MemoryFact(id="mf_t", kind="time", canonical="after hours", binding={"tod_after": "20:00",
                      "tod_before": "06:00"}, source="statement", created_at=0.0)
    plan = QueryPlan(intent="exists", targets=[Target(noun="car", cls=["car"], attributes=["red"],
                                                      embed_text="a photo of a red car")],
                     place=Referent(text="main gate", role="place"), action="pass_through",
                     unresolved=[Referent(text="main gate", role="place"), Referent(text="after hours", role="time")])
    resolver = FakeResolver({"main gate": place_fact(), "after hours": fact})
    events = await collect(make_router(ws, resolver, gateway=FakeGateway(plan)).answer("odd question", "s1"))
    ans = of(events, "answer")[0]
    assert ans["plan"]["time"]["tod_after"] == "20:00" and ans["plan"]["time"]["tod_before"] == "06:00"
    assert ans["verdict"] == "yes"  # 05:18 UTC is inside the wrapped range


def test_router_config_defaults():
    assert RouterConfig().accept == 0.40


@pytest.mark.asyncio
async def test_an_unknown_time_word_is_asked_once_through_memory(ws):
    red_car_crossing(ws, tid="inside", eid="e1", t=1100.0 + 3600 * 5)
    plan = QueryPlan(intent="exists", targets=[Target(noun="car", cls=["car"], attributes=["red"],
                                                      embed_text="a photo of a red car")],
                     place=Referent(text="main gate", role="place"), action="pass_through",
                     time=TimeWindow(phrase="after hours"), unresolved=[Referent(text="main gate", role="place")])
    resolver = FakeResolver({"main gate": place_fact()})
    clarifier = FakeClarifier(resolver)
    events = await collect(make_router(ws, resolver, clarifier, gateway=FakeGateway(plan)).answer("odd", "s1"))
    assert types(events) == ["plan", "clarify"]
    assert of(events, "plan")[0]["unresolved"][-1] == {"text": "after hours", "role": "time"}
    assert clarifier.asked[0][0] == "after hours"

    fact = MemoryFact(id="mf_t", kind="time", canonical="after hours",
                      binding={"tod_after": "20:00", "tod_before": "06:00"}, source="statement", created_at=0.0)
    resolver.known["after hours"] = fact  # once remembered it is never asked again
    again = await collect(make_router(ws, resolver, clarifier, gateway=FakeGateway(plan)).answer("odd", "s1"))
    assert types(again)[-2:] == ["answer", "done"] and "clarify" not in types(again)
    assert of(again, "answer")[0]["plan"]["time"]["tod_after"] == "20:00"


# --------------------------------------------------------------------- path
def path_plan(**kw):
    base = dict(intent="path", targets=[Target(noun="car", cls=["car"], attributes=["red"],
                                               embed_text="a photo of a red car")], action="any", limit=10)
    base.update(kw)
    return QueryPlan(**base)


@pytest.mark.asyncio
async def test_path_follows_the_identity_across_cameras_in_time_order(ws):
    ws.camera("cam_02", "Lobby", source="/data/lobby.mp4")
    ws.track("cam_01:t1", "cam_01", attrs=RED_CAR, gid="g1", t0=1100.0, t1=1110.0, crops=[E[0]])
    ws.track("cam_02:t7", "cam_02", attrs=RED_CAR, gid="g1", t0=1180.0, t1=1190.0, crops=[E[3]])  # looks different
    ws.track("cam_02:t8", "cam_02", attrs=BLUE_CAR, gid="g2", t0=1100.0, t1=1110.0, crops=[E[1]])  # someone else
    router = make_router(ws, gateway=FakeGateway(path_plan()))
    events = await collect(router.answer("where did the red car go", "s1"))
    assert types(events) == ["plan", "evidence", "evidence", "answer", "done"]
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "found" and [e["camera_name"] for e in ans["evidence"]] == ["Gate", "Lobby"]
    assert [h["camera_name"] for h in ans["path"]] == ["Gate", "Lobby"]
    assert [h["evidence_id"] for h in ans["path"]] == [e["id"] for e in ans["evidence"]]  # hops cite our evidence ids
    assert ans["text"].startswith("Path of red car: Gate ") and " → Lobby " in ans["text"]
    assert any("same identity" in w for w in ans["evidence"][1]["why"])
    with ws.db.read() as c:  # every hop is renderable
        assert c.execute("SELECT COUNT(*) AS n FROM evidence").fetchone()["n"] == 2


@pytest.mark.asyncio
async def test_a_track_with_no_identity_gives_a_single_stop_and_says_so(ws):
    ws.track("cam_01:t1", "cam_01", attrs=RED_CAR, crops=[E[0]])
    events = await collect(make_router(ws, gateway=FakeGateway(path_plan())).answer("where did the red car go", "s1"))
    ans = of(events, "answer")[0]
    assert len(ans["path"]) == 1 and any("single stop" in n for n in ans["notes"])


@pytest.mark.asyncio
async def test_path_with_no_matching_car_is_an_honest_negative(ws):
    ws.track("cam_01:t1", "cam_01", attrs=BLUE_CAR, crops=[E[1]])
    events = await collect(make_router(ws, gateway=FakeGateway(path_plan())).answer("where did the red car go", "s1"))
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "not_found" and ans["path"] == []


@pytest.mark.asyncio
async def test_a_path_provider_can_be_injected(ws):
    ws.track("cam_01:t1", "cam_01", attrs=RED_CAR, gid="g1", crops=[E[0]])
    from contracts.models import PathHop

    calls = []

    def provider(gid):
        calls.append(gid)
        return [PathHop(camera_id="cam_01", camera_name="Gate", t_in=1100.0, t_out=1110.0, evidence_id="cam_01_t1")]

    router = make_router(ws, gateway=FakeGateway(path_plan()), path_for=provider)
    events = await collect(router.answer("where did the red car go", "s1"))
    assert calls == ["g1"] and len(of(events, "answer")[0]["path"]) == 1


# ----------------------------------------------------------------- describe
def describe_plan(**kw):
    base = dict(intent="describe", targets=[], camera_ids=["cam_01"], time=TimeWindow(phrase="today"))
    base.update(kw)
    return QueryPlan(**base)


class Narrator:
    def __init__(self, lines=None, error=None):
        self.lines, self.error = lines or [], error

    async def chat_json(self, task, messages, schema):
        if self.error:
            raise self.error
        return schema.model_validate({"sentences": self.lines})


def two_events(ws):
    red_car_crossing(ws, tid="t1", eid="e1", t=1105.0)
    ws.track("t2", "cam_01", cls="person", attrs={"upper_color": "red"}, t0=1150.0, t1=1160.0, crops=[E[2]])
    ws.event("e2", "cam_01", "t2", "enter_zone", 1155.0)


@pytest.mark.asyncio
async def test_describe_summarises_the_events_with_cited_evidence(ws):
    two_events(ws)
    router = make_router(ws, gateway=FakeGateway(describe_plan()))
    events = await collect(router.answer("what happened at the gate today", "s1"))
    assert types(events)[0] == "plan" and types(events)[-2:] == ["answer", "done"]
    assert types(events).count("evidence") == 2
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "found" and "were tracked at the Gate camera today" in ans["text"]
    assert "A red car crossed the Gate camera at" in ans["text"]
    assert {e["track_id"] for e in ans["evidence"]} == {"t1", "t2"}
    assert not any("language model" in n for n in ans["notes"])  # no model configured: plain summary


@pytest.mark.asyncio
async def test_describe_with_a_model_adds_the_times_itself(ws):
    two_events(ws)
    lines = [{"text": "A red car crossed the Gate camera and a person came in", "cites": ["E1", "E2"]}]
    router = make_router(ws, gateway=FakeGateway(describe_plan()), narrator=Narrator(lines))
    ans = of(await collect(router.answer("what happened at the gate today", "s1")), "answer")[0]
    assert ans["text"].startswith("A red car crossed the Gate camera and a person came in. At ")
    assert "into gate.mp4" not in ans["text"] or "on Gate" in ans["text"]
    assert any("language model" in n for n in ans["notes"])
    assert len(ans["evidence"]) == 2


@pytest.mark.asyncio
async def test_describe_ignores_wording_that_breaks_the_rules(ws):
    two_events(ws)
    bad = [{"text": "A red car crossed at 09:14", "cites": ["E1"]}]  # a clock time written by the model
    router = make_router(ws, gateway=FakeGateway(describe_plan()), narrator=Narrator(bad))
    ans = of(await collect(router.answer("what happened at the gate today", "s1")), "answer")[0]
    assert "were tracked" in ans["text"] and not any("language model" in n for n in ans["notes"])
    router = make_router(ws, gateway=FakeGateway(describe_plan()), narrator=Narrator(error=LLMError("down")))
    again = of(await collect(router.answer("what happened at the gate today", "s1")), "answer")[0]
    assert "were tracked" in again["text"]


@pytest.mark.asyncio
async def test_describe_with_nothing_to_report_is_a_grounded_negative(ws):
    events = await collect(make_router(ws, gateway=FakeGateway(describe_plan())).answer("what happened at the gate today", "s1"))
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "not_found" and ans["evidence"] == []
    assert ans["text"] == "No activity was recorded at the Gate camera today."
    assert "evidence" not in types(events)


@pytest.mark.asyncio
async def test_describe_uses_the_remembered_place_name_for_events_in_a_zone(ws):
    two_events(ws)
    with ws.db.write() as c:
        c.execute("INSERT INTO memory_facts(id,kind,canonical,aliases,binding,source,created_at) "
                  "VALUES('mf9','place','main gate','[]','{}','clarification',0)")
        c.execute("UPDATE zones SET fact_id='mf9' WHERE id='z1'")
    ans = of(await collect(make_router(ws, gateway=FakeGateway(describe_plan())).answer("what happened at the gate today", "s1")),
             "answer")[0]
    assert "A red car crossed the main gate at" in ans["text"]
