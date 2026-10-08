import json

import pytest
from contracts.models import QueryPlan, Target, TimeWindow

from evora.query.actions import detected_action, unsupported_action
from evora.query.logic import ACTION_EVENT_KINDS
from evora.query.router import RouterConfig
from tests.query.test_router import E, FakeVerifier, StubPlanner, collect, make_router, of, types
from tests.query.ws_helpers import Workspace


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.camera("cam_01", "Gate", source="/data/gate.mp4")
    yield w
    w.close()


# ---- phrases -> the actions perception detects --------------------------------------------------------------------------
@pytest.mark.parametrize("text,classes,key", [
    ("Did a vehicle reverse on 9 March between 10:10 and 10:15?", ["car"], "reverse"),
    ("did the car back up", ["car"], "reverse"),
    ("Did a person get out of a vehicle?", ["person"], "vehicle_out"),
    ("did someone exit the car", ["person"], "vehicle_out"),
    ("Did a person get into a vehicle?", ["person"], "vehicle_in"),
    ("did anyone board the bus", [], "vehicle_in"),
    ("Did a vehicle make a U-turn?", [], "u_turn"),
    ("did a car turn around", ["car"], "u_turn"),
    ("Did a vehicle turn left?", ["car"], "turn_left"),
    ("did a truck turn right", ["truck"], "turn_right"),
    ("Did a vehicle stop?", [], "stop"),
    ("did a car stop at the gate", ["car"], "stop"),
    ("Did a vehicle start moving?", [], "start"),
    ("did a car drive off", ["car"], "start"),
    ("Were two people talking to each other?", ["person"], "talk"),
    ("did anyone chat near the entrance", ["person"], "talk"),
])
def test_phrases_map_to_the_action_perception_detects(text, classes, key):
    assert detected_action(text, classes).key == key


@pytest.mark.parametrize("text,classes", [
    ("Did a person turn left?", ["person"]),                       # only vehicles are followed through a turn
    ("did someone stop at the gate", ["person"]),                  # a person stopping is dwell, answered by the tracker
    ("Is anyone talking on a phone?", ["person"]),                 # a phone call is not two people standing together
    ("was a person texting", ["person"]),
    ("Did a person pick something up?", ["person"]),               # nobody detects this
    ("Did a person open a door?", ["person"]),
    ("Did a person stand up?", ["person"]),
    ("did a red car pass through the main gate", ["car"]),
    ("how many people are in the room", ["person"]),
])
def test_what_nothing_detects_is_not_mapped(text, classes):
    assert detected_action(text, classes) is None


def test_every_action_perception_writes_has_a_phrase_that_reaches_it():
    from evora.perception.actions import ACTION_KINDS
    mapped = {kind for kinds in ACTION_EVENT_KINDS.values() for kind in kinds}
    assert set(ACTION_KINDS) <= mapped, f"perception writes {set(ACTION_KINDS) - mapped} but no question can reach it"
    assert {d for d in mapped} == set(ACTION_KINDS)         # and nothing is mapped that perception never writes


def test_a_phone_call_is_a_phone_question_not_a_talking_one():
    assert unsupported_action("is anyone talking on a phone") == "using a phone"
    assert unsupported_action("were two people talking") == "talking"


# ---- through the router ------------------------------------------------------------------------------------------------
def vehicle(ws, tid="v1", cam="cam_01", t0=1100.0, t1=1110.0):
    ws.track(tid, cam, cls="car", crops=[E[0]], t0=t0, t1=t1, bbox=(0.3, 0.3, 0.5, 0.6))


def person(ws, tid="p1", cam="cam_01", t0=1100.0, t1=1110.0):
    ws.track(tid, cam, cls="person", crops=[E[2]], t0=t0, t1=t1, bbox=(0.6, 0.3, 0.65, 0.7))


def add_event(ws, eid, tid, kind, t, cam="cam_01", **payload):
    ws.event(eid, cam, tid, kind, t, zone=None, **payload)


def plan_for(intent="exists", noun="car", cls=("car",), time=None):
    return QueryPlan(intent=intent, targets=[Target(noun=noun, cls=list(cls), embed_text=f"a photo of a {noun}")],
                     action="any", time=time)


def asking(ws, plan, **kw):
    router = make_router(ws, **kw)
    router._planner = StubPlanner(plan)
    return router


@pytest.mark.asyncio
async def test_a_yes_comes_from_a_matching_event_with_the_moment_and_what_was_measured(ws):
    vehicle(ws)
    add_event(ws, "v1:vehicle_turn_left:-:0", "v1", "vehicle_turn_left", 1105.0, turn_deg=82.0)
    events = await collect(asking(ws, plan_for()).answer("did a car turn left", "s1"))
    assert types(events) == ["plan", "evidence", "answer", "done"]
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "yes" and ans["text"].startswith("Yes. A vehicle turning left was detected:")
    ev = ans["evidence"][0]
    assert ev["t_peak"] == 1105.0 and ev["track_id"] == "v1" and ev["bbox"] is not None
    assert "detected: a vehicle turning left" in ev["why"] and "path turned 82 degrees" in ev["why"]
    assert any("not by watching the footage" in n and "none was detected, not that none happened" in n for n in ans["notes"])


@pytest.mark.asyncio
async def test_with_no_matching_event_the_answer_is_no_and_shows_nothing(ws):
    vehicle(ws)                                                        # a car is there, but it never reversed
    add_event(ws, "v1:vehicle_turn_left:-:0", "v1", "vehicle_turn_left", 1105.0, turn_deg=82.0)
    events = await collect(asking(ws, plan_for()).answer("did a car reverse", "s1"))
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "no" and ans["evidence"] == [] and "evidence" not in types(events)
    assert ans["text"].startswith("No, a vehicle reversing was not detected")


@pytest.mark.asyncio
async def test_only_events_inside_the_time_window_count(ws):
    vehicle(ws)
    add_event(ws, "v1:vehicle_stop:-:0", "v1", "vehicle_stop", 1105.0)
    add_event(ws, "v1:vehicle_stop:-:1", "v1", "vehicle_stop", 1500.0)
    inside = plan_for("count", time=TimeWindow(start=1000.0, end=1200.0))
    ans = of(await collect(asking(ws, inside).answer("how many times did a car stop", "s1")), "answer")[0]
    assert ans["verdict"] == "count" and ans["count"] == 1 and "was detected 1 time" in ans["text"]
    after = plan_for("exists", time=TimeWindow(start=1400.0, end=1600.0))
    ans = of(await collect(asking(ws, after).answer("did a car stop", "s2")), "answer")[0]
    assert ans["verdict"] == "yes" and ans["evidence"][0]["t_peak"] == 1500.0
    nothing = plan_for("exists", time=TimeWindow(start=2000.0, end=2100.0))
    assert of(await collect(asking(ws, nothing).answer("did a car stop", "s3")), "answer")[0]["verdict"] == "no"


@pytest.mark.asyncio
async def test_first_last_and_list_use_the_events_in_time_order(ws):
    vehicle(ws)
    for n, t in enumerate((1103.0, 1106.0, 1109.0)):
        add_event(ws, f"v1:vehicle_start:-:{n}", "v1", "vehicle_start", t)
    first = of(await collect(asking(ws, plan_for("first")).answer("when did a car first start moving", "s1")), "answer")[0]
    assert first["verdict"] == "found" and [e["t_peak"] for e in first["evidence"]] == [1103.0]
    last = of(await collect(asking(ws, plan_for("last")).answer("when did a car last start moving", "s2")), "answer")[0]
    assert [e["t_peak"] for e in last["evidence"]] == [1109.0]
    listed = of(await collect(asking(ws, plan_for("list")).answer("show me a car starting to move", "s3")), "answer")[0]
    assert [e["t_peak"] for e in listed["evidence"]] == [1103.0, 1106.0, 1109.0] and "3 times" in listed["text"]


@pytest.mark.asyncio
async def test_a_person_event_needs_a_person_and_a_vehicle_event_needs_a_vehicle(ws):
    vehicle(ws)
    person(ws)
    add_event(ws, "p1:person_exits_vehicle:-:0", "p1", "person_exits_vehicle", 1104.0, with_track="v1")
    add_event(ws, "v1:vehicle_stop:-:0", "v1", "vehicle_stop", 1103.0)
    out = of(await collect(asking(ws, plan_for(noun="person", cls=("person",))).answer(
        "did a person get out of a vehicle", "s1")), "answer")[0]
    assert out["verdict"] == "yes" and out["evidence"][0]["track_id"] == "p1" and "next to v1" in out["evidence"][0]["why"]
    stop = of(await collect(asking(ws, plan_for()).answer("did a car stop", "s2")), "answer")[0]
    assert stop["evidence"][0]["track_id"] == "v1"


@pytest.mark.asyncio
async def test_a_named_place_is_not_used_and_the_answer_says_so(ws):
    from contracts.models import Referent
    vehicle(ws)
    add_event(ws, "v1:vehicle_stop:-:0", "v1", "vehicle_stop", 1103.0)
    plan = plan_for().model_copy(update={"place": Referent(text="loading dock", role="place"), "unresolved": []})
    ans = of(await collect(asking(ws, plan).answer("did a car stop at the loading dock", "s1")), "answer")[0]
    assert ans["verdict"] == "yes" and any('"loading dock" is not used' in n for n in ans["notes"])


# ---- an action nothing detects ------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_an_action_nobody_detects_is_partial_with_no_evidence_whoever_was_there(ws):
    person(ws, "p1")
    person(ws, "p2")
    for question in ("did a person pick something up", "did a person put something down", "did someone open the door",
                     "did a person drop a bag"):
        events = await collect(asking(ws, plan_for(noun="person", cls=("person",))).answer(question, "s"))
        ans = of(events, "answer")[0]
        assert ans["verdict"] == "partial" and ans["evidence"] == [] and "evidence" not in types(events), question
        assert ans["text"].startswith("I can't verify whether anyone was") and ans["unsupported_action"], question
        assert any("What evora does recognise: vehicles turning" in n for n in ans["notes"])


@pytest.mark.asyncio
async def test_a_person_turning_left_is_not_a_vehicle_turning_left(ws):
    person(ws)
    vehicle(ws)
    add_event(ws, "v1:vehicle_turn_left:-:0", "v1", "vehicle_turn_left", 1105.0, turn_deg=80.0)
    ans = of(await collect(asking(ws, plan_for(noun="person", cls=("person",))).answer(
        "did a person turn left", "s1")), "answer")[0]
    assert ans["verdict"] == "partial" and ans["evidence"] == []         # the car's turn is not the person's


@pytest.mark.asyncio
async def test_the_estimates_can_still_be_switched_on(ws):
    person(ws)
    router = asking(ws, plan_for(noun="person", cls=("person",)))
    router.cfg = RouterConfig(accept=0.4, show_unverified_actions=True)
    ans = of(await collect(router.answer("did a person pick something up", "s1")), "answer")[0]
    assert ans["unsupported_action"] == "picking something up" and ans["text"].startswith("I can't tell whether")


@pytest.mark.asyncio
async def test_the_detected_route_can_be_switched_off(ws):
    vehicle(ws)
    add_event(ws, "v1:vehicle_stop:-:0", "v1", "vehicle_stop", 1103.0)
    router = asking(ws, plan_for())
    router.cfg = RouterConfig(accept=0.4, detected_actions=False)
    ans = of(await collect(router.answer("did a car stop", "s1")), "answer")[0]
    assert "was detected" not in ans["text"]


# ---- the visual check on the frames at those moments -------------------------------------------------------------------
class RecordingVerifier(FakeVerifier):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = []

    async def verify(self, plan, evidence):
        self.seen.append((plan.targets[0].embed_text, [(e.track_id, e.bbox) for e in evidence]))
        async for item in super().verify(plan, evidence):
            yield item


@pytest.mark.asyncio
async def test_people_standing_together_are_checked_on_the_whole_frame_and_the_check_is_reported(ws):
    person(ws, "p1")
    add_event(ws, "p1:people_close:-:0", "p1", "people_close", 1105.0, with_track="p2", seconds=6.0)
    verifier = RecordingVerifier(default=True)
    events = await collect(asking(ws, plan_for(noun="person", cls=("person",)), verifier=verifier).answer(
        "were two people talking", "s1"))
    first, final = of(events, "answer")
    assert first["verdict"] == "yes" and "together for 6 s" in first["evidence"][0]["why"]
    assert verifier.seen == [("a photo of two people standing close together", [(None, None)])]   # the scene, not one track
    assert final["verdict"] == "yes" and final["evidence"][0]["verified"] is True
    assert any("confirmed 1 of 1 (two people close together)" in n for n in final["notes"])


@pytest.mark.asyncio
async def test_a_check_that_confirms_nothing_makes_the_yes_partial_but_keeps_the_events(ws):
    person(ws, "p1")
    add_event(ws, "p1:people_close:-:0", "p1", "people_close", 1105.0, with_track="p2", seconds=6.0)
    events = await collect(asking(ws, plan_for(noun="person", cls=("person",)), verifier=RecordingVerifier(default=False)).answer(
        "were two people talking", "s1"))
    final = of(events, "answer")[-1]
    assert final["verdict"] == "partial" and len(final["evidence"]) == 1 and final["evidence"][0]["verified"] is False
    assert any("treat it as unconfirmed" in n for n in final["notes"])


@pytest.mark.asyncio
async def test_motion_actions_are_not_sent_to_a_single_frame_check(ws):
    vehicle(ws)
    add_event(ws, "v1:vehicle_turn_left:-:0", "v1", "vehicle_turn_left", 1105.0, turn_deg=82.0)
    verifier = RecordingVerifier(default=False)
    events = await collect(asking(ws, plan_for(), verifier=verifier).answer("did a car turn left", "s1"))
    assert verifier.seen == [] and types(events).count("answer") == 1       # one frame cannot show a turn


@pytest.mark.asyncio
async def test_the_events_table_payloads_are_read_as_json(ws):
    vehicle(ws)
    with ws.db.write() as c:
        c.execute("INSERT INTO events(id,camera_id,track_id,kind,zone_id,t,payload) VALUES(?,?,?,?,?,?,?)",
                  ("v1:vehicle_u_turn:-:0", "cam_01", "v1", "vehicle_u_turn", None, 1105.0, json.dumps({"turn_deg": -171.5})))
    ans = of(await collect(asking(ws, plan_for()).answer("did a car make a u-turn", "s1")), "answer")[0]
    assert "path turned 172 degrees" in ans["evidence"][0]["why"]


@pytest.mark.asyncio
async def test_the_verb_agrees_with_the_subject(ws):
    person(ws, "p1")
    vehicle(ws)
    add_event(ws, "p1:people_close:-:0", "p1", "people_close", 1105.0, with_track="p2", seconds=6.0)
    add_event(ws, "p1:people_close:-:1", "p1", "people_close", 1107.0, with_track="p2", seconds=6.0)
    add_event(ws, "v1:vehicle_stop:-:0", "v1", "vehicle_stop", 1103.0)
    people = of(await collect(asking(ws, plan_for(noun="person", cls=("person",))).answer("were two people talking", "s1")),
                "answer")[0]
    assert people["text"].startswith("Yes. Two people standing together were detected")
    count = of(await collect(asking(ws, plan_for("count", noun="person", cls=("person",))).answer(
        "how many times were two people talking", "s2")), "answer")[0]
    assert count["text"].startswith("Two people standing together were detected 2 times")
    car = of(await collect(asking(ws, plan_for()).answer("did a car stop", "s3")), "answer")[0]
    assert car["text"].startswith("Yes. A vehicle stopping was detected")
