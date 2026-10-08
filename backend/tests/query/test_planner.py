from datetime import datetime
from types import SimpleNamespace

import pytest
from contracts.models import QueryPlan, Referent, Target, TimeWindow

from evora.core.db import close_all, open_db
from evora.llm.schemas import LLMError
from evora.query import planner as planner_mod
from evora.query.planner import (
    MemoryPlanCache,
    Planner,
    PlanningError,
    SqlitePlanCache,
    cache_key,
    reference_now,
    sanitize,
    workspace_tz,
)
from evora.query.timeparse import parse_tz

CAMS = [SimpleNamespace(id="cam_01", name="Gate"), SimpleNamespace(id="cam_02", name="Lobby")]
IST = parse_tz("+05:30")
REF = datetime(2026, 10, 9, 10, 30, tzinfo=IST).timestamp()


class FakeGateway:
    def __init__(self, plan=None, backend="groq", error=None):
        self.plan, self.backend, self.error, self.calls = plan, backend, error, []

    async def chat_json_ex(self, task, messages, schema):
        self.calls.append((task, messages, schema))
        if self.error:
            raise self.error
        return self.plan.model_copy(deep=True), self.backend


def model_plan(**kw):
    base = dict(
        intent="exists",
        targets=[Target(noun="person", cls=["person"], attributes=["red"], embed_text="a photo of a person in red")],
        action="enter",
        time=TimeWindow(phrase="yesterday evening"),
        source="llm",
    )
    base.update(kw)
    return QueryPlan(**base)


# an awkward question the fast path refuses
HARD = "where did the woman in the green jacket go after the lobby?"


@pytest.mark.asyncio
async def test_fast_path_needs_no_model_and_anchors_time():
    gw = FakeGateway()
    res = await Planner(gw).plan("did a red car pass through the gate in the last hour", CAMS, REF, IST)
    assert gw.calls == []
    assert res.plan.source == "fastpath" and res.plan.camera_ids == ["cam_01"]
    assert (res.plan.time.start, res.plan.time.end) == (REF - 3600, REF)
    assert "plan_fastpath" in res.timings_ms and res.notes == []


@pytest.mark.asyncio
async def test_model_plan_is_sanitized_cached_and_reanchored():
    plan = model_plan(
        camera_ids=["cam_99", "cam_02"],
        limit=500,
        targets=[Target(noun="woman", cls=["person", "spaceship"], embed_text="a woman")],
        place=Referent(text="gate", role="place"),  # really a camera name
    )
    gw = FakeGateway(plan)
    planner = Planner(gw)
    first = await planner.plan(HARD, CAMS, REF, IST)
    assert first.plan.source == "llm"
    assert first.plan.camera_ids == ["cam_02", "cam_01"]  # invented id dropped, "gate" became a filter
    assert first.plan.place is None and first.plan.unresolved == []
    assert first.plan.targets[0].cls == ["person"] and first.plan.limit == 50
    assert gw.calls[0][0] == "planner" and gw.calls[0][2] is QueryPlan
    assert HARD in gw.calls[0][1][1]["content"]

    later = REF + 86400  # a day later: same words, different dates
    second = await planner.plan("Where did the woman in the green jacket go after the lobby", CAMS, later, IST)
    assert len(gw.calls) == 1  # punctuation and case do not matter
    assert second.plan.source == "cache" and "plan_cache" in second.timings_ms
    assert second.plan.time.end != first.plan.time.end
    assert second.plan.time.end - second.plan.time.start == first.plan.time.end - first.plan.time.start


@pytest.mark.asyncio
async def test_cache_is_per_camera_set():
    gw = FakeGateway(model_plan())
    planner = Planner(gw)
    await planner.plan(HARD, CAMS, REF, IST)
    await planner.plan(HARD, CAMS[:1], REF, IST)
    assert len(gw.calls) == 2
    assert cache_key(HARD, CAMS) != cache_key(HARD, CAMS[:1])
    assert cache_key(HARD, CAMS) == cache_key(HARD.upper() + "??", list(reversed(CAMS)))


@pytest.mark.asyncio
async def test_local_backend_marks_source_and_adds_a_note():
    res = await Planner(FakeGateway(model_plan(), backend="local")).plan(HARD, CAMS, REF, IST)
    assert res.plan.source == "local_llm"
    assert any("local model" in n for n in res.notes)


@pytest.mark.asyncio
async def test_place_is_always_listed_for_the_memory_check():
    plan = model_plan(place=Referent(text="server room", role="place"), unresolved=[])
    res = await Planner(FakeGateway(plan)).plan(HARD, CAMS, REF, IST)
    assert [r.text for r in res.plan.unresolved] == ["server room"]


@pytest.mark.asyncio
async def test_failure_modes_raise_planning_error():
    with pytest.raises(PlanningError):
        await Planner(FakeGateway(error=LLMError("all down"))).plan(HARD, CAMS, REF, IST)
    with pytest.raises(PlanningError):
        await Planner(None).plan(HARD, CAMS, REF, IST)


@pytest.mark.asyncio
async def test_failed_planning_is_not_cached():
    gw = FakeGateway(error=LLMError("down"))
    planner = Planner(gw)
    with pytest.raises(PlanningError):
        await planner.plan(HARD, CAMS, REF, IST)
    gw.error, gw.plan = None, model_plan()
    assert (await planner.plan(HARD, CAMS, REF, IST)).plan.source == "llm"


@pytest.mark.asyncio
async def test_an_unknown_time_word_becomes_a_time_referent_for_memory():
    plan = model_plan(time=TimeWindow(phrase="after hours"))
    res = await Planner(FakeGateway(plan)).plan(HARD, CAMS, REF, IST)
    assert res.plan.time.start is None and res.plan.time.end is None
    assert Referent(text="after hours", role="time") in res.plan.unresolved
    # a model that already listed it is not duplicated
    listed = model_plan(time=TimeWindow(phrase="After  hours"), unresolved=[Referent(text="after hours", role="time")])
    again = await Planner(FakeGateway(listed)).plan(HARD, CAMS, REF, IST)
    assert [r.text for r in again.plan.unresolved if r.role == "time"] == ["after hours"]


@pytest.mark.asyncio
async def test_time_of_day_and_ordinary_phrases_never_go_to_memory():
    for window in (TimeWindow(phrase="after 8pm", tod_after="20:00"), TimeWindow(phrase="last week"),
                   TimeWindow(phrase="yesterday evening")):
        res = await Planner(FakeGateway(model_plan(time=window))).plan(HARD, CAMS, REF, IST)
        assert not [r for r in res.plan.unresolved if r.role == "time"], window.phrase


@pytest.mark.asyncio
async def test_standing_plans_keep_places_actions_and_time_of_day_but_no_fixed_dates():
    plan = QueryPlan(
        intent="standing",
        targets=[Target(noun="person", cls=["person"], embed_text="a photo of a person")],
        place=Referent(text="server room", role="place"), action="enter",
        time=TimeWindow(phrase="after 8pm", tod_after="20:00"),
        unresolved=[Referent(text="server room", role="place")])
    res = await Planner(FakeGateway(plan)).plan("alert me if anyone enters the server room after 8pm", CAMS, REF, IST)
    p = res.plan
    assert (p.intent, p.action, p.place.text, p.time.tod_after) == ("standing", "enter", "server room", "20:00")
    assert p.time.start is None and p.time.end is None and p.source == "llm"
    dated = plan.model_copy(update={"time": TimeWindow(phrase="today")})
    again = await Planner(FakeGateway(dated)).plan("alert me today", CAMS, REF, IST)
    assert again.plan.time.start is None and again.plan.time.phrase == "today"  # the rule must not freeze today's date


def test_sanitize_keeps_valid_plans_unchanged():
    plan = model_plan(camera_ids=["cam_01"], place=Referent(text="loading dock", role="place"),
                      unresolved=[Referent(text="loading dock", role="place")])
    assert sanitize(plan, CAMS) == plan


def test_sanitize_drops_camera_names_from_unresolved():
    plan = model_plan(unresolved=[Referent(text="the lobby", role="place"), Referent(text="my car", role="object")])
    assert [r.text for r in sanitize(plan, CAMS).unresolved] == ["my car"]


def test_memory_cache_roundtrip():
    cache = MemoryPlanCache()
    assert cache.get("k") is None
    cache.put("k", model_plan())
    assert cache.get("k") == model_plan()


@pytest.fixture
def db(tmp_path):
    database = open_db(tmp_path / "evora.db")
    yield database
    close_all()


def test_sqlite_cache_roundtrip_upsert_and_bad_rows(db):
    cache = SqlitePlanCache(db)
    assert cache.get("k") is None
    cache.put("k", model_plan())
    cache.put("k", model_plan(intent="list"))
    assert cache.get("k").intent == "list"
    with db.write() as conn:
        conn.execute("UPDATE plan_cache SET plan=? WHERE norm_text='k'", ('{"intent": "nonsense"}',))
    assert cache.get("k") is None


def add_camera(db, cam_id, t0, duration):
    with db.write() as conn:
        conn.execute(
            "INSERT INTO cameras(id, name, kind, source_uri, t0, t0_source, duration_s, created_at) "
            "VALUES(?, ?, 'file', 'x.mp4', ?, 'manual', ?, 0)",
            (cam_id, cam_id, t0, duration),
        )


def test_reference_now_is_the_end_of_the_latest_footage(db):
    assert reference_now(db, fallback=123.0) == 123.0  # nothing ingested yet
    add_camera(db, "cam_01", 1000.0, 300.0)
    add_camera(db, "cam_02", 1100.0, 600.0)
    assert reference_now(db) == 1700.0
    db.set_meta("reference_now", "5000")
    assert reference_now(db) == 5000.0
    db.set_meta("reference_now", "garbage")
    assert reference_now(db) == 1700.0


def test_workspace_tz_from_meta(db):
    assert workspace_tz(db).utcoffset(None).total_seconds() == 0
    db.set_meta("tz", "+05:30")
    assert workspace_tz(db).utcoffset(None).total_seconds() == 19800


def test_sanitize_repairs_what_small_models_get_wrong():
    plan = model_plan(targets=[
        Target(noun="woman", cls=[], attributes=[], embed_text="a photo of a woman in a green jacket"),
        Target(noun="green", cls=["person"], embed_text="green"),     # a colour posing as an object
        Target(noun="woman", cls=["person"], embed_text="duplicate"),  # same noun twice
        Target(noun="car", cls=["spaceship"], embed_text="a photo of a car"),
    ])
    fixed = sanitize(plan, CAMS).targets
    assert [t.noun for t in fixed] == ["woman", "car"]
    assert fixed[0].cls == ["person"] and fixed[0].attributes == ["green"]
    assert fixed[1].cls == ["car"] and fixed[1].attributes == []  # an invalid class is replaced from the noun


def test_sanitize_drops_cameras_the_question_never_named():
    plan = model_plan(camera_ids=["cam_01", "cam_02"])
    assert sanitize(plan, CAMS, "alert me if anyone enters the server room").camera_ids == []
    assert sanitize(plan, CAMS, "what happened at the gate and in the Lobby camera?").camera_ids == ["cam_01", "cam_02"]
    assert sanitize(plan, CAMS, "look at cam_02 please").camera_ids == ["cam_02"]
    assert sanitize(plan, CAMS).camera_ids == ["cam_01", "cam_02"]  # no question given: only unknown ids are dropped


@pytest.mark.asyncio
async def test_an_invented_camera_in_a_model_plan_does_not_survive_planning():
    plan = model_plan(camera_ids=["cam_01"], place=Referent(text="server room", role="place"),
                      unresolved=[Referent(text="server room", role="place")])
    res = await Planner(FakeGateway(plan, backend="local")).plan("alert me if anyone enters the server room", CAMS,
                                                                 REF, IST)
    assert res.plan.camera_ids == [] and [r.text for r in res.plan.unresolved] == ["server room"]


def test_sanitize_recovers_a_missing_place_from_unresolved():
    plan = model_plan(place=None, unresolved=[Referent(text="the lot", role="place"),
                                              Referent(text="my car", role="object")])
    fixed = sanitize(plan, CAMS)
    assert fixed.place == Referent(text="the lot", role="place")
    two = model_plan(place=None, unresolved=[Referent(text="the lot", role="place"),
                                             Referent(text="the dock", role="place")])
    assert sanitize(two, CAMS).place is None  # ambiguous: do not guess


@pytest.mark.asyncio
async def test_ordinary_times_listed_as_unresolved_by_a_model_are_dropped_but_idiosyncratic_ones_stay():
    plan = model_plan(
        time=TimeWindow(phrase="after 8pm", tod_after="20:00"),
        unresolved=[Referent(text="after 8pm", role="time"), Referent(text="last week", role="time"),
                    Referent(text="in the last hour", role="time"), Referent(text="night shift", role="time"),
                    Referent(text="server room", role="place")])
    res = await Planner(FakeGateway(plan)).plan(HARD, CAMS, REF, IST)
    assert [(r.text, r.role) for r in res.plan.unresolved] == [("night shift", "time"), ("server room", "place")]


def test_only_remembered_style_objects_stay_unresolved():
    plan = model_plan(unresolved=[
        Referent(text="my car", role="object"), Referent(text="that van", role="object"),
        Referent(text="Our delivery", role="object"), Referent(text="something heavy", role="object"),
        Referent(text="vehicle door", role="object"), Referent(text="the vehicle", role="object"),
        Referent(text="the loading dock", role="place")])
    kept = [(r.text, r.role) for r in sanitize(plan, CAMS).unresolved]
    assert kept == [("my car", "object"), ("that van", "object"), ("Our delivery", "object"),
                    ("the loading dock", "place")]


def test_the_default_calibration_matches_real_siglip_scores():
    from evora.query.fuse import Calibration

    cal = Calibration()
    assert cal(0.105) > 0.65   # a correct-class crop against "a photo of a <class>"
    assert cal(0.045) < 0.1    # a wrong-class crop
    assert cal(0.045) < cal(0.07) < cal(0.105) < cal(0.15)


def test_only_possessive_or_pointed_at_objects_are_kept_as_referents_to_remember():
    plan = model_plan(unresolved=[
        Referent(text="my car", role="object"), Referent(text="that van", role="object"),
        Referent(text="Our delivery", role="object"), Referent(text="something heavy", role="object"),
        Referent(text="vehicle door", role="object"), Referent(text="a vehicle", role="object"),
        Referent(text="the loading dock", role="place"),
    ])
    kept = [r.text for r in sanitize(plan, CAMS).unresolved]
    assert kept == ["my car", "that van", "Our delivery", "the loading dock"]  # places are unaffected


def test_a_garment_target_describes_the_person_instead_of_being_a_second_object():
    plan = model_plan(targets=[
        Target(noun="guy", cls=["person"], embed_text="a photo of a person"),
        Target(noun="brown shirt", cls=[], embed_text="a brown shirt"),
    ])
    fixed = sanitize(plan, CAMS).targets
    assert [t.noun for t in fixed] == ["guy"]
    assert fixed[0].attributes == ["brown"] and "brown shirt" in fixed[0].embed_text


@pytest.mark.asyncio
@pytest.mark.parametrize("phrase", ["at what time", "what time", "when"])
async def test_asking_what_time_is_not_a_range_to_clarify(phrase):
    plan = model_plan(time=TimeWindow(phrase=phrase), unresolved=[Referent(text=phrase, role="time")])
    res = await Planner(FakeGateway(plan, backend="local")).plan("at what time did the guy enter the room", CAMS, REF, IST)
    assert res.plan.time is None and res.plan.unresolved == []


@pytest.mark.parametrize("phrase, bounds", [
    ("at 00:13:53", ("00:13", "00:14")), ("at 13:53", ("13:53", "13:54")), ("at 1pm", ("13:00", "13:01")),
    ("between 4pm to 5pm", ("16:00", "17:00")), ("at 13:24 to 13:27", ("13:24", "13:27")),
    ("in 3 days", None), ("on 2026-10-09", None), ("at 25:00", None),
])
def test_clock_times_said_outright_become_time_of_day_bounds(phrase, bounds):
    from evora.query.planner import _clock_bounds
    assert _clock_bounds(phrase) == bounds


@pytest.mark.asyncio
async def test_a_clock_time_in_the_question_is_not_clarified():
    plan = model_plan(time=TimeWindow(phrase="at 00:13:53"), unresolved=[Referent(text="at 00:13:53", role="time")])
    res = await Planner(FakeGateway(plan, backend="local")).plan("where did it go (from the lobby at 00:13:53)", CAMS, REF, IST)
    assert res.plan.unresolved == []
    assert (res.plan.time.tod_after, res.plan.time.tod_before) == ("00:13", "00:14")


@pytest.mark.asyncio
async def test_a_model_plan_with_the_same_time_twice_means_that_minute():
    plan = model_plan(time=TimeWindow(phrase="at 00:13:53", tod_after="00:13:53", tod_before="00:13:53"))
    res = await Planner(FakeGateway(plan, backend="local")).plan("where was the guy at 00:13:53", CAMS, REF, IST)
    assert (res.plan.time.tod_after, res.plan.time.tod_before) == ("00:13", "00:14")


# ------------------------------------------------- clock times said outright, and what is not one
@pytest.mark.parametrize("phrase,expected", [
    ("at 13:53", ("13:53", "13:54")),
    ("at 00:13:53", ("00:13", "00:14")),
    ("at 23:59", ("23:59", "00:00")),
    ("from 4pm to 5pm", ("16:00", "17:00")),
    ("8am to 6pm", ("08:00", "18:00")),
    ("13:24 to 13:27", ("13:24", "13:27")),
    ("between 10:12 and 10:13", ("10:12", "10:13")),
    ("at 8pm", ("20:00", "20:01")),
    ("between 8 and 9 pm", ("20:00", "21:00")),          # the 8 takes the pm of the 9
    ("between 8 and 9pm", ("20:00", "21:00")),
    ("8 to 9 am", ("08:00", "09:00")),
    ("2018-03-09 10:12", ("10:12", "10:13")),            # the digits of a date are not times
    ("on 3/9 at 10:12", ("10:12", "10:13")),
])
def test_clock_times_said_outright_are_read_exactly(phrase, expected):
    assert planner_mod._clock_bounds(phrase) == expected


@pytest.mark.parametrize("phrase", [
    "after 8 pm", "before 6 am", "around 9:30", "until 5pm",     # open-ended or approximate: not a minute
    "in 3 days", "9", "between 8 and 9",                           # numbers that are not clock times
    "between 11 and 1 pm",                                         # could cross noon: ask rather than guess
    "13:99", "25:10", "0pm",                                       # impossible times
])
def test_what_is_not_an_exact_clock_time_is_left_to_be_asked_about(phrase):
    assert planner_mod._clock_bounds(phrase) is None


@pytest.mark.parametrize("text,expected", [
    ("at what time did he enter", True), ("what time", True), ("which day was it", True),
    ("when", True), ("when did the red car arrive", True), ("when was it", True),
    ("when it was dark", False), ("when the lights were off", False), ("after 8pm", False),
])
def test_only_a_question_for_the_time_is_dropped_from_the_plan(text, expected):
    assert planner_mod._is_time_question(text) is expected


def test_a_carried_item_becomes_a_carrying_attribute_of_the_person():
    person = Target(noun="person", cls=["person"], embed_text="a photo of a person")
    bag = Target(noun="backpack", cls=[], embed_text="a photo of a backpack")
    merged = planner_mod._repair_targets([person, bag])
    assert len(merged) == 1 and merged[0].noun == "person"
    assert merged[0].attributes == ["backpack"]                     # so attribute matching can use it
    assert merged[0].embed_text == "a photo of a person carrying backpack"
    shirt = planner_mod._repair_targets([
        Target(noun="person", cls=["person"], embed_text="a photo of a person"),
        Target(noun="red jacket", cls=[], attributes=["red"], embed_text="a photo of a red jacket")])
    assert shirt[0].attributes == ["red"] and shirt[0].embed_text == "a photo of a person wearing red jacket"


# ------------------------------------------------- questions about actions perception detects need no model to plan
@pytest.mark.asyncio
@pytest.mark.parametrize("question,intent,noun", [
    ("did a vehicle reverse in the last hour", "exists", "vehicle"),
    ("did a person get out of a vehicle", "exists", "person"),
    ("how many times did a car stop", "count", "car"),
    ("when did a car first start moving", "first", "car"),
    ("were two people talking to each other", "exists", "person"),
])
async def test_an_action_the_system_detects_is_planned_without_a_model(question, intent, noun):
    gw = FakeGateway()
    res = await Planner(gw).plan(question, CAMS, REF, IST)
    assert gw.calls == [] and res.plan.source == "fastpath"
    assert res.plan.intent == intent and res.plan.targets[0].noun == noun and res.plan.action == "any"


@pytest.mark.asyncio
async def test_the_action_plan_keeps_the_time_the_camera_and_the_right_subject_classes():
    res = await Planner(FakeGateway()).plan("did a vehicle make a u-turn on Lobby in the last hour", CAMS, REF, IST)
    assert res.plan.camera_ids == ["cam_02"] and (res.plan.time.start, res.plan.time.end) == (REF - 3600, REF)
    assert set(res.plan.targets[0].cls) == {"car", "truck", "bus", "motorcycle"}
    person = await Planner(FakeGateway()).plan("did a person get into a vehicle", CAMS, REF, IST)
    assert person.plan.targets[0].cls == ["person"]               # the person is the one acting, the vehicle is not a target


@pytest.mark.asyncio
async def test_actions_nobody_detects_still_go_to_the_planner():
    gw = FakeGateway(plan=model_plan())
    res = await Planner(gw).plan("did a person pick something up", CAMS, REF, IST)
    assert len(gw.calls) == 1 and res.plan.source == "llm", "an undetected action is not shortcut: the planner is asked"
