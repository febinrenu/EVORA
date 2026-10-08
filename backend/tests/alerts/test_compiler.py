import pytest
from contracts.models import Referent, Target, TimeWindow

from evora.alerts.compiler import Compiled, CompileError, NeedsClarification
from tests.alerts.conftest import place_plan

TEXT = "tell me when someone enters the main gate after 8pm"


async def compile_(env, text=TEXT, **plan):
    env.planner.add(text, **plan)
    return await env.compiler.compile(text)


async def test_a_line_watch_with_direction_and_time(env):
    env.line()
    env.remember_gate()
    got = await compile_(env, action="enter", time=TimeWindow(tod_after="20:00"), **place_plan())
    assert isinstance(got, Compiled)
    r = got.rule
    assert (r.targets, r.events, r.direction, r.tod_after, r.tod_before) == (["person"], ["cross_line"], "a_to_b", "20:00", None)
    assert (r.zone_id, r.camera_ids, r.place, r.cooldown_s) == ("z_gate", ["cam_01"], "main gate", 30.0)
    assert r.summary == "Alert when a person enters main gate after 20:00."


async def test_leaving_a_line_is_the_other_direction(env):
    env.line()
    env.remember_gate()
    got = await compile_(env, action="exit", **place_plan())
    assert (got.rule.events, got.rule.direction) == (["cross_line"], "b_to_a")


async def test_polygon_watches_use_entry_and_exit_events(env):
    env.polygon()
    env.kb.create("place", "yard", {"camera_id": "cam_01", "zone_id": "z_yard"})
    ref = Referent(text="yard", role="place")
    enter = await compile_(env, "watch the yard", action="enter", place=ref, unresolved=[ref])
    leave = await compile_(env, "watch the yard leaving", action="exit", place=ref, unresolved=[ref])
    assert enter.rule.events == ["enter_zone"] and leave.rule.events == ["exit_zone"] and enter.rule.direction is None


async def test_a_bare_place_means_the_natural_event_for_that_zone(env):
    env.line()
    env.remember_gate()
    assert (await compile_(env, action="any", **place_plan())).rule.events == ["cross_line"]


async def test_a_camera_name_is_a_camera_filter_not_a_question(env):
    got = await compile_(env, "tell me when a person appears on the lobby camera", action="appear", camera_ids=["cam_02"])
    assert isinstance(got, Compiled)
    assert (got.rule.camera_ids, got.rule.events, got.rule.zone_id) == (["cam_02"], ["appear"], None)
    assert "Lobby" in got.rule.summary


async def test_no_place_means_every_camera(env):
    got = await compile_(env, "tell me when a person appears anywhere", action="appear")
    assert got.rule.camera_ids == [] and got.rule.summary == "Alert when a person appears at any camera."


async def test_a_remembered_time_word_fills_the_hours(env):
    env.line()
    env.remember_gate()
    env.kb.create("time", "after hours", {"tod_after": "20:00", "tod_before": "06:00"})
    got = await compile_(
        env, "tell me when someone enters the main gate after hours", action="enter",
        **place_plan(unresolved=[Referent(text="main gate", role="place"), Referent(text="after hours", role="time")]),
    )
    assert (got.rule.tod_after, got.rule.tod_before) == ("20:00", "06:00")


async def test_an_unknown_place_asks_once_and_creates_nothing(env):
    got = await compile_(env, action="enter", **place_plan())
    assert isinstance(got, NeedsClarification)
    assert got.request.kind == "choose_camera" and [o.camera_id for o in got.request.options] == ["cam_01", "cam_02"]
    assert env.memory.clarifier.pending(got.request.query_id).text == TEXT


async def test_the_retry_after_the_answer_compiles(env):
    from contracts.models import ClarifyResponse

    first = await compile_(env, action="enter", **place_plan())
    env.memory.apply(ClarifyResponse(query_id=first.request.query_id, camera_id="cam_01"))
    again = await env.compiler.compile(TEXT)
    assert isinstance(again, Compiled) and again.rule.camera_ids == ["cam_01"]


async def test_a_deleted_zone_falls_back_to_watching_the_camera(env):
    env.remember_gate(zone_id="z_gone")
    got = await compile_(env, action="enter", **place_plan())
    assert got.rule.zone_id is None and got.rule.events == ["appear"] and got.rule.camera_ids == ["cam_01"]


@pytest.mark.parametrize("text", ["", "   "])
async def test_empty_text_is_refused(env, text):
    with pytest.raises(CompileError):
        await env.compiler.compile(text)


async def test_no_object_to_watch_is_refused(env):
    with pytest.raises(CompileError, match="what to watch"):
        await compile_(env, "tell me when something happens", targets=[], action="appear")


async def test_an_action_that_cannot_apply_is_refused(env):
    env.line()
    env.remember_gate()
    with pytest.raises(CompileError, match="does not apply"):
        await compile_(env, "tell me when someone stays at the main gate", action="dwell", **place_plan())


async def test_a_planner_failure_is_a_readable_error(env):
    from evora.query.planner import PlanningError

    async def broken(*_):
        raise PlanningError("could not plan this question")

    env.planner.plan = broken
    with pytest.raises(CompileError, match="could not plan"):
        await env.compiler.compile("something")


async def test_a_non_standing_plan_is_accepted_because_the_user_used_the_watch_endpoint(env):
    got = await compile_(
        env, "any car at the lobby camera", intent="exists", camera_ids=["cam_02"], action="appear",
        targets=[Target(noun="car", cls=["car"], attributes=["red"], embed_text="a red car")],
    )
    assert isinstance(got, Compiled) and got.rule.targets == ["car"] and got.rule.attributes == ["red"]
