from datetime import datetime
from types import SimpleNamespace

import pytest
from contracts.models import QueryPlan, Referent, Target, TimeWindow

from evora.core.db import close_all, open_db
from evora.llm.schemas import LLMError
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
async def test_uninterpretable_time_adds_a_note_and_leaves_bounds_open():
    plan = model_plan(time=TimeWindow(phrase="after hours"))
    res = await Planner(FakeGateway(plan)).plan(HARD, CAMS, REF, IST)
    assert res.plan.time.start is None and res.plan.time.end is None
    assert any("after hours" in n for n in res.notes)


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
