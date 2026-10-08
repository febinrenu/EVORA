import pytest
from contracts.models import ClarifyResponse, QueryPlan, Referent

from evora.memory.clarify import ClarifyError
from evora.memory.resolve import Ambiguous, Bound, Unknown
from evora.memory.service import MemoryService
from evora.memory.tod import TodError, parse_tod, parse_tod_range


def place(text):
    return Referent(text=text, role="place")


# ---- inferred aliases ------------------------------------------------------------------------------------------

async def test_silently_learned_aliases_are_marked_as_guesses(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {"camera_id": "cam_01"}, aliases=["entrance gate"])
    await env.resolver.resolve(place("front gate"))
    got = env.kb.get(f.id)
    assert got.aliases == ["entrance gate", "front gate"] and got.inferred_aliases == ["front gate"]


async def test_aliases_the_user_gave_are_never_guesses(make_env):
    env = make_env()
    f1 = env.kb.create("place", "gate east", {})
    env.kb.create("place", "gate west", {})
    env.clarifier.ask("q", "t", QueryPlan(intent="exists"), place("gate"), Ambiguous([f1]))
    env.clarifier.apply(ClarifyResponse(query_id="q", fact_id=f1.id))
    got = env.kb.get(f1.id)
    assert "gate" in got.aliases and got.inferred_aliases == []


async def test_a_correction_drops_guessed_aliases_but_keeps_confirmed_ones(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {"camera_id": "cam_01"}, aliases=["entrance gate"])
    await env.resolver.resolve(place("front gate"))  # a guess
    new = env.kb.supersede(f.id, {"camera_id": "cam_03"})
    assert new.aliases == ["entrance gate"] and new.inferred_aliases == []
    assert env.kb.find_exact("place", "front gate") == [], "the guessed phrasing is no longer an exact alias"
    assert (await env.resolver.resolve(place("entrance gate"))).fact.binding == {"camera_id": "cam_03"}


async def test_confirming_a_guess_makes_it_survive_a_correction(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {"camera_id": "cam_01"})
    await env.resolver.resolve(place("front gate"))
    assert env.kb.confirm_alias(f.id, "the Front Gate") is True
    assert env.kb.confirm_alias(f.id, "front gate") is False
    assert env.kb.get(f.id).inferred_aliases == []
    new = env.kb.supersede(f.id, {"camera_id": "cam_03"})
    assert new.aliases == ["front gate"]


async def test_removing_a_guess_through_update_clears_the_mark(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {})
    await env.resolver.resolve(place("front gate"))
    env.kb.update(f.id, aliases=[])
    assert env.kb.get(f.id).inferred_aliases == []


async def test_deleting_a_fact_cascades_its_guesses(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {})
    await env.resolver.resolve(place("front gate"))
    env.kb.delete(f.id)
    with env.db.read() as c:
        assert c.execute("SELECT count(*) FROM memory_inferred").fetchone()[0] == 0


def test_resolutions_expose_the_status_and_facts_the_router_reads(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {})
    assert (Bound(f, "exact").status, Bound(f, "exact").facts) == ("bound", [f])
    assert (Ambiguous([f]).status, Ambiguous([f]).facts) == ("ambiguous", [f])
    assert (Unknown().status, Unknown().facts) == ("unknown", [])


# ---- time aliases ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [("8pm", "20:00"), ("8 PM", "20:00"), ("8:30pm", "20:30"), ("12am", "00:00"), ("12pm", "12:00"), ("6", "06:00"),
     ("20:00", "20:00"), ("noon", "12:00"), ("Midnight", "00:00"), ("6 a.m.", "06:00")],
)
def test_parse_tod(text, expected):
    assert parse_tod(text) == expected


@pytest.mark.parametrize("bad", ["", "25:00", "13pm", "0am", "8:75", "later", "pm"])
def test_parse_tod_rejects(bad):
    with pytest.raises(TodError):
        parse_tod(bad)


@pytest.mark.parametrize(
    "text", ["8pm to 6am", "20:00-06:00", "from 8 pm until 6 am", "between 8pm and 6am", "8pm – 6am"]
)
def test_parse_tod_range(text):
    assert parse_tod_range(text) == ("20:00", "06:00")


@pytest.mark.parametrize("bad", ["8pm", "8pm to", "8pm to 6am to 9am", "evening"])
def test_parse_tod_range_rejects(bad):
    with pytest.raises(TodError):
        parse_tod_range(bad)


def make_service(env):
    from evora.memory.clarify import Clarifier  # noqa: F401 - env already holds one

    return MemoryService(env.db, env.kb, env.resolver, env.clarifier)


async def test_time_phrase_is_defined_once_and_reused_for_paraphrases(make_env):
    async def yes(*_):
        return True

    env = make_env(equivalence=yes)
    svc = make_service(env)
    assert await svc.resolve_time("after hours") is None
    svc.define_time("after hours", "8pm to 6am")
    assert await svc.resolve_time("After Hours") == ("20:00", "06:00")
    assert await svc.resolve_time("night shift") == ("20:00", "06:00")  # a paraphrase, no new question
    assert "night shift" in env.kb.list("time")[0].inferred_aliases


def test_defining_a_time_again_corrects_it(make_env):
    env = make_env()
    svc = make_service(env)
    first = svc.define_time("after hours", "8pm to 6am")
    second = svc.define_time("After hours", "9pm to 5am")
    assert env.kb.get(first.id).superseded_by == second.id
    assert second.binding == {"tod_after": "21:00", "tod_before": "05:00"} and second.source == "correction"
    with pytest.raises(TodError):
        svc.define_time("lunch", "whenever")


def test_typed_time_answer_to_a_clarify_question(make_env):
    env = make_env()
    r = Referent(text="after hours", role="time")
    env.clarifier.ask("q1", "who came after hours", QueryPlan(intent="exists"), r, Unknown())
    fact = env.clarifier.apply(ClarifyResponse(query_id="q1", text="8pm to 6am")).fact
    assert fact.binding == {"tod_after": "20:00", "tod_before": "06:00"}
    env.clarifier.ask("q2", "x", QueryPlan(intent="exists"), Referent(text="late", role="time"), Unknown())
    with pytest.raises(ClarifyError):
        env.clarifier.apply(ClarifyResponse(query_id="q2", text="sometime"))
