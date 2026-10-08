import pytest
from contracts.models import Referent

from evora.memory.resolve import Ambiguous, Bound, Unknown


def place(text):
    return Referent(text=text, role="place")


async def test_exact_match_ignores_case_articles_and_punctuation(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {"camera_id": "cam_01"}, aliases=["Front Gate"])
    for phrase in ("main gate", "the Main Gate!", "at the main gate", "front gate"):
        r = await env.resolver.resolve(place(phrase))
        assert isinstance(r, Bound) and r.fact.id == f.id and r.via == "exact"
    assert env.kb.get(f.id).use_count == 4
    assert env.embedder.calls == 1, "exact hits never need the embedder (one call = indexing the fact)"


async def test_same_phrase_for_two_facts_is_ambiguous(make_env):
    env = make_env()
    a = env.kb.create("place", "gate east", {"camera_id": "cam_01"}, aliases=["gate"])
    b = env.kb.create("place", "gate west", {"camera_id": "cam_02"}, aliases=["gate"])
    r = await env.resolver.resolve(place("gate"))
    assert isinstance(r, Ambiguous) and {f.id for f in r.facts} == {a.id, b.id}


async def test_kind_is_respected(make_env):
    env = make_env()
    env.kb.create("object", "my car", {})
    assert isinstance(await env.resolver.resolve(place("my car")), Unknown)


async def test_paraphrase_above_tau_hi_binds_and_is_remembered(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {"camera_id": "cam_01"})
    r = await env.resolver.resolve(place("front gate"))
    assert isinstance(r, Bound) and r.fact.id == f.id and r.via == "embedding"
    assert env.kb.get(f.id).aliases == ["front gate"]
    calls = env.embedder.calls
    again = await env.resolver.resolve(place("Front gate"))
    assert again.via == "exact" and env.embedder.calls == calls


async def test_unrelated_phrase_is_unknown(make_env):
    env = make_env()
    env.kb.create("place", "main gate", {})
    assert isinstance(await env.resolver.resolve(place("loading dock")), Unknown)


async def test_two_close_facts_make_the_referent_ambiguous(make_env):
    env = make_env()
    env.kb.create("place", "gate east", {"camera_id": "cam_01"})
    env.kb.create("place", "gate west", {"camera_id": "cam_02"})
    r = await env.resolver.resolve(place("gate"))
    assert isinstance(r, Ambiguous) and len(r.facts) == 2


async def test_grey_band_asks_the_equivalence_check(make_env):
    asked = []

    async def yes(new, known, kind):
        asked.append((new, known, kind))
        return True

    env = make_env(equivalence=yes)
    f = env.kb.create("place", "main gate", {"camera_id": "cam_01"})
    r = await env.resolver.resolve(place("entrance"))
    assert isinstance(r, Bound) and r.fact.id == f.id and r.via == "equivalence"
    assert asked == [("entrance", "main gate", "place")]
    assert "entrance" in env.kb.get(f.id).aliases


async def test_grey_band_no_means_unknown_and_nothing_learned(make_env):
    async def no(*_):
        return False

    env = make_env(equivalence=no)
    f = env.kb.create("place", "main gate", {})
    assert isinstance(await env.resolver.resolve(place("entrance")), Unknown)
    assert env.kb.get(f.id).aliases == []


async def test_equivalence_failure_degrades_to_unknown(make_env):
    async def boom(*_):
        raise RuntimeError("no network")

    env = make_env(equivalence=boom)
    env.kb.create("place", "main gate", {})
    assert isinstance(await env.resolver.resolve(place("entrance")), Unknown)


async def test_no_equivalence_configured_means_unknown(make_env):
    env = make_env()
    env.kb.create("place", "main gate", {})
    assert isinstance(await env.resolver.resolve(place("entrance")), Unknown)


async def test_alias_embed_off_is_exact_only(make_env):
    env = make_env(alias_embed=False)
    env.kb.create("place", "main gate", {})
    calls = env.embedder.calls
    assert isinstance(await env.resolver.resolve(place("front gate")), Unknown)
    assert env.embedder.calls == calls


async def test_camera_name_is_a_safety_net_for_places(make_env):
    env = make_env()
    cam = env.camera("Lobby")
    r = await env.resolver.resolve(place("the lobby"))
    assert isinstance(r, Bound) and r.via == "camera" and r.fact.binding == {"camera_id": cam.id} and not r.persisted
    assert env.kb.list() == [], "camera matches are never stored as facts"


async def test_empty_phrase_is_unknown(make_env):
    assert isinstance(await make_env().resolver.resolve(place("the")), Unknown)


async def test_superseded_binding_wins_for_every_phrasing(make_env):
    env = make_env()
    old = env.kb.create("place", "main gate", {"camera_id": "cam_01"}, aliases=["front gate"])
    env.kb.supersede(old.id, {"camera_id": "cam_03"})
    for phrase in ("main gate", "front gate", "the main gate"):
        r = await env.resolver.resolve(place(phrase))
        assert isinstance(r, Bound) and r.fact.binding == {"camera_id": "cam_03"}


@pytest.mark.parametrize("role,phrase,partner", [("time", "night shift", "after hours")])
async def test_time_aliases_resolve_like_places(make_env, role, phrase, partner):
    async def yes(*_):
        return True

    env = make_env(equivalence=yes)
    f = env.kb.create("time", partner, {"tod_after": "20:00", "tod_before": "06:00"})
    r = await env.resolver.resolve(Referent(text=phrase, role=role))
    assert isinstance(r, Bound) and r.fact.id == f.id


# ---- a place name with a generic word added is the same place ------------------------------------------------------------

@pytest.mark.parametrize("asked", ["the loading bay area", "Loading Bay zone", "at the loading bay side", "loading bay area"])
async def test_a_generic_word_after_a_known_place_does_not_ask_again(make_env, asked):
    env = make_env()
    f = env.kb.create("place", "the loading bay", {"camera_id": "cam_03"}, source="clarification")
    r = await env.resolver.resolve(place(asked))
    assert isinstance(r, Bound) and r.fact.id == f.id and r.via == "variant"
    again = await env.resolver.resolve(place(asked))
    assert isinstance(again, Bound) and again.via == "exact", "the wording is remembered after the first time"


async def test_it_works_the_other_way_round(make_env):
    env = make_env()
    f = env.kb.create("place", "parking zone", {"camera_id": "cam_04", "zone_id": "z1"}, source="clarification")
    r = await env.resolver.resolve(place("the parking"))
    assert isinstance(r, Bound) and r.fact.id == f.id


async def test_two_places_that_differ_only_by_the_generic_word_are_ambiguous(make_env):
    env = make_env()
    a = env.kb.create("place", "east", {"camera_id": "cam_01"})
    b = env.kb.create("place", "east side", {"camera_id": "cam_02"})
    r = await env.resolver.resolve(place("east area"))
    assert isinstance(r, Ambiguous) and {f.id for f in r.facts} == {a.id, b.id}


async def test_a_bare_generic_word_or_another_kind_is_not_stretched(make_env):
    env = make_env()
    env.kb.create("place", "loading bay", {"camera_id": "cam_03"})
    env.kb.create("object", "my car", {})
    assert not isinstance(await env.resolver.resolve(place("area")), Bound)
    assert isinstance(await env.resolver.resolve(Referent(text="my car area", role="object")), Unknown)


def test_place_core():
    from evora.memory.kb import place_core

    assert place_core("The Loading Bay Area") == "loading bay"
    assert place_core("north side area") == "north"
    assert place_core("area") == "area" and place_core("zone") == "zone", "never emptied"
    assert place_core("main gate") == "main gate"
