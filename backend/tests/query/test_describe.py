import pytest
from contracts.models import TimeWindow

from evora.llm.schemas import LLMError
from evora.query.describe import (
    MAX_FACTS,
    Fact,
    Narration,
    Overview,
    deterministic_sentences,
    gather_facts,
    narrate,
    valid_narration,
)
from evora.query.timeparse import parse_tz
from tests.query.ws_helpers import Workspace

IST = parse_tz("+05:30")


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.camera("cam_01", "Gate")
    w.camera("cam_02", "Lobby")
    yield w
    w.close()


def remember_zone(ws, zone_id="z1", name="main gate", cam="cam_01"):
    ws.zone(zone_id, cam)
    with ws.db.write() as c:
        c.execute("INSERT INTO memory_facts(id,kind,canonical,aliases,binding,source,created_at) "
                  "VALUES('mf1','place',?,'[]','{}','clarification',0)", (name,))
        c.execute("UPDATE zones SET fact_id='mf1' WHERE id=?", (zone_id,))


def fact(n, text="a person crossed the main gate", t=1105.0, cam="cam_01"):
    return Fact(f"E{n}", cam, f"{cam}:t{n}", "cross_line", t, text, t - 5, t + 5)


def test_notable_events_become_readable_facts_with_the_remembered_place_name(ws):
    remember_zone(ws)
    ws.track("t1", "cam_01", cls="person", attrs={"upper_color": "red", "carrying": ["backpack"]})
    ws.track("t2", "cam_01", cls="car", attrs={"color": "white", "vehicle_type": "suv"}, t0=1200.0, t1=1210.0)
    ws.event("e1", "cam_01", "t1", "cross_line", 1105.0)
    ws.event("e2", "cam_01", "t2", "enter_zone", 1205.0)
    ws.event("e3", "cam_01", "t1", "appear", 1100.0, zone=None)  # background noise: every track has one
    facts, overview = gather_facts(ws.db, set(), None)
    assert [f.text for f in facts] == ["a person in red carrying a backpack crossed the main gate",
                                       "a white suv entered the main gate"]
    assert [f.id for f in facts] == ["E1", "E2"] and facts[0].kind == "cross_line"
    assert overview.tracks_by_class == {"person": 1, "car": 1} and overview.total_events == 3


def test_without_notable_events_it_falls_back_to_appearances_named_by_camera(ws):
    ws.track("t1", "cam_01", cls="person", attrs={"is_ir": True, "upper_color": "blue"})
    ws.event("e1", "cam_01", "t1", "appear", 1100.0, zone=None)
    facts, _ = gather_facts(ws.db, set(), None)
    assert [f.text for f in facts] == ["a person appeared at the Gate camera"]  # infrared colour is not reported


def test_scope_and_window_filter_the_facts(ws):
    ws.track("a", "cam_01", t0=1100.0, t1=1110.0)
    ws.track("b", "cam_02", t0=1100.0, t1=1110.0)
    ws.track("c", "cam_01", t0=5000.0, t1=5010.0)
    for i, (tid, cam, t) in enumerate([("a", "cam_01", 1105.0), ("b", "cam_02", 1105.0), ("c", "cam_01", 5005.0)]):
        ws.event(f"e{i}", cam, tid, "cross_line", t)
    assert {f.track_id for f in gather_facts(ws.db, set(), None)[0]} == {"a", "b", "c"}
    assert {f.track_id for f in gather_facts(ws.db, {"cam_01"}, None)[0]} == {"a", "c"}
    late = gather_facts(ws.db, set(), TimeWindow(start=4000.0, end=6000.0))[0]
    assert [f.track_id for f in late] == ["c"]
    assert gather_facts(ws.db, {"cam_99"}, None) == ([], Overview())


def test_many_events_are_thinned_but_keep_the_first_and_last(ws):
    for i in range(60):
        ws.track(f"t{i}", "cam_01", t0=1000.0 + i * 10, t1=1005.0 + i * 10)
        ws.event(f"e{i}", "cam_01", f"t{i}", "cross_line", 1002.0 + i * 10)
    facts, overview = gather_facts(ws.db, set(), None)
    assert len(facts) == MAX_FACTS and facts[0].t == 1002.0 and facts[-1].t == 1002.0 + 59 * 10
    assert [f.t for f in facts] == sorted(f.t for f in facts) and overview.total_events == 60


def test_time_of_day_bounds_apply_to_events(ws):
    ws.track("day", "cam_01", t0=0.0, t1=10.0)  # 05:30 IST
    ws.event("e1", "cam_01", "day", "cross_line", 5.0)
    ws.track("night", "cam_01", t0=3600 * 20.0, t1=3600 * 20.0 + 10)  # 01:30 IST next day
    ws.event("e2", "cam_01", "night", "cross_line", 3600 * 20.0 + 5)
    window = TimeWindow(tod_after="22:00", tod_before="06:00")
    assert [f.track_id for f in gather_facts(ws.db, set(), window, IST)[0]] == ["day", "night"]
    narrow = TimeWindow(tod_after="00:00", tod_before="02:00")
    assert [f.track_id for f in gather_facts(ws.db, set(), narrow, IST)[0]] == ["night"]


def test_deterministic_summary_counts_and_cites():
    facts = [fact(1, t=1100.0), fact(2, "a car entered the main gate", t=1200.0)]
    overview = Overview({"person": 3, "car": 1}, ["cam_01"], 1000.0, 1300.0, 5)
    out = deterministic_sentences(facts, overview, "the main gate", "yesterday evening", IST, {"cam_01": "Gate"})
    assert out[0].text == "3 people and 1 car were tracked at the main gate yesterday evening."
    assert out[0].fact_ids == ("E1", "E2")
    assert out[1].text.startswith("A person crossed the main gate at ") and out[1].fact_ids == ("E1",)
    assert all(s.fact_ids for s in out)


def test_deterministic_summary_for_nothing_and_for_unmarked_activity():
    nothing = deterministic_sentences([], Overview(), "the lobby camera", "today", IST, {})
    assert nothing[0].text == "No activity was recorded at the lobby camera today." and nothing[0].fact_ids == ()
    quiet = deterministic_sentences([], Overview({"person": 2}, ["cam_01"], 1.0, 2.0, 0), "", "", IST, {})
    assert "2 people were tracked." in quiet[0].text and "None of them crossed" in quiet[0].text
    many = deterministic_sentences([fact(i) for i in range(1, 9)], Overview({"person": 8}), "", "", IST, {})
    assert any("3 more events" in s.text for s in many)


FACTS = {f.id: f for f in (fact(1), fact(2))}


def line(text, *cites):
    return Narration.Line(text=text, cites=list(cites))


def test_valid_narration_accepts_cited_sentences_and_tidies_them():
    out = valid_narration([line("Two people crossed the main gate", "E1", "E2"), line("A car arrived.", "E2")], FACTS)
    assert [s.text for s in out] == ["Two people crossed the main gate.", "A car arrived."]
    assert out[0].fact_ids == ("E1", "E2")


@pytest.mark.parametrize("lines", [
    [],                                                                    # nothing
    [line("Someone crossed.")],                                            # no citation
    [line("Someone crossed.", "E9")],                                      # a fact that does not exist
    [line("Someone crossed.", "gate")],                                    # not a fact id at all
    [line("Someone crossed at 09:14.", "E1")],                             # a clock time
    [line("Someone crossed at 9 pm.", "E1")],                              # a clock time, spoken
    [line("x" * 400, "E1")],                                               # too long
    [line("One.", "E1")] * 5,                                              # too many sentences
    [line("", "E1")],                                                      # empty
])
def test_valid_narration_rejects_anything_ungrounded(lines):
    assert valid_narration(lines, FACTS) is None


class Narrator:
    def __init__(self, lines=None, error=None):
        self.lines, self.error, self.calls = lines or [], error, []

    async def chat_json(self, task, messages, schema):
        self.calls.append((task, messages))
        if self.error:
            raise self.error
        return Narration(sentences=self.lines)


@pytest.mark.asyncio
async def test_narrate_uses_the_describe_task_and_shows_only_the_facts():
    gw = Narrator([line("A person crossed the main gate", "E1")])
    out = await narrate(gw, list(FACTS.values()), Overview({"person": 1}), "the main gate", "today")
    assert out[0].fact_ids == ("E1",) and gw.calls[0][0] == "describe"
    prompt = gw.calls[0][1][1]["content"]
    assert "[E1] a person crossed the main gate" in prompt and "Never write a clock time" in gw.calls[0][1][0]["content"]


@pytest.mark.asyncio
async def test_narrate_gives_up_quietly_when_it_cannot_or_must_not():
    facts = list(FACTS.values())
    assert await narrate(None, facts, Overview(), "", "") is None  # no model configured
    assert await narrate(Narrator(), [], Overview(), "", "") is None  # nothing to describe
    assert await narrate(Narrator(error=LLMError("down")), facts, Overview(), "", "") is None
    assert await narrate(Narrator([line("At 10:00 something", "E1")]), facts, Overview(), "", "") is None
