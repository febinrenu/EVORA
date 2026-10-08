import pytest
from contracts.models import ClarifyResponse
from eval import harness
from eval.queries import QueryItem
from eval.systems import B0System, OursSystem, RawPlanner, apply_overrides, clarify_response

from evora.api.context import AppContext
from evora.core.config import load_config
from evora.llm.schemas import LLMError
from evora.memory.embedder import HashingEmbedder
from evora.query.fuse import Calibration
from evora.query.planner import Planner
from evora.query.retrieve import RetrievalConfig, Retriever
from tests.query.ws_helpers import E, ToyEmbedder, Workspace

RED = {"color": "red", "color_conf": 1.0}


class OfflineGateway:
    """No model reachable: the fast path and memory must carry the whole run."""

    async def chat_json_ex(self, task, messages, schema):
        raise LLMError("offline")

    async def chat_json(self, task, messages, schema):
        raise LLMError("offline")


def item(qid, text, clarify=None, answer=None, hits=True, intent="exists"):
    expected = {"verdict": "yes", "hits": [{"camera_id": "cam_01", "start": 1099.0, "end": 1111.0}]} if hits \
        else {"verdict": "no"}
    return QueryItem.model_validate({"id": qid, "text": text, "workspace": "w", "intent": intent,
                                     "first_time_requires_clarify": clarify or [], "clarify_answer": answer,
                                     "expected": expected})


@pytest.fixture
def ours(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "eval-test")
    monkeypatch.delenv("evora_MOCK", raising=False)
    ctx = AppContext.build(load_config(), tmp_path / "workspaces", gateway=OfflineGateway(), mock=False,
                           embedder=HashingEmbedder())
    ws = Workspace(db=ctx.db, vectors_dir=ctx.ws.vectors_dir)
    ws.camera("cam_01", "Gate", source="/data/gate.mp4")
    ws.track("t_red", "cam_01", attrs=RED, crops=[E[0]], bbox=(0.3, 0.2, 0.45, 0.7))
    ctx.router._retriever = Retriever(ctx.db, ws.store, ToyEmbedder(),
                                      RetrievalConfig(calibration=Calibration(0.5, 0.2)))
    ctx.router._verifier = None  # no crops or face blur in this fixture
    yield OursSystem(ctx), ws
    ws.close()


GATE = "did a red car pass through the main gate"


@pytest.mark.asyncio
async def test_ours_clarifies_from_the_scripted_answer_then_remembers_it(ours):
    system, _ = ours
    first = await system.run(item("q1", GATE, clarify=["main gate"], answer={"camera_id": "cam_01"}))
    assert first.error is None and first.asked_clarify is True and first.reasks == 0
    assert first.answer is not None and first.answer.verdict == "yes"
    assert first.answer.evidence[0].camera_id == "cam_01"
    assert first.plan_source == "fastpath" and 0 <= first.ttfa_ms <= first.ttva_ms

    again = await system.run(item("q2", GATE))  # memory now knows the place: nothing to ask
    assert again.asked_clarify is False and again.reasks == 0 and again.answer.verdict == "yes"

    paraphrase = await system.run(item("q3", "did a red car pass through the Main Gate"))
    assert paraphrase.asked_clarify is False  # same words, different case: still bound


@pytest.mark.asyncio
async def test_ours_counts_a_question_asked_when_ground_truth_says_it_was_already_known(ours):
    system, _ = ours
    await system.run(item("q1", GATE, clarify=["main gate"], answer={"camera_id": "cam_01"}))
    fresh = await system.run(item("q2", "did a red car pass through the back lot", answer={"camera_id": "cam_01"}))
    assert fresh.asked_clarify is True and fresh.reasks == 1  # GT lists nothing to clarify, yet it asked


@pytest.mark.asyncio
async def test_a_question_with_no_scripted_answer_stays_open(ours):
    system, _ = ours
    res = await system.run(item("q1", GATE, clarify=["main gate"]))  # no clarify_answer
    assert res.asked_clarify is True and res.answer is None and res.error is None


@pytest.mark.asyncio
async def test_planning_failures_are_reported_as_errors(ours):
    system, _ = ours
    res = await system.run(item("q1", "an unusual question the fast path cannot parse"))
    assert res.answer is None and "could not plan" in res.error


@pytest.mark.asyncio
async def test_ours_runs_through_the_harness_and_scores(ours):
    system, _ = ours
    queries = [item("q1", GATE, clarify=["main gate"], answer={"camera_id": "cam_01"}), item("q2", GATE)]
    report = await harness.evaluate(system, queries, "dev")
    m = report.metrics
    assert m["hit@1"].value == 1.0 and m["ask_recall"].value == 1.0 and m["reask_count"].value == 0.0
    assert m["no_model_share"].value == 1.0  # both plans came from the fast path


def test_clarify_response_builds_zones_and_passes_other_fields():
    answer = {"camera_id": "cam_gate", "zone": {"kind": "line", "points": [[0.1, 0.7], [0.9, 0.6]]}}
    resp = clarify_response("q1", answer)
    assert isinstance(resp, ClarifyResponse) and resp.zone.camera_id == "cam_gate"
    assert resp.zone.points[1] == (0.9, 0.6)
    timed = clarify_response("q2", {"tod_after": "20:00", "tod_before": "06:00", "ignored": 1})
    assert (timed.tod_after, timed.tod_before) == ("20:00", "06:00")
    assert clarify_response("q3", None) is None and clarify_response("q3", {}) is None


def test_apply_overrides_sets_dotted_keys_on_a_copy():
    cfg = {"retrieval": {"unit": "track"}, "memory": {"alias_embed": True}}
    out = apply_overrides(cfg, {"retrieval.unit": "frame", "ingest.motion_gate": False})
    assert out["retrieval"]["unit"] == "frame" and out["ingest"]["motion_gate"] is False
    assert cfg == {"retrieval": {"unit": "track"}, "memory": {"alias_embed": True}}  # the original is untouched


@pytest.mark.asyncio
async def test_raw_planner_searches_the_whole_question_as_text():
    res = await RawPlanner().plan("anything at all", [], 0.0, None)
    assert res.plan.targets[0].embed_text == "anything at all" and res.plan.camera_ids == []
    assert res.plan.source == "fastpath" and any("not parsed" in n for n in res.notes)


# --------------------------------------------------------------------------- B0
@pytest.fixture
def b0(tmp_path):
    ws = Workspace(tmp_path)
    ws.camera("cam_01", "Gate", t0=1000.0)
    ws.camera("cam_02", "Lobby", t0=1000.0)
    for t in (1100.0, 1101.0, 1102.0):
        ws.scene("cam_01", t, E[0])
    ws.scene("cam_01", 1150.0, E[0], tile="tl")  # tiles are not whole frames
    ws.scene("cam_02", 1100.0, E[1])
    system = B0System(ws.db, ws.store, Planner(None), ToyEmbedder())
    yield system, ws
    ws.close()


@pytest.mark.asyncio
async def test_b0_ranks_whole_frames_and_merges_adjacent_hits(b0):
    system, _ = b0
    res = await system.run(item("q1", "was there a red car at the Gate"))
    ev = res.answer.evidence
    assert res.answer.verdict == "found"
    assert (ev[0].camera_id, ev[0].t_start, ev[0].t_end) == ("cam_01", 1100.0, 1102.0)
    assert all(e.camera_id == "cam_01" for e in ev)  # the camera filter comes from the plan, as for ours
    assert ev[0].offset_s == ev[0].t_peak - 1000.0 and res.plan_source == "fastpath"
    assert len(system.index) == 4  # only whole-frame tiles were loaded: three on cam_01 and one on cam_02


@pytest.mark.asyncio
async def test_b0_uses_the_camera_from_the_scripted_clarification(b0):
    system, _ = b0
    q = item("q1", "was there a red car at the main gate", clarify=["main gate"], answer={"camera_id": "cam_02"})
    res = await system.run(q)
    assert {e.camera_id for e in res.answer.evidence} == {"cam_02"}


@pytest.mark.asyncio
async def test_b0_has_no_negatives_so_it_always_returns_its_best_frames(b0):
    system, _ = b0
    res = await system.run(item("q1", "was there a blue truck at the Lobby", hits=False))
    assert res.answer.verdict == "found" and res.answer.evidence  # the weakness the comparison is meant to show


def test_building_a_system_without_a_workspace_explains_what_is_missing():
    with pytest.raises(harness.SystemNotReady, match="needs an indexed workspace"):
        harness.build_system("ours")
    with pytest.raises(harness.SystemNotReady, match="unknown system"):
        harness.build_system("nope", "ws")


# ------------------------------------------------------------------------ null
@pytest.fixture
def null_sys(tmp_path):
    from eval.systems import NullSystem

    ws = Workspace(tmp_path)
    ws.camera("cam_01", "Gate", t0=1000.0, duration=300.0)
    ws.camera("cam_02", "Lobby", t0=1000.0, duration=300.0)
    yield NullSystem(ws.db, Planner(None)), ws
    ws.close()


@pytest.mark.asyncio
async def test_null_returns_random_moments_inside_the_camera_filter_and_the_window(null_sys):
    system, _ = null_sys
    q = item("q1", "was there a person at the Gate between 00:17 and 00:18")  # 1000 s is 00:16:40 UTC: 1020-1080 s
    res = await system.run(q)
    ev = res.answer.evidence
    assert len(ev) == 5 and {e.camera_id for e in ev} == {"cam_01"}
    assert all(1020.0 <= e.t_peak <= 1080.0 for e in ev)  # inside the window the plan carries
    assert all(e.t_end - e.t_start == 3.0 and e.score == 0.0 for e in ev)
    assert res.plan_source == "fastpath" and res.answer.verdict == "found"


@pytest.mark.asyncio
async def test_null_is_deterministic_per_query_and_differs_between_queries(null_sys):
    system, _ = null_sys
    a = await system.run(item("q1", "was there a person at the Gate"))
    b = await system.run(item("q1", "was there a person at the Gate"))
    c = await system.run(item("q2", "was there a person at the Gate"))
    times = lambda r: [round(e.t_peak, 3) for e in r.answer.evidence]  # noqa: E731
    assert times(a) == times(b) and times(a) != times(c)


@pytest.mark.asyncio
async def test_null_follows_the_scripted_camera_and_never_says_nothing(null_sys):
    system, _ = null_sys
    q = item("q1", "was there a red car at the main gate", clarify=["main gate"], answer={"camera_id": "cam_02"})
    res = await system.run(q)
    assert {e.camera_id for e in res.answer.evidence} == {"cam_02"}
    negative = await system.run(item("q2", "was there a vehicle at the Lobby", hits=False))
    assert negative.answer.verdict == "found"  # chance has no way to answer "nothing there"


@pytest.mark.asyncio
async def test_a_window_outside_the_footage_gives_no_moments_rather_than_invented_ones(null_sys):
    system, _ = null_sys
    res = await system.run(item("q1", "was there a person at the Gate between 05:00 and 05:01"))
    assert res.answer.evidence == [] and res.answer.verdict == "not_found"


def test_null_is_a_registered_system():
    from eval.systems import SYSTEM_BUILDERS

    assert {"ours", "b0", "null"} <= set(SYSTEM_BUILDERS)
