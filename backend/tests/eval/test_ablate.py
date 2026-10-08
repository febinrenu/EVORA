import json

import pytest
from contracts.models import Answer, Evidence, QueryPlan
from eval import ablate, harness
from eval.metrics import RunResult
from eval.queries import QueryItem

START, END = "2026-10-09T09:14:00+05:30", "2026-10-09T09:14:10+05:30"


def make_item(qid):
    return QueryItem.model_validate({
        "id": qid, "text": qid, "workspace": "w", "intent": "exists", "split": "test",
        "expected": {"verdict": "yes", "hits": [{"camera_id": "cam_01", "start": START, "end": END}]}})


class Sys:
    """Right answers unless the switch under test is off."""

    name = "s"

    def __init__(self, overrides):
        self.overrides = overrides

    async def run(self, item):
        gt = item.expected.hits[0]
        if self.overrides.get("retrieval.attributes") is False:
            return RunResult(item.id, Answer(query_id="q", text="", verdict="no", confidence=0.1,
                                             plan=QueryPlan(intent="exists")))
        ev = Evidence(id="e", camera_id="cam_01", camera_name="G", t_start=gt.start, t_end=gt.end,
                      t_peak=gt.start, offset_s=0, thumb_url="t", clip_url="c", score=0.9)
        return RunResult(item.id, Answer(query_id="q", text="", verdict="yes", evidence=[ev], confidence=0.9,
                                         plan=QueryPlan(intent="exists")), ttfa_ms=10.0)


ITEMS = [make_item("a"), make_item("b")]


@pytest.mark.asyncio
async def test_each_switch_is_compared_with_the_full_system():
    seen = []

    def factory(overrides):
        seen.append(dict(overrides))
        return Sys(overrides)

    rows = await ablate.run_ablation(factory, ITEMS, "test", switches=ablate.SWITCHES[:3])
    assert [r.label for r in rows][0] == "full system" and len(rows) == 4
    assert seen[0] == {} and seen[1] == {"retrieval.unit": "frame"} and seen[2] == {"retrieval.attributes": False}
    full, no_unit, no_attr, _ = rows
    assert full.report.metrics["hit@1"].value == 1.0 and no_unit.report.metrics["hit@1"].value == 1.0
    assert no_attr.report.metrics["hit@1"].value == 0.0


@pytest.mark.asyncio
async def test_switches_that_change_the_index_are_skipped_without_a_reindex_hook():
    rows = await ablate.run_ablation(Sys, ITEMS, "test", switches=[s for s in ablate.SWITCHES if s.id in ("C5", "C6")])
    assert all(r.report is None and "re-indexed" in r.skipped for r in rows[1:])


@pytest.mark.asyncio
async def test_reindex_hook_runs_around_the_ablated_run_and_restores_the_full_index():
    calls = []

    async def reindex(overrides):
        calls.append(dict(overrides))

    c6 = [s for s in ablate.SWITCHES if s.id == "C6"]
    rows = await ablate.run_ablation(Sys, ITEMS, "test", switches=c6, reindex=reindex)
    assert calls == [{"ingest.motion_gate": False}, {}] and rows[1].report is not None


@pytest.mark.asyncio
async def test_a_failing_ablation_is_recorded_and_the_rest_continue():
    def factory(overrides):
        if overrides.get("retrieval.unit"):
            raise RuntimeError("frame index missing")
        return Sys(overrides)

    rows = await ablate.run_ablation(factory, ITEMS, "test", switches=ablate.SWITCHES[:2])
    assert rows[1].report is None and "frame index missing" in rows[1].skipped and rows[2].report is not None


@pytest.mark.asyncio
async def test_markdown_shows_the_contribution_metric_and_its_change(tmp_path):
    rows = await ablate.run_ablation(Sys, ITEMS, "test", switches=ablate.SWITCHES[1:2])
    table = ablate.format_markdown(rows)
    assert table.splitlines()[0].startswith("| Configuration | n | Hit@1")
    assert "| full system | 2 | 1.00 |" in table
    assert "hit@1 0.00 (-1.00)" in table  # attributes off loses everything in this toy system
    jp, mp = ablate.write_reports(rows, tmp_path)
    data = json.loads(jp.read_text())
    assert data["no Attribute grounding (C2)"]["switch"]["key"] == "retrieval.attributes"
    assert mp.read_text().startswith("| Configuration")


def test_skipped_rows_render_with_the_reason():
    row = ablate.Row("no Topology (C5)", ablate.SWITCHES[5], None, "needs the footage re-indexed with the switch off")
    full = ablate.Row("full system", None, harness.score([], [], "test"))
    out = ablate.format_markdown([full, row])
    assert "skipped: needs the footage re-indexed" in out and "n/a" in out


def test_every_contribution_has_one_switch_with_an_owner():
    ids = [s.id for s in ablate.SWITCHES]
    assert {i[:2] for i in ids} == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert all(s.owner in ("M1", "M2", "M3") and s.full != s.ablated for s in ablate.SWITCHES)


def test_cli_reports_that_ours_is_not_wired(tmp_path, capsys):
    q = tmp_path / "q.yaml"
    q.write_text("- {id: a, text: x, workspace: w, intent: exists, split: test, expected: {verdict: no}}\n")
    assert ablate.main(["--queries", str(q), "--out", str(tmp_path)]) == 2
    assert "needs an indexed workspace" in capsys.readouterr().err
