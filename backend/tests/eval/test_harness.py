import json
from pathlib import Path

import pytest
from contracts.models import Answer, Evidence, QueryPlan
from eval import harness
from eval.metrics import RunResult
from eval.queries import QueryItem

GT_START = "2026-10-09T09:14:00+05:30"
GT_END = "2026-10-09T09:14:10+05:30"


def make_item(qid, hit=True):
    expected = {"verdict": "yes", "hits": [{"camera_id": "cam_01", "start": GT_START, "end": GT_END}]} if hit \
        else {"verdict": "no"}
    return QueryItem.model_validate({"id": qid, "text": qid, "workspace": "w", "intent": "exists",
                                     "expected": expected})


def evidence_for(item):
    gt = item.expected.hits[0]
    return Evidence(id="e", camera_id="cam_01", camera_name="Gate", t_start=gt.start, t_end=gt.end,
                    t_peak=(gt.start + gt.end) / 2, offset_s=0, thumb_url="t", clip_url="c", score=0.9)


def answer(verdict, evidence=()):
    return Answer(query_id="q", text="x", verdict=verdict, evidence=list(evidence), confidence=0.5,
                  plan=QueryPlan(intent="exists"))


class Oracle:
    name = "oracle"

    async def run(self, item):
        if item.expected.hits:
            return RunResult(item.id, answer("yes", [evidence_for(item)]), ttfa_ms=5.0, ttva_ms=9.0)
        return RunResult(item.id, answer("no"), ttfa_ms=5.0, ttva_ms=9.0)


class Flaky:
    name = "flaky"

    async def run(self, item):
        if item.id == "boom":
            raise RuntimeError("index missing")
        return RunResult(item.id, answer("no"))


@pytest.mark.asyncio
async def test_perfect_system_scores_perfectly():
    items = [make_item("a"), make_item("b"), make_item("n", hit=False)]
    report = await harness.evaluate(Oracle(), items, "dev")
    m = report.metrics
    assert m["hit@1"].value == 1.0 and m["negative_precision"].value == 1.0 and m["existence_f1"].value == 1.0
    assert m["ttfa_p50_ms"].value == 5.0


@pytest.mark.asyncio
async def test_one_crash_is_recorded_and_the_run_continues():
    seen = []
    results = await harness.run_system(Flaky(), [make_item("ok"), make_item("boom"), make_item("after")],
                                       on_result=lambda item, r: seen.append(item.id))
    assert seen == ["ok", "boom", "after"]
    assert results[1].answer is None and "index missing" in results[1].error
    assert results[0].error is None


@pytest.mark.asyncio
async def test_latency_is_measured_by_the_harness_when_the_system_does_not():
    results = await harness.run_system(Flaky(), [make_item("x")])
    assert results[0].ttva_ms is not None and results[0].ttva_ms >= 0
    assert results[0].ttfa_ms == results[0].ttva_ms


@pytest.mark.asyncio
async def test_markdown_and_scoreboard_row(tmp_path):
    items = [make_item("a"), make_item("n", hit=False)]
    report = await harness.evaluate(Oracle(), items, "dev")
    table = harness.format_markdown({"ours": report})
    assert table.splitlines()[0].startswith("| System | Split | n | Hit@1")
    assert "| ours | dev | 2 | 1.00 |" in table
    row = harness.scoreboard_row(report, "abc1234", baseline_hit1=0.5, notes="smoke")
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:4] == ["abc1234", "dev", "1.00"] and cells[-2:] == ["0.50", "strict Hit@1 1.00 smoke"]
    assert len(cells) == 13  # matches the PROGRESS.md scoreboard header

    json_path, md_path = harness.write_reports({"ours": report}, tmp_path, stem="t")
    assert json.loads(json_path.read_text())["ours"]["metrics"]["hit@1"]["n"] == 1
    assert md_path.read_text().startswith("| System")


def test_missing_values_print_na():
    report = harness.score([], [], "dev")
    assert "n/a" in harness.format_markdown({"x": report})


def test_cli_reports_unwired_systems_and_empty_splits(tmp_path, capsys):
    qfile = tmp_path / "q.yaml"
    qfile.write_text("- {id: a, text: x, workspace: w, intent: exists, split: dev, expected: {verdict: no}}\n")
    assert harness.main(["--system", "ours", "--queries", str(qfile), "--out", str(tmp_path)]) == 2
    assert "needs an indexed workspace" in capsys.readouterr().err
    assert harness.main(["--system", "ours", "--split", "test", "--queries", str(qfile)]) == 2
    assert "No test queries" in capsys.readouterr().err


def test_cli_end_to_end_with_a_registered_system(tmp_path, monkeypatch, capsys):
    qfile = tmp_path / "q.yaml"
    qfile.write_text(
        "- {id: a, text: x, workspace: w, intent: exists, split: dev, expected: {verdict: no}}\n")
    monkeypatch.setattr(harness, "build_system", lambda *args, **kwargs: Oracle())
    assert harness.main(["--system", "oracle", "--queries", str(qfile), "--out", str(tmp_path / "out")]) == 0
    assert "| oracle | dev | 1 |" in capsys.readouterr().out
    assert Path(tmp_path / "out" / "eval_dev.json").exists()


# ------------------------------------------------------------ capability tables
def cap_item(qid, cap, hit=True, intent="exists", count=None):
    expected = {"verdict": "yes", "hits": [{"camera_id": "cam_01", "start": GT_START, "end": GT_END}]} if hit \
        else {"verdict": "count", "count": count} if intent == "count" else {"verdict": "no"}
    tags = [f"cap:{cap}"] if cap else []
    return QueryItem.model_validate({"id": qid, "text": qid, "workspace": "w", "intent": intent, "tags": tags,
                                     "expected": expected})


def test_capability_of_defaults_untagged_queries_to_the_activity_diagnostic():
    assert harness.capability_of(cap_item("a", "object")) == "object"
    assert harness.capability_of(cap_item("b", None)) == harness.ACTIVITY


@pytest.mark.asyncio
async def test_one_report_per_capability_never_a_blended_headline():
    items = [cap_item("o1", "object"), cap_item("o2", "object"), cap_item("n1", "negative", hit=False),
             cap_item("a1", None)]
    results = []
    for it in items:
        results.append(await Oracle().run(it))
    caps = harness.by_capability(items, results, "dev")
    assert list(caps) == ["object", "negative", harness.ACTIVITY]  # table order, only what exists
    assert caps["object"].n_queries == 2 and caps["negative"].n_queries == 1
    assert caps["object"].metrics["hit@1"].value == 1.0 and caps["negative"].metrics["negative_precision"].value == 1.0


@pytest.mark.asyncio
async def test_capability_markdown_lists_unsupported_capabilities_instead_of_inventing_rows():
    items = [cap_item("o1", "object"), cap_item("n1", "negative", hit=False)]
    caps = harness.by_capability(items, [await Oracle().run(i) for i in items], "dev")
    text = harness.format_capability_markdown(caps, "ours")
    assert "| object | 1 | 1.00 |" in text and "System: ours" in text
    assert "colour" not in text.split("Not evaluated")[0]  # no row for a capability without ground truth
    tail = text.split("Not evaluated")[1]
    assert "- colour:" in tail and "- carrying:" in tail and "- path:" in tail
    covered = harness.by_capability([cap_item("c1", "colour")], [await Oracle().run(cap_item("c1", "colour"))], "dev")
    assert "- colour:" not in harness.format_capability_markdown(covered)  # a capability with ground truth leaves the list


def test_count_error_metrics_show_how_far_off_the_counts_are():
    from eval.metrics import RunResult, score

    items = [cap_item(f"c{i}", "count", hit=False, intent="count", count=n) for i, n in enumerate([2, 4, 0, 3])]
    got = [2, 3, 2, None]  # exact, off by one, off by two, no answer
    results = []
    for it, g in zip(items, got, strict=True):
        ans = None if g is None else Answer(query_id="q", text="", verdict="count", count=g, confidence=1.0,
                                            plan=QueryPlan(intent="count"))
        results.append(RunResult(it.id, ans))
    m = score(items, results).metrics
    assert m["count_accuracy"].value == 0.25
    assert m["count_mae"].value == pytest.approx((0 + 1 + 2 + 3) / 4)  # no answer counts as having counted nothing
    assert m["count_within_1"].value == 0.5
