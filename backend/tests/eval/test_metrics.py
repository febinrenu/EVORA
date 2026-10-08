from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from contracts.models import Answer, Evidence, PathHop, QueryPlan
from eval.metrics import RunResult, is_correct, percentile, score, temporal_iou
from eval.queries import GroundTruthHit, QueryItem, load_queries

IST = timezone(timedelta(hours=5, minutes=30))


def t(h, m=0, s=0):
    return datetime(2026, 10, 9, h, m, s, tzinfo=IST).timestamp()


def gt(cam, start, end):
    return {"camera_id": cam, "start": start, "end": end}


def item(qid, intent="exists", hits=(), verdict=None, **kw):
    verdict = verdict or ("yes" if hits else "no")
    return QueryItem.model_validate({
        "id": qid, "text": qid, "workspace": "w", "intent": intent,
        "expected": {"verdict": verdict, "hits": list(hits), **kw.pop("expected", {})}, **kw})


def ev(cam, t0, t1, peak=None, eid="e"):
    peak = peak if peak is not None else (t0 + t1) / 2
    return Evidence(id=eid, camera_id=cam, camera_name=cam, t_start=t0, t_end=t1, t_peak=peak, offset_s=0,
                    thumb_url="t", clip_url="c", score=0.5)


def answer(verdict, evidence=(), count=None, path=()):
    return Answer(query_id="q", text="x", verdict=verdict, count=count, evidence=list(evidence), path=list(path),
                  confidence=0.5, plan=QueryPlan(intent="exists"))


def run(qid, ans, **kw):
    return RunResult(qid, ans, **kw)


GOOD = gt("cam_01", t(9, 14), t(9, 14, 10))


def metric(report, name):
    return report.metrics[name]


def test_hit_at_k_and_mrr():
    items = [item(f"q{i}", hits=[GOOD]) for i in range(4)]
    right, wrong = ev("cam_01", t(9, 14, 2), t(9, 14, 8)), ev("cam_01", t(11), t(11, 0, 5))
    results = [
        run("q0", answer("yes", [right, wrong])),                  # rank 1
        run("q1", answer("yes", [wrong, right])),                  # rank 2
        run("q2", answer("yes", [wrong] * 5 + [right])),           # rank 6: outside top 5
        run("q3", answer("not_found")),                            # nothing returned
    ]
    r = score(items, results)
    assert metric(r, "hit@1").value == 0.25 and metric(r, "hit@1").n == 4
    assert metric(r, "hit@5").value == 0.5
    assert metric(r, "mrr").value == pytest.approx((1 + 0.5 + 1 / 6 + 0) / 4)


def test_tau_slack_and_camera_must_match():
    assert is_correct(ev("cam_01", t(9, 14, 11), t(9, 14, 12)), [GroundTruthHit(**GOOD)])      # within 2 s
    assert not is_correct(ev("cam_01", t(9, 14, 13), t(9, 14, 14)), [GroundTruthHit(**GOOD)])  # 3 s late
    assert not is_correct(ev("cam_02", t(9, 14, 2), t(9, 14, 8)), [GroundTruthHit(**GOOD)])


def test_camera_accuracy_timestamp_error_and_iou():
    items = [item("a", hits=[GOOD]), item("b", hits=[GOOD])]
    results = [
        run("a", answer("yes", [ev("cam_01", t(9, 14), t(9, 14, 10), peak=t(9, 14, 8))])),   # centre is 9:14:05
        run("b", answer("yes", [ev("cam_02", t(9, 14), t(9, 14, 10))])),                     # wrong camera
    ]
    r = score(items, results)
    assert metric(r, "camera_accuracy").value == 0.5
    assert metric(r, "timestamp_error_s").value == pytest.approx(3.0) and metric(r, "timestamp_error_s").n == 1
    assert metric(r, "temporal_iou").value == pytest.approx(0.5)  # (1.0 + 0.0) / 2


def test_temporal_iou_partial_overlap():
    e = ev("cam_01", t(9, 14, 5), t(9, 14, 15))
    assert temporal_iou(e, [GroundTruthHit(**GOOD)]) == pytest.approx(5 / 15)


def test_existence_negatives_and_f1():
    items = [item("pos1", hits=[GOOD]), item("pos2", hits=[GOOD]), item("neg1"), item("neg2")]
    right = ev("cam_01", t(9, 14), t(9, 14, 10))
    results = [
        run("pos1", answer("yes", [right])),     # TP
        run("pos2", answer("no")),               # FN
        run("neg1", answer("no")),               # TN
        run("neg2", answer("yes", [right])),     # FP
    ]
    r = score(items, results)
    assert metric(r, "existence_accuracy").value == 0.5
    assert metric(r, "existence_f1").value == pytest.approx(0.5)
    assert metric(r, "negative_precision").value == 0.5 and metric(r, "negative_precision").n == 2


def test_missing_results_count_as_wrong_not_as_skipped():
    items = [item("pos", hits=[GOOD]), item("neg")]
    r = score(items, [])
    assert metric(r, "hit@1").value == 0.0 and metric(r, "negative_precision").value == 0.0


def test_count_accuracy():
    items = [item("c1", "count", verdict="count", expected={"count": 3}),
             item("c2", "count", verdict="count", expected={"count": 2})]
    r = score(items, [run("c1", answer("count", count=3)), run("c2", answer("count", count=5))])
    assert metric(r, "count_accuracy").value == 0.5


def test_clarify_metrics():
    asks = {"first_time_requires_clarify": ["main gate"]}
    items = [item("ask", hits=[GOOD], **asks), item("known", hits=[GOOD]), item("missed", hits=[GOOD], **asks)]
    results = [
        run("ask", answer("yes"), asked_clarify=True),
        run("known", answer("yes"), asked_clarify=True, reasks=1),   # asked although already bound
        run("missed", answer("yes"), asked_clarify=False),
    ]
    r = score(items, results)
    assert metric(r, "ask_precision").value == 0.5   # 1 right of 2 asks
    assert metric(r, "ask_recall").value == 0.5      # 1 of 2 required
    assert metric(r, "reask_count").value == 1.0


def test_path_hop_accuracy():
    gt_path = [gt("cam_01", t(9, 0), t(9, 1)), gt("cam_02", t(9, 2), t(9, 3)), gt("cam_03", t(9, 4), t(9, 5))]
    items = [item("p", "path", verdict="found", expected={"path": gt_path})]
    hops = [PathHop(camera_id="cam_01", camera_name="a", t_in=t(9, 0), t_out=t(9, 1), evidence_id="1"),
            PathHop(camera_id="cam_03", camera_name="c", t_in=t(9, 2), t_out=t(9, 3), evidence_id="2")]  # wrong camera
    r = score(items, [run("p", answer("found", path=hops))])
    assert metric(r, "path_hop_accuracy").value == pytest.approx(1 / 3) and metric(r, "path_hop_accuracy").n == 3


def test_latency_percentiles_and_errors():
    items = [item(f"q{i}") for i in range(10)]
    results = [run(f"q{i}", answer("no"), ttfa_ms=float(i + 1) * 10, ttva_ms=float(i + 1) * 100) for i in range(10)]
    results[0].error = "boom"
    r = score(items, results)
    assert metric(r, "ttfa_p50_ms").value == 50.0 and metric(r, "ttfa_p95_ms").value == 100.0
    assert metric(r, "ttva_p50_ms").value == 500.0
    assert metric(r, "errors").value == 1.0


def test_percentile_edges():
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0
    assert percentile([1, 2, 3, 4], 50) == 2


def test_empty_inputs_give_none_not_zero():
    r = score([], [])
    assert metric(r, "hit@1").value is None and metric(r, "hit@1").n == 0
    assert r.to_dict()["metrics"]["mrr"] == {"value": None, "n": 0}


# ------------------------------------------------------------------- loader
def test_example_file_matches_the_documented_format():
    items = load_queries([Path(__file__).resolve().parents[3] / "eval" / "queries" / "_example.yaml"])
    assert [i.id for i in items] == ["own_017", "own_018"]
    first, second = items
    assert first.expected.hits[0].start == t(9, 14) and first.first_time_requires_clarify == ["main gate"]
    assert second.is_negative and second.expected.verdict == "no"
    assert first.is_retrieval and not second.is_retrieval


def test_loader_rejects_naive_times_duplicates_and_filters_by_split(tmp_path):
    good = """
- {id: a, text: x, workspace: w, intent: exists, split: test,
   expected: {verdict: yes, hits: [{camera_id: c, start: "2026-10-09T09:00:00+05:30",
                                    end: "2026-10-09T09:00:05+05:30"}]}}
- {id: b, text: y, workspace: w, intent: exists, expected: {verdict: no}}
"""
    f = tmp_path / "set.yaml"
    f.write_text(good)
    assert [i.id for i in load_queries([f])] == ["a", "b"]
    assert [i.id for i in load_queries([f], split="test")] == ["a"]
    bare_no = tmp_path / "bare.yaml"
    bare_no.write_text("- {id: z, text: x, workspace: w, intent: exists, expected: {verdict: no}}")
    assert load_queries([bare_no])[0].expected.verdict == "no"  # YAML reads bare `no` as False

    naive = tmp_path / "naive.yaml"
    naive.write_text(good.replace("+05:30", ""))
    with pytest.raises(ValueError):
        load_queries([naive])
    dup = tmp_path / "dup.yaml"
    line = "- {id: a, text: x, workspace: w, intent: exists, expected: {verdict: no}}\n"
    dup.write_text(line + line)
    with pytest.raises(ValueError, match="duplicate"):
        load_queries([dup])


def test_no_model_share_counts_fastpath_and_cache_but_not_llm_plans():
    items = [item(f"q{i}") for i in range(5)]
    sources = ["fastpath", "cache", "llm", "local_llm", None]
    results = [run(f"q{i}", answer("no"), plan_source=src) for i, src in enumerate(sources)]
    share = metric(score(items, results), "no_model_share")
    assert share.value == 0.5 and share.n == 4  # the result with no recorded source is not counted
    assert metric(score(items, []), "no_model_share").value is None
