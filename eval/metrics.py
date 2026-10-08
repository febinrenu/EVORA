"""Metrics from PLAN section 9.3, computed from ground truth and what a system returned.

A system (ours, or a baseline) produces one `RunResult` per query. `score` turns the
list into the numbers reported per split, always with the count behind each number.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from contracts.models import Answer, Evidence

from eval.queries import GroundTruthHit, QueryItem

TAU_S = 2.0  # slack added around a ground-truth window
POSITIVE_VERDICTS = {"yes", "found", "partial"}
NEGATIVE_VERDICTS = {"no", "not_found"}
NO_MODEL_SOURCES = {"fastpath", "cache"}  # planned without any language-model call


@dataclass
class RunResult:
    query_id: str
    answer: Answer | None
    asked_clarify: bool = False
    reasks: int = 0  # clarifications for referents that were already bound
    ttfa_ms: float | None = None
    ttva_ms: float | None = None
    error: str | None = None
    plan_source: str | None = None  # QueryPlan.source: fastpath | cache | llm | local_llm


@dataclass
class Metric:
    value: float | None
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {"value": None if self.value is None else round(self.value, 4), "n": self.n}


@dataclass
class Report:
    split: str
    n_queries: int
    metrics: dict[str, Metric] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "n_queries": self.n_queries,
                "metrics": {k: m.to_dict() for k, m in self.metrics.items()}}


# ---------------------------------------------------------------- matching
def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def is_correct(ev: Evidence, hits: list[GroundTruthHit], tau: float = TAU_S) -> bool:
    """Right camera, and the evidence window overlaps a ground-truth window widened by tau."""
    return any(ev.camera_id == h.camera_id and ev.t_end >= h.start - tau and ev.t_start <= h.end + tau for h in hits)


def is_correct_strict(ev: Evidence, hits: list[GroundTruthHit], tau: float = TAU_S) -> bool:
    """Right camera, and the moment the evidence points at (`t_peak`) lies inside a ground-truth window widened by tau.

    The lenient `is_correct` accepts any overlap between the whole evidence window and the labelled one, so a result
    that spans most of a minute is right almost by construction. This one cannot be met by returning a long window.
    """
    return any(ev.camera_id == h.camera_id and h.start - tau <= ev.t_peak <= h.end + tau for h in hits)


def _first_correct_rank(evidence: list[Evidence], hits: list[GroundTruthHit], strict: bool = False) -> int | None:
    check = is_correct_strict if strict else is_correct
    for rank, ev in enumerate(evidence, start=1):
        if check(ev, hits):
            return rank
    return None


def temporal_iou(ev: Evidence, hits: list[GroundTruthHit]) -> float:
    best = 0.0
    for h in hits:
        if h.camera_id != ev.camera_id:
            continue
        inter = _overlap(ev.t_start, ev.t_end, h.start, h.end)
        union = (ev.t_end - ev.t_start) + (h.end - h.start) - inter
        if union > 0:
            best = max(best, inter / union)
    return best


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _mean(values: list[float]) -> Metric:
    return Metric(sum(values) / len(values) if values else None, len(values))


# ------------------------------------------------------------------ scoring
def score(items: list[QueryItem], results: list[RunResult], split: str = "all") -> Report:
    by_id = {r.query_id: r for r in results}
    report = Report(split=split, n_queries=len(items))
    m = report.metrics

    def answered(item: QueryItem) -> Answer | None:
        r = by_id.get(item.id)
        return r.answer if r else None

    # retrieval quality: queries that have ground-truth hits
    retrieval = [i for i in items if i.is_retrieval]
    hit1: list[float] = []
    hit5: list[float] = []
    rr: list[float] = []
    hit1_s: list[float] = []
    hit5_s: list[float] = []
    rr_s: list[float] = []
    cam_ok: list[float] = []
    ts_err: list[float] = []
    ious: list[float] = []
    for item in retrieval:
        ans = answered(item)
        evidence = list(ans.evidence) if ans else []
        rank = _first_correct_rank(evidence, item.expected.hits)
        hit1.append(1.0 if rank == 1 else 0.0)
        hit5.append(1.0 if rank is not None and rank <= 5 else 0.0)
        rr.append(1.0 / rank if rank else 0.0)
        strict = _first_correct_rank(evidence, item.expected.hits, strict=True)
        hit1_s.append(1.0 if strict == 1 else 0.0)
        hit5_s.append(1.0 if strict is not None and strict <= 5 else 0.0)
        rr_s.append(1.0 / strict if strict else 0.0)
        if evidence:
            top = evidence[0]
            right_cam = any(top.camera_id == h.camera_id for h in item.expected.hits)
            cam_ok.append(1.0 if right_cam else 0.0)
            ious.append(temporal_iou(top, item.expected.hits))
            if right_cam:
                centres = [(h.start + h.end) / 2 for h in item.expected.hits if h.camera_id == top.camera_id]
                ts_err.append(min(abs(top.t_peak - c) for c in centres))
        else:
            cam_ok.append(0.0)
            ious.append(0.0)
    m["hit@1"], m["hit@5"], m["mrr"] = _mean(hit1), _mean(hit5), _mean(rr)
    m["hit@1_strict"], m["hit@5_strict"], m["mrr_strict"] = _mean(hit1_s), _mean(hit5_s), _mean(rr_s)
    m["camera_accuracy"] = _mean(cam_ok)
    m["temporal_iou"] = _mean(ious)
    m["timestamp_error_s"] = Metric(statistics.median(ts_err) if ts_err else None, len(ts_err))

    # existence: every yes/no style query, negatives included
    exist = [i for i in items if i.intent not in ("count", "path", "standing")]
    correct = tp = fp = fn = 0
    for item in exist:
        ans = answered(item)
        predicted = bool(ans and ans.verdict in POSITIVE_VERDICTS and ans.evidence)
        actual = bool(item.expected.hits)
        correct += predicted == actual
        tp += predicted and actual
        fp += predicted and not actual
        fn += (not predicted) and actual
    m["existence_accuracy"] = Metric(correct / len(exist) if exist else None, len(exist))
    # an answer that says "I can't verify that action" is neither right nor wrong: report how often it abstains and how
    # often it is right when it does answer, so honesty is not scored as a miss and an abstention is not hidden
    abstained = [i for i in exist if (a := answered(i)) is not None and a.unsupported_action]
    committed = [i for i in exist if i not in abstained]
    right = sum(1 for i in committed if (bool((a := answered(i)) and a.verdict in POSITIVE_VERDICTS and a.evidence))
                == bool(i.expected.hits))
    m["abstain_rate"] = Metric(len(abstained) / len(exist) if exist else None, len(exist))
    m["existence_accuracy_answered"] = Metric(right / len(committed) if committed else None, len(committed))
    f1_den = 2 * tp + fp + fn
    m["existence_f1"] = Metric(2 * tp / f1_den if f1_den else None, len(exist))

    negatives = [i for i in items if i.is_negative]
    neg_ok = sum(1 for i in negatives if (a := answered(i)) is not None and a.verdict in NEGATIVE_VERDICTS)
    m["negative_precision"] = Metric(neg_ok / len(negatives) if negatives else None, len(negatives))

    counts = [i for i in items if i.intent == "count" and i.expected.count is not None]
    count_ok = sum(1 for i in counts if (a := answered(i)) is not None and a.count == i.expected.count)
    m["count_accuracy"] = Metric(count_ok / len(counts) if counts else None, len(counts))
    errors = []
    for i in counts:
        a = answered(i)
        errors.append(abs((a.count if a is not None and a.count is not None else 0) - i.expected.count))
    m["count_mae"] = Metric(sum(errors) / len(errors) if errors else None, len(errors))
    m["count_within_1"] = Metric(sum(1 for e in errors if e <= 1) / len(errors) if errors else None, len(errors))

    # clarify-once
    asked_tp = asked_fp = asked_fn = 0
    for item in items:
        r = by_id.get(item.id)
        if r is None:
            continue
        should = bool(item.first_time_requires_clarify)
        asked_tp += r.asked_clarify and should
        asked_fp += r.asked_clarify and not should
        asked_fn += (not r.asked_clarify) and should
    m["ask_precision"] = Metric(asked_tp / (asked_tp + asked_fp) if asked_tp + asked_fp else None, asked_tp + asked_fp)
    m["ask_recall"] = Metric(asked_tp / (asked_tp + asked_fn) if asked_tp + asked_fn else None, asked_tp + asked_fn)
    m["reask_count"] = Metric(float(sum(r.reasks for r in results if r.query_id in {i.id for i in items})), len(items))

    # paths
    hops_total = hops_ok = 0
    for item in (i for i in items if i.intent == "path" and i.expected.path):
        ans = answered(item)
        got = list(ans.path) if ans else []
        for idx, gt in enumerate(item.expected.path):
            hops_total += 1
            if idx < len(got):
                hop = got[idx]
                hops_ok += (hop.camera_id == gt.camera_id and hop.t_out >= gt.start - TAU_S
                            and hop.t_in <= gt.end + TAU_S)
    m["path_hop_accuracy"] = Metric(hops_ok / hops_total if hops_total else None, hops_total)

    # latency
    wanted = {i.id for i in items}
    ttfa = [r.ttfa_ms for r in results if r.query_id in wanted and r.ttfa_ms is not None]
    ttva = [r.ttva_ms for r in results if r.query_id in wanted and r.ttva_ms is not None]
    for name, vals in (("ttfa", ttfa), ("ttva", ttva)):
        m[f"{name}_p50_ms"] = Metric(percentile(vals, 50), len(vals))
        m[f"{name}_p95_ms"] = Metric(percentile(vals, 95), len(vals))
    sourced = [r for r in results if r.query_id in wanted and r.plan_source]
    no_model = sum(1 for r in sourced if r.plan_source in NO_MODEL_SOURCES)
    m["no_model_share"] = Metric(no_model / len(sourced) if sourced else None, len(sourced))
    m["errors"] = Metric(float(sum(1 for r in results if r.error)), len(results))
    return report
