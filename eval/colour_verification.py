"""Does the visual check agree with a human? Scored on the colour labels, through the full router.

For each colour question that eval.colour_retrieval can score ("a person wearing red on G419"), the router answers
once without waiting for the check and once after it. Only tracks that have a human label for that colour are scored.

* verifier accuracy: of the labelled tracks the check looked at, how often it said yes to a track with the colour
  (recall) and no to a track without it (specificity);
* effect of filtering: Hit@1 and precision of the labelled evidence before and after the check set candidates aside.

    python -m eval.colour_verification --labels data/share/colour_label/labels.json --root data/share/ws-v5

The check is a small local vision model, so these numbers belong to that model and to this footage. A track the
check never looked at (only the top few are checked) is not scored for accuracy.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from eval.colour_retrieval import build_queries


def question(query: dict[str, Any], camera_name: str) -> str:
    if query["kind"] == "person":
        return f"Was there a person wearing {query['colour']} on {camera_name}?"
    return f"Was there a {query['colour']} car on {camera_name}?"


def verifier_counts(checked: dict[str, bool | None], positives: set[str], judged: set[str]) -> dict[str, int]:
    """Confusion counts of the check against the labels, over labelled tracks it gave a yes or a no to."""
    out = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "undecided": 0, "unlabelled": 0}
    for track_id, verdict in checked.items():
        if track_id not in judged:
            out["unlabelled"] += 1
        elif verdict is None:
            out["undecided"] += 1
        elif track_id in positives:
            out["tp" if verdict else "fn"] += 1
        else:
            out["fp" if verdict else "tn"] += 1
    return out


def labelled_view(track_ids: list[str], positives: set[str], judged: set[str]) -> dict[str, Any]:
    """Hit@1 and precision over the labelled tracks in an evidence list (None when it holds none)."""
    seen = [t for t in track_ids if t in judged]
    if not seen:
        return {"n": 0, "hit1": None, "precision": None}
    return {"n": len(seen), "hit1": seen[0] in positives, "precision": sum(1 for t in seen if t in positives) / len(seen)}


def rates(c: dict[str, int]) -> dict[str, Any]:
    pos, neg = c["tp"] + c["fn"], c["tn"] + c["fp"]
    return {"yes_on_positives": round(c["tp"] / pos, 4) if pos else None, "n_positives": pos,
            "no_on_negatives": round(c["tn"] / neg, 4) if neg else None, "n_negatives": neg}


def effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean Hit@1 and precision before and after the check, over queries that have a labelled track in both lists."""
    both = [r for r in rows if r["before"]["n"] and r["after"]["n"]]
    def mean(key: str, when: str) -> float | None:
        values = [float(r[when][key]) for r in both]
        return round(sum(values) / len(values), 4) if values else None
    return {"n": len(both), "n_queries": len(rows),
            "lost_all_labelled": sum(1 for r in rows if r["before"]["n"] and not r["after"]["n"]),
            "hit1_before": mean("hit1", "before"), "hit1_after": mean("hit1", "after"),
            "precision_before": mean("precision", "before"), "precision_after": mean("precision", "after")}


async def evaluate(labels: Path, workspace: str, root: Path | None, limit: int | None = None) -> dict[str, Any]:
    from eval.systems import build_ours

    items = json.loads(labels.read_text(encoding="utf-8"))["items"]
    queries = build_queries(items)[:limit]
    system = build_ours(workspace, root)
    names = {c["id"]: c["name"] for c in (await asyncio.to_thread(_cameras, system))}
    rows: list[dict[str, Any]] = []
    totals = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "undecided": 0, "unlabelled": 0}
    skipped = 0
    try:
        for q in queries:
            events = [e async for e in system.ctx.router.answer(question(q, names[q["camera"]]), "colour_verification")]
            answers = [e.data for e in events if e.type == "answer"]
            plan = next((e.data for e in events if e.type == "plan"), {})
            attrs = [a for t in (plan.get("plan", plan).get("targets") or []) for a in (t.get("attributes") or [])]
            if not answers or q["colour"] not in attrs:
                skipped += 1  # the question was not understood as that colour (or was answered with a clarification)
                continue
            by_ev = {e.data["id"]: e.data["track_id"] for e in events if e.type == "evidence"}
            checked = {by_ev[e.data["evidence_id"]]: e.data["verified"] for e in events
                       if e.type == "verified" and e.data["evidence_id"] in by_ev}
            positives, judged = set(q["positives"]), set(q["judged"])
            counts = verifier_counts(checked, positives, judged)
            for k, v in counts.items():
                totals[k] += v
            rows.append({"camera": q["camera"], "kind": q["kind"], "colour": q["colour"], "split": q["split"],
                         "counts": counts, "checked": len(checked),
                         "before": labelled_view([e["track_id"] for e in answers[0]["evidence"]], positives, judged),
                         "after": labelled_view([e["track_id"] for e in answers[-1]["evidence"]], positives, judged)})
    finally:
        await system.aclose()
    by_kind = {k: effect([r for r in rows if r["kind"] == k]) for k in sorted({r["kind"] for r in rows})}
    return {"labels": str(labels), "queries": rows, "skipped": skipped, "verifier": {**totals, **rates(totals)},
            "effect": effect(rows), "effect_by_kind": by_kind}


def _cameras(system: Any) -> list[dict[str, str]]:
    with system.ctx.router._db.read() as conn:
        return [{"id": r["id"], "name": r["name"]} for r in conn.execute("SELECT id, name FROM cameras")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=Path("data/share/colour_label/labels.json"))
    parser.add_argument("--workspace", default="meva-school")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("eval/reports/colour_verification.json"))
    args = parser.parse_args(argv)
    result = asyncio.run(evaluate(args.labels, args.workspace, args.root, args.limit))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("verifier:", result["verifier"])
    print("effect:", result["effect"])
    for kind, e in result["effect_by_kind"].items():
        print(kind, e)
    print(f"{len(result['queries'])} scored, {result['skipped']} skipped; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
