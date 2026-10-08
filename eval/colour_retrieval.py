"""Does asking for a colour put the right tracks first? Scored against human colour labels, not our own colour model.

Ground truth is `labels.json` from scripts/colour_label_tool.py: a person has a labelled upper and lower colour, a
vehicle one colour, each possibly 'unsure'. For every camera and colour that has at least one labelled track with it
and at least one without it, the question "a person wearing <colour> on <camera>" is run through retrieval and the
ranking is read over the *labelled* tracks only (an unlabelled track's colour is unknown, so it cannot be scored).
A track counts as a positive when either person slot has the colour, as a negative when both slots are known and
neither does, and is left out when a slot is 'unsure'. The same queries are run with attribute grounding off (image
similarity only) and compared with a random order of the same tracks, the exact chance level.

    python -m eval.colour_retrieval --labels data/share/colour_label/labels.json --root data/share/ws-v5

Caveat that travels with every number: the colour model (M2) was tuned on part of these same labels, so scores on
tracks it was tuned on are optimistic for the colour step; the labels cover a sample of tracks, not all of them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

SPLIT_BY_CAMERA = {  # from eval/frozen.json via eval/meva_school_cameras.json; cameras never listed there can only be dev
    "cam_03": "dev", "cam_05": "dev", "cam_07": "dev", "cam_02": "test", "cam_06": "test", "cam_01": "judge_sim",
    "cam_04": "dev", "cam_08": "dev",
}
UNKNOWN = {None, "", "unsure"}
VEHICLE_CLASSES = ["car", "truck", "bus", "van"]
MIN_JUDGED = 2


def slots(row: dict[str, Any]) -> list[str | None]:
    return [row.get("label_color")] if row.get("kind") == "vehicle" else [row.get("label_upper"), row.get("label_lower")]


def judge(row: dict[str, Any], colour: str) -> bool | None:
    """True: the track has the colour. False: it is known not to. None: a slot is unsure, so it cannot be scored."""
    values = slots(row)
    if colour in values:
        return True
    return None if any(v in UNKNOWN for v in values) else False


def build_queries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One query per (camera, kind, colour) with a scoreable positive and a scoreable negative."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        groups[(row["camera"], row["kind"])].append(row)
    queries = []
    for (camera, kind), rows in sorted(groups.items()):
        colours = sorted({v for r in rows for v in slots(r) if v not in UNKNOWN})
        for colour in colours:
            verdicts = {r["id"]: judge(r, colour) for r in rows}
            positives = {i for i, v in verdicts.items() if v is True}
            judged = {i for i, v in verdicts.items() if v is not None}
            if positives and len(judged) - len(positives) >= 1 and len(judged) >= MIN_JUDGED:
                classes = sorted({r["cls"] for r in rows}) if kind == "vehicle" else ["person"]
                queries.append({"camera": camera, "kind": kind, "colour": colour, "classes": classes,
                                "positives": sorted(positives), "judged": sorted(judged),
                                "split": SPLIT_BY_CAMERA.get(camera, "dev")})
    return queries


def first_positive_rank(ranking: list[str], positives: set[str], judged: set[str]) -> int | None:
    """1-based rank of the first positive among the judged tracks in the order retrieval returned them."""
    rank = 0
    for track_id in ranking:
        if track_id not in judged:
            continue  # unlabelled or unscoreable: it neither helps nor hurts
        rank += 1
        if track_id in positives:
            return rank
    return None


def chance(n_judged: int, n_positive: int) -> tuple[float, float]:
    """Expected (hit@1, reciprocal rank) when the judged tracks are put in a random order."""
    hit1 = n_positive / n_judged
    rr = sum((1 / r) * comb(n_judged - r, n_positive - 1) / comb(n_judged, n_positive)
             for r in range(1, n_judged - n_positive + 2))
    return hit1, rr


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean hit@1 and MRR per system over queries, with n; rows carry per-system ranks and the chance level."""
    out: dict[str, Any] = {"n": len(rows)}
    for system in ("attributes_on", "attributes_off"):
        ranks = [r[system] for r in rows]
        out[system] = {"hit@1": round(sum(1 for k in ranks if k == 1) / len(ranks), 4) if ranks else None,
                       "mrr": round(sum(1 / k for k in ranks if k) / len(ranks), 4) if ranks else None}
    out["chance"] = {"hit@1": round(sum(r["chance_hit1"] for r in rows) / len(rows), 4) if rows else None,
                     "mrr": round(sum(r["chance_rr"] for r in rows) / len(rows), 4) if rows else None}
    return out


async def _rank_all(queries: list[dict[str, Any]], workspace: str, root: Path | None, attributes: bool) -> list[int | None]:
    from contracts.models import QueryPlan, Target

    from eval.systems import build_ours
    from evora.query.retrieve import SearchScope

    system = build_ours(workspace, root, overrides={"retrieval.attributes": attributes})
    retriever = system.ctx.router._retriever
    ranks: list[int | None] = []
    try:
        for q in queries:
            noun = "person" if q["kind"] == "person" else "car"
            text = (f"a photo of a person wearing {q['colour']}" if q["kind"] == "person"
                    else f"a photo of a {q['colour']} car")
            plan = QueryPlan(intent="list", camera_ids=[q["camera"]], targets=[Target(
                noun=noun, cls=q["classes"], attributes=[q["colour"]], embed_text=text)])
            found = await retriever.search(plan, SearchScope(frozenset([q["camera"]])))
            ranking = [c.track.id for c in found.candidates]
            ranks.append(first_positive_rank(ranking, set(q["positives"]), set(q["judged"])))
    finally:
        await system.aclose()
    return ranks


async def evaluate(labels: Path, workspace: str, root: Path | None) -> dict[str, Any]:
    items = json.loads(labels.read_text(encoding="utf-8"))["items"]
    queries = build_queries(items)
    on = await _rank_all(queries, workspace, root, True)
    off = await _rank_all(queries, workspace, root, False)
    rows = []
    for q, a, b in zip(queries, on, off, strict=True):
        h, rr = chance(len(q["judged"]), len(q["positives"]))
        rows.append({**{k: q[k] for k in ("camera", "kind", "colour", "split")}, "n_judged": len(q["judged"]),
                     "n_positive": len(q["positives"]), "attributes_on": a, "attributes_off": b,
                     "chance_hit1": h, "chance_rr": rr})
    result: dict[str, Any] = {"labels": str(labels), "queries": rows, "all": summarise(rows), "by_kind": {}, "by_split": {}}
    for key, field in (("by_kind", "kind"), ("by_split", "split")):
        for value in sorted({r[field] for r in rows}):
            result[key][value] = summarise([r for r in rows if r[field] == value])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=Path("data/share/colour_label/labels.json"))
    parser.add_argument("--workspace", default="meva-school")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("eval/reports/colour_retrieval.json"))
    args = parser.parse_args(argv)
    result = asyncio.run(evaluate(args.labels, args.workspace, args.root))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for scope in ("all",):
        s = result[scope]
        print(f"{scope}: n={s['n']} on {s['attributes_on']} off {s['attributes_off']} chance {s['chance']}")
    for key in ("by_kind", "by_split"):
        for value, s in result[key].items():
            print(f"{value:10s} n={s['n']:2d} on {s['attributes_on']} off {s['attributes_off']} chance {s['chance']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
