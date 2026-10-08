# ruff: noqa: E501
"""Score colour predictions against human labels from scripts/colour_label_tool.py.

Usage: python scripts/colour_eval.py labels.json [--predictor stored|baseline|current] [--workspace meva-school] [--split all|tune|test]

  stored    the predictions saved in the workspace at labelling time (track level, what search sees today)
  baseline  the original method (fixed geometric regions, hand colour prototypes) re-run on the labelled crop
  current   whatever evora.perception.colourmodel does now (added during the colour upgrade)

Labels are split by position: even items form the tune half, odd items the test half. Tune thresholds on
`tune`, and report the final number from `test` only. "Adjacent" counts neighbouring terms as acceptable
(grey/black, red/pink, brown/orange ...), because clothing in poor light sits between them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO))

COLOURS = ["black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"]
ADJACENT = {frozenset(p) for p in [
    ("black", "grey"), ("grey", "white"), ("red", "pink"), ("red", "orange"), ("red", "brown"), ("orange", "brown"),
    ("orange", "yellow"), ("yellow", "green"), ("blue", "purple"), ("purple", "pink"), ("black", "brown"), ("black", "blue"),
    ("green", "blue"), ("pink", "white"), ("yellow", "white"), ("brown", "grey")]}
SLOTS = {"person": ("upper", "lower"), "vehicle": ("color",)}


def is_adjacent(a: str, b: str) -> bool:
    return a == b or frozenset((a, b)) in ADJACENT


def load_pairs(items: list[dict], predictions: dict[tuple[str, str], tuple[str | None, float | None]], split: str):
    """(truth, predicted, confidence, id, slot) for every labelled item in the chosen half."""
    rows = []
    for idx, it in enumerate(items):
        if split == "tune" and idx % 2 or split == "test" and not idx % 2:
            continue
        for slot in SLOTS[it["kind"]]:
            truth = it.get(f"label_{slot}")
            if truth in (None, "unsure", "skip"):
                continue
            pred, conf = predictions.get((it["id"], slot), (None, None))
            rows.append((truth, pred or "none", conf, it["id"], slot))
    return rows


def report(rows: list[tuple], title: str) -> dict:
    n = len(rows)
    if not n:
        print(f"{title}: no labelled items")
        return {}
    exact = sum(1 for t, p, *_ in rows if t == p)
    adj = sum(1 for t, p, *_ in rows if is_adjacent(t, p))
    print(f"\n== {title}: {n} labels | exact {exact / n:.1%} | exact-or-adjacent {adj / n:.1%}")
    per = defaultdict(lambda: [0, 0, 0])      # truth count, true positives, predicted count
    for t, p, *_ in rows:
        per[t][0] += 1
        per[p][2] += 1
        if t == p:
            per[t][1] += 1
    print(f"{'term':<8} {'labelled':>8} {'recall':>7} {'precision':>9}")
    recalls = []
    for term in COLOURS:
        tc, tp, pc = per[term]
        if tc or pc:
            rec = tp / tc if tc else float("nan")
            print(f"{term:<8} {tc:>8} {rec:>7.0%} {(tp / pc if pc else float('nan')):>9.0%}")
            if tc:
                recalls.append(rec)
    print(f"macro recall {sum(recalls) / len(recalls):.1%} (every colour counts equally; exact accuracy favours the common ones)")
    conf = Counter((t, p) for t, p, *_ in rows if t != p)
    print("most common mistakes (true -> predicted):", ", ".join(f"{t}->{p} x{c}" for (t, p), c in conf.most_common(8)) or "none")
    bins = defaultdict(lambda: [0, 0])
    for t, p, c, *_ in rows:
        if c is not None:
            b = min(int(c * 5), 4)
            bins[b][0] += 1
            bins[b][1] += t == p
    if bins:
        print("confidence vs accuracy:", ", ".join(f"{b / 5:.1f}-{(b + 1) / 5:.1f}: {v[1] / v[0]:.0%} of {v[0]}" for b, v in sorted(bins.items())))
    return {"n": n, "exact": exact / n, "adjacent": adj / n, "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0}


def stored_predictions(items: list[dict]) -> dict:
    out = {}
    for it in items:
        if it["kind"] == "person":
            out[(it["id"], "upper")] = (it.get("pred_upper"), it.get("pred_conf"))
            out[(it["id"], "lower")] = (it.get("pred_lower"), None)
        else:
            out[(it["id"], "color")] = (it.get("pred_color"), it.get("pred_conf"))
    return out


def rescored_predictions(items: list[dict], workspace: str, which: str) -> dict:
    import cv2
    from evora.perception import attributes as at

    root = REPO / "workspaces" / workspace
    gains_by_cam: dict[str, object] = {}
    out = {}
    for it in items:
        cam, n = it["camera"], int(it["id"].split(":t")[1])
        crop = cv2.imread(str(root / "media" / "crops" / cam / f"t{n:06d}_0.jpg"))
        if crop is None:
            continue
        if cam not in gains_by_cam:
            frames = [cv2.imread(str(f)) for f in sorted((root / "media" / "scenes" / cam).glob("*.jpg"))[::10][:20]]
            gains_by_cam[cam] = at.white_balance_gains([f for f in frames if f is not None])
        gains = gains_by_cam[cam]
        if which == "current":
            from evora.perception import colourmodel as cm

            res = cm.predict(it["kind"], crop, gains)
            for slot, val in res.items():
                out[(it["id"], slot)] = val
            continue
        if it["kind"] == "person":
            up, lo = at.person_regions(crop)
            out[(it["id"], "upper")] = at.dominant_colour(up, gains) or (None, None)
            out[(it["id"], "lower")] = at.dominant_colour(lo, gains) or (None, None)
        else:
            out[(it["id"], "color")] = at.dominant_colour(at.vehicle_region(crop), gains) or (None, None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--predictor", choices=["stored", "baseline", "current"], default="stored")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--split", choices=["all", "tune", "test"], default="all")
    args = ap.parse_args()
    data = json.loads(args.labels.read_text(encoding="utf-8"))
    items = data["items"]
    ws = args.workspace or data.get("workspace", "meva-school")
    done = sum(1 for it in items if any(f"label_{s}" in it for s in SLOTS[it["kind"]]))
    print(f"{args.labels}: {len(items)} items, {done} with at least one label; predictor={args.predictor}, split={args.split}")
    preds = stored_predictions(items) if args.predictor == "stored" else rescored_predictions(items, ws, args.predictor)
    rows = load_pairs(items, preds, args.split)
    for slot, title in (("upper", "upper body"), ("lower", "lower body"), ("color", "vehicles")):
        report([r for r in rows if r[4] == slot], title)
    unsure = sum(1 for it in items for s in SLOTS[it["kind"]] if it.get(f"label_{s}") == "unsure")
    print(f"\nmarked unsure by the labeller: {unsure} (excluded; a high number means the crops themselves are ambiguous)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
