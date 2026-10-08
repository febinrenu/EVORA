# ruff: noqa: E501
"""Tune the colour model's thresholds on labelled crops and fit calibrated confidence.

Usage: python scripts/colour_tune.py labels.json [--workspace meva-school] [--write]

The labels (human labels from scripts/colour_label_tool.py, or the local model's from colour_vlm_check.py as a stand-in)
are split by position: even items tune, odd items test. Thresholds are searched by coordinate descent on the tune half
only; the score printed for the test half is the honest one. The objective is the mean of exact accuracy and macro recall
(every colour counts equally), so a method cannot win by calling everything black. A confidence calibration is fitted on
the tune half (isotonic regression) and checked on the test half. `--write` stores the tuned values in
config/colour_calibration.json, which `evora.perception.colourmodel` loads. Masks are computed once and cached.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import colour_eval as ce  # noqa: E402

GRID = {
    "achro_chroma": [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 22.0],
    "black_l": [16.0, 20.0, 24.0, 28.0, 32.0],
    "white_l": [72.0, 76.0, 80.0, 84.0, 88.0],
    "trim": [0.15, 0.2, 0.25, 0.3, 0.35],
    "dark_floor_l": [6.0, 10.0, 14.0, 18.0],
    "dark_floor_chroma": [20.0, 28.0, 36.0, 48.0],
    "knn": [5, 7, 11],
    "lower_achro_chroma": [4.0, 6.0, 8.0, 10.0, 12.0, 16.0],
    "lower_black_l": [10.0, 14.0, 18.0, 22.0, 24.0, 28.0],
    "vehicle_achro_chroma": [8.0, 12.0, 16.0, 20.0],
    "vehicle_black_l": [14.0, 18.0, 24.0, 30.0],
    "vehicle_white_l": [56.0, 62.0, 68.0, 74.0, 80.0],
}


def prepare(items: list[dict], workspace: str):
    """Load crops, masks and white-balance gains once."""
    import cv2
    from evora.perception import attributes as at
    from evora.perception.segment import get_segmenter

    root = REPO / "workspaces" / workspace
    seg = get_segmenter()
    gains: dict[str, object] = {}
    prepared = []
    for it in items:
        cam, n = it["camera"], int(it["id"].split(":t")[1])
        crop = cv2.imread(str(root / "media" / "crops" / cam / f"t{n:06d}_0.jpg"))
        if crop is None:
            continue
        if cam not in gains:
            frames = [cv2.imread(str(f)) for f in sorted((root / "media" / "scenes" / cam).glob("*.jpg"))[::10][:20]]
            gains[cam] = at.white_balance_gains([f for f in frames if f is not None])
        mask = seg.mask(crop, it["kind"]) if seg is not None else None
        prepared.append((it, crop, gains[cam], mask))
    return prepared


def predict_all(prepared, params, split_items: set[str]) -> dict:
    from evora.perception import colourmodel as cm

    out = {}
    for it, crop, g, mask in prepared:
        if it["id"] not in split_items:
            continue
        for slot, val in cm.predict(it["kind"], crop, g, mask, params).items():
            out[(it["id"], slot)] = val
    return out


def objective(items: list[dict], preds: dict, split: str) -> tuple[float, float, float]:
    rows = ce.load_pairs(items, preds, split)
    if not rows:
        return 0.0, 0.0, 0.0
    exact = sum(1 for t, p, *_ in rows if t == p) / len(rows)
    by: dict = defaultdict(lambda: [0, 0])
    for t, p, *_ in rows:
        by[t][0] += 1
        by[t][1] += t == p
    macro = sum(v[1] / v[0] for v in by.values()) / len(by)
    return 0.5 * (exact + macro), exact, macro


def isotonic(conf: list[float], correct: list[bool]) -> tuple[tuple[float, float], ...]:
    """Pool-adjacent-violators fit of accuracy as a non-decreasing function of raw confidence."""
    order = np.argsort(conf)
    xs, ys = np.asarray(conf)[order], np.asarray(correct, dtype=float)[order]
    blocks = [[y, 1, x, x] for x, y in zip(xs, ys, strict=True)]       # mean, weight, x_min, x_max
    out: list[list[float]] = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            a, c = out[-2], out.pop()
            w = a[1] + c[1]
            out[-1] = [(a[0] * a[1] + c[0] * c[1]) / w, w, a[2], c[3]]
    points: list[tuple[float, float]] = []
    for mean, _, lo, hi in out:
        points += [(float(lo), float(mean)), (float(hi), float(mean))]
    dedup: dict[float, float] = {}
    for x, y in points:
        dedup[x] = y
    return tuple(sorted(dedup.items()))


def ece(conf: list[float], correct: list[bool], mapping=None, bins: int = 5) -> float:
    c = np.asarray(conf, dtype=float)
    if mapping:
        xs, ys = zip(*mapping, strict=True)
        c = np.interp(c, xs, ys)
    ok = np.asarray(correct, dtype=float)
    total = 0.0
    for b in range(bins):
        m = (c >= b / bins) & (c < (b + 1) / bins + (b == bins - 1))
        if m.any():
            total += m.mean() * abs(ok[m].mean() - c[m].mean())
    return float(total)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--write", action="store_true", help="store the tuned values in config/colour_calibration.json")
    args = ap.parse_args()
    from evora.perception import colourmodel as cm

    data = json.loads(args.labels.read_text(encoding="utf-8"))
    items = data["items"]
    ws = args.workspace or data.get("workspace", "meva-school")
    tune_ids = {it["id"] for i, it in enumerate(items) if i % 2 == 0}
    test_ids = {it["id"] for i, it in enumerate(items) if i % 2 == 1}
    print(f"preparing crops and masks for {len(items)} items ...", flush=True)
    prepared = prepare(items, ws)
    base = cm.ColourParams()

    def score(params, split):
        ids = tune_ids if split == "tune" else test_ids
        return objective(items, predict_all(prepared, params, ids), split)

    print(f"{'':<24} {'objective':>9} {'exact':>7} {'macro':>7}")
    for label, params in (("defaults on tune", base),):
        s = score(params, "tune")
        print(f"{label:<24} {s[0]:9.3f} {s[1]:7.1%} {s[2]:7.1%}")
    s_test_before = score(base, "test")
    print(f"{'defaults on test':<24} {s_test_before[0]:9.3f} {s_test_before[1]:7.1%} {s_test_before[2]:7.1%}")

    best, best_score = base, score(base, "tune")[0]
    for sweep in range(2):
        improved = False
        for name, values in GRID.items():
            for v in values:
                cand = best.with_overrides(**{name: v})
                s = score(cand, "tune")[0]
                if s > best_score + 1e-9:
                    best, best_score, improved = cand, s, True
                    print(f"  sweep {sweep + 1}: {name}={v} -> tune objective {s:.3f}", flush=True)
        if not improved:
            break
    changed = {k: getattr(best, k) for k in GRID if getattr(best, k) != getattr(base, k)}
    s_tune, s_test = score(best, "tune"), score(best, "test")
    print(f"\ntuned values: {changed or 'unchanged'}")
    print(f"{'tuned on tune':<24} {s_tune[0]:9.3f} {s_tune[1]:7.1%} {s_tune[2]:7.1%}")
    print(f"{'tuned on TEST (honest)':<24} {s_test[0]:9.3f} {s_test[1]:7.1%} {s_test[2]:7.1%}   (defaults on test: {s_test_before[0]:.3f})")

    # confidence calibration, fitted on tune and checked on test
    def conf_pairs(ids):
        preds = predict_all(prepared, best, ids)
        rows = ce.load_pairs(items, preds, "all")
        rows = [r for r in rows if r[3] in ids and r[2] is not None]
        return [r[2] for r in rows], [r[0] == r[1] for r in rows]

    c_tune, k_tune = conf_pairs(tune_ids)
    c_test, k_test = conf_pairs(test_ids)
    mapping = isotonic(c_tune, k_tune) if len(c_tune) >= 20 else ()
    if mapping:
        print(f"calibration error on test: raw {ece(c_test, k_test):.3f} -> calibrated {ece(c_test, k_test, mapping):.3f} ({len(c_test)} predictions)")
    if args.write:
        out = {k: getattr(best, k) for k in GRID}
        out["calibration"] = [list(p) for p in mapping]
        target = REPO / "config" / "colour_calibration.json"
        target.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
