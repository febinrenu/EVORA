"""Calibrate the memory alias thresholds (tau_hi, tau_lo) against a labelled set of phrase pairs.

The resolver compares a new phrase with a remembered one by embedding cosine:
  cosine >= tau_hi          accept silently and remember the new wording as an alias
  tau_lo <= cosine < tau_hi grey band: ask the language model whether they mean the same thing
  cosine <  tau_lo          not the same: ask the user (or bind to another fact)

A wrong silent accept binds a question to the wrong place with nobody noticing, so tau_hi is
set above (almost) every pair that means something different. tau_lo is set low enough that
(almost) every true paraphrase reaches the grey band, where the model can still confirm it.

    python -m eval.calibrate_alias               # real bge-small (downloads it once)
    python -m eval.calibrate_alias --hashing     # the offline fallback embedder, for comparison
"""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass

import numpy as np

# (phrase asked now, phrase already remembered, kind, same thing?)
PAIRS: list[tuple[str, str, str, bool]] = [
    # places: paraphrases
    ("front gate", "main gate", "place", True), ("entrance gate", "main gate", "place", True),
    ("the main gate", "main gate", "place", True), ("main entrance", "main gate", "place", True),
    ("front entrance", "main entrance", "place", True), ("lobby", "reception", "place", True),
    ("reception area", "lobby", "place", True), ("the lobby", "lobby", "place", True),
    ("car park", "parking lot", "place", True), ("parking", "parking lot", "place", True),
    ("the parking area", "parking lot", "place", True), ("back door", "rear exit", "place", True),
    ("rear door", "back door", "place", True), ("back entrance", "back door", "place", True),
    ("loading dock", "loading bay", "place", True), ("server room", "data room", "place", True),
    ("stairwell", "stairs", "place", True), ("cafeteria", "canteen", "place", True),
    ("staff entrance", "employee entrance", "place", True), ("main gate", "mian gate", "place", True),
    ("front door", "main door", "place", True), ("corridor", "hallway", "place", True),
    # places: different things, including near neighbours
    ("main gate", "back gate", "place", False), ("back gate", "main gate", "place", False),
    ("lobby", "parking lot", "place", False), ("back door", "front door", "place", False),
    ("first floor lobby", "second floor lobby", "place", False), ("main gate", "main building", "place", False),
    ("loading dock", "parking lot", "place", False), ("server room", "meeting room", "place", False),
    ("cafeteria", "lobby", "place", False), ("north gate", "south gate", "place", False),
    ("staff entrance", "visitor entrance", "place", False), ("stairwell", "elevator", "place", False),
    ("corridor", "garage", "place", False), ("gate", "gym", "place", False),
    ("main gate", "main street", "place", False), ("back door", "back gate", "place", False),
    ("reception", "security desk", "place", False), ("loading dock", "main gate", "place", False),
    # objects
    ("my car", "the red car", "object", True), ("my vehicle", "my car", "object", True),
    ("the white van", "our van", "object", True), ("my bike", "my bicycle", "object", True),
    ("my bag", "my backpack", "object", True), ("my car", "my car", "object", True),
    ("my car", "my bike", "object", False), ("my red car", "my blue car", "object", False),
    ("the white van", "the white truck", "object", False), ("my bag", "my car", "object", False),
    ("the delivery van", "the school bus", "object", False), ("my scooter", "my car", "object", False),
    # times of day
    ("night shift", "after hours", "time", True), ("outside business hours", "after hours", "time", True),
    ("after closing", "after hours", "time", True), ("overnight", "night", "time", True),
    ("lunch time", "lunch hour", "time", True), ("early morning", "dawn", "time", True),
    ("working hours", "business hours", "time", True), ("after hours", "after-hours", "time", True),
    ("after hours", "business hours", "time", False), ("morning", "evening", "time", False),
    ("lunch time", "night shift", "time", False), ("weekend", "weekday", "time", False),
    ("rush hour", "after hours", "time", False), ("midnight", "noon", "time", False),
    ("opening time", "closing time", "time", False), ("morning shift", "night shift", "time", False),
]


@dataclass(frozen=True)
class Recommendation:
    tau_hi: float
    tau_lo: float
    false_accepts: int          # different things at or above tau_hi (silent mistakes)
    paraphrases_below_lo: int   # same thing below tau_lo (will be asked about again)
    grey_share: float           # share of all pairs that need the language-model check


def recommend(same: list[float], different: list[float], max_false_accept_rate: float = 0.0,
              min_paraphrase_recall: float = 0.95, margin: float = 0.01) -> Recommendation:
    """Pick thresholds from the two score distributions."""
    if not same or not different:
        raise ValueError("need scores for both kinds of pair")
    diffs = sorted(different, reverse=True)
    allowed = int(max_false_accept_rate * len(diffs))
    # tau_hi sits just above the highest `allowed + 1`-th different pair
    tau_hi = diffs[allowed] + margin
    sames = sorted(same)
    miss = int((1 - min_paraphrase_recall) * len(sames))
    tau_lo = sames[miss] - margin  # at most `miss` paraphrases fall below it
    tau_lo = min(tau_lo, tau_hi - margin)
    all_scores = same + different
    grey = sum(1 for s in all_scores if tau_lo <= s < tau_hi) / len(all_scores)
    return Recommendation(
        round(tau_hi, 3), round(tau_lo, 3),
        sum(1 for s in different if s >= tau_hi), sum(1 for s in same if s < tau_lo), round(grey, 3))


def score_pairs(embed, pairs=PAIRS) -> list[tuple[str, str, str, bool, float]]:
    """Cosine for every pair. `embed(list[str]) -> L2-normalised matrix`."""
    phrases = sorted({p for a, b, _, _ in pairs for p in (a, b)})
    mat = np.asarray(embed(phrases), dtype=np.float32)
    index = {p: i for i, p in enumerate(phrases)}
    return [(a, b, kind, same, float(mat[index[a]] @ mat[index[b]])) for a, b, kind, same in pairs]


def report(scored, current_hi: float | None = None, current_lo: float | None = None) -> str:
    same = [s for *_, y, s in scored if y]
    diff = [s for *_, y, s in scored if not y]
    rec = recommend(same, diff)
    lines = [f"pairs: {len(same)} same-thing, {len(diff)} different",
             f"same-thing cosine:  min {min(same):.3f}  median {statistics.median(same):.3f}  max {max(same):.3f}",
             f"different cosine:   min {min(diff):.3f}  median {statistics.median(diff):.3f}  max {max(diff):.3f}",
             "", "hardest different pairs (highest cosine):"]
    for a, b, kind, _, s in sorted((r for r in scored if not r[3]), key=lambda r: -r[4])[:6]:
        lines.append(f"  {s:.3f}  {kind:6s} {a!r} vs {b!r}")
    lines += ["", "weakest same-thing pairs (lowest cosine):"]
    for a, b, kind, _, s in sorted((r for r in scored if r[3]), key=lambda r: r[4])[:6]:
        lines.append(f"  {s:.3f}  {kind:6s} {a!r} vs {b!r}")
    lines += ["", f"recommended: tau_hi {rec.tau_hi}  tau_lo {rec.tau_lo}",
              f"  silent wrong accepts at tau_hi: {rec.false_accepts}   paraphrases below tau_lo: "
              f"{rec.paraphrases_below_lo}   pairs needing the model check: {rec.grey_share:.0%}"]
    if current_hi is not None and current_lo is not None:
        cur_fa = sum(1 for s in diff if s >= current_hi)
        cur_miss = sum(1 for s in same if s < current_lo)
        cur_grey = sum(1 for s in same + diff if current_lo <= s < current_hi) / len(scored)
        lines.append(f"current config: tau_hi {current_hi}  tau_lo {current_lo}  -> silent wrong accepts {cur_fa}, "
                     f"paraphrases below tau_lo {cur_miss}, model check {cur_grey:.0%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hashing", action="store_true", help="use the offline hashing fallback instead of bge-small")
    parser.add_argument("--tau-hi", type=float, default=0.85)
    parser.add_argument("--tau-lo", type=float, default=0.60)
    args = parser.parse_args(argv)
    try:
        if args.hashing:
            from evora.memory.embedder import HashingEmbedder
            embedder = HashingEmbedder()
            embed = embedder.embed
        else:
            from fastembed import TextEmbedding

            from evora.memory.embedder import MODEL_ID, _normalise

            model = TextEmbedding(MODEL_ID)  # downloads once into fastembed's default cache

            def embed(texts):
                return _normalise(np.asarray(list(model.embed(texts)), dtype=np.float32))
    except ImportError as exc:
        print(f"missing dependency: {exc}. Install fastembed (pip install fastembed).", file=sys.stderr)
        return 2
    print(report(score_pairs(embed), args.tau_hi, args.tau_lo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
