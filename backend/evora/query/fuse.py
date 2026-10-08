"""Scoring pieces for retrieval: pure functions, no stores, no models.

Signals per track, each in [0, 1] or None when it cannot be judged:
  image      calibrated SigLIP2 similarity of the track's crops to the query wording
  attributes agreement of the stored colour / carrying / vehicle type with the query
  caption    lexical match against the track's caption (BM25)
  scene      similarity of coarse scene tiles that overlap the track in time

The final score is a weighted blend over the signals that exist (weights renormalised),
because the accept threshold for "not found" needs an absolute number. Reciprocal-rank
fusion discards magnitude, so it is not used for the score.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

COLOUR_TERMS = {"black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"}
CARRY_TERMS = {"backpack", "handbag", "suitcase", "large_bag", "umbrella"}
VEHICLE_TYPES = {"car", "suv", "truck", "bus", "van", "motorcycle", "bicycle", "auto_rickshaw"}
IR_NEUTRAL = None  # infrared footage: colour is not evidence either way

DEFAULT_WEIGHTS = {"image": 0.55, "attributes": 0.25, "caption": 0.10, "scene": 0.10}


@dataclass(frozen=True)
class Calibration:
    """Maps a raw cosine to [0, 1] with a logistic.

    Fitted to real SigLIP2 text-to-crop cosines on the MEVA school workspace: a crop of the right class
    scores about 0.105 to 0.12 against "a photo of a <class>", a crop of the wrong class about 0.04 to 0.07,
    so the midpoint sits between them (0.09) and the scale spreads that gap over the logistic's slope.
    """

    midpoint: float = 0.09
    scale: float = 0.015

    def __call__(self, cosine: float) -> float:
        z = (cosine - self.midpoint) / self.scale
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


@dataclass
class TrackSignals:
    track_id: str
    image: float | None = None
    attributes: float | None = None
    caption: float | None = None
    scene: float | None = None
    raw_cosine: float | None = None
    peak_t: float | None = None  # time of the best-matching crop
    why: list[str] = field(default_factory=list)


# --------------------------------------------------------------- aggregation
def aggregate_crops(sims: Sequence[float], unit: str = "track", top: int = 3) -> float:
    """Track-level similarity from its crops' cosines.

    "track": max-mean, half the best crop and half the mean of the best few, so one lucky crop
    cannot carry a track. "frame": the single best crop, like frame-level retrieval.
    """
    if not sims:
        raise ValueError("no similarities to aggregate")
    best = sorted(sims, reverse=True)
    if unit == "frame":
        return best[0]
    return 0.5 * best[0] + 0.5 * (sum(best[:top]) / len(best[:top]))


# --------------------------------------------------------------- attributes
def attribute_score(wanted: Iterable[str], attrs: Mapping[str, object], colour_threshold: float = 0.0) -> float | None:
    """Agreement in [0, 1] between the query's attributes and a track's stored attributes.

    None when nothing can be judged: no attributes asked for, attributes not computed yet, or
    only colours asked for on infrared footage. Unknown attributes never count against a track.
    """
    asked = list(dict.fromkeys(wanted))
    if not asked or not attrs:
        return None
    is_ir = bool(attrs.get("is_ir"))
    carrying = set(attrs.get("carrying") or [])
    colours = {c for c in (attrs.get("color"), attrs.get("upper_color")) if c}
    colour_conf = attrs.get("color_conf")
    conf = float(colour_conf) if isinstance(colour_conf, (int, float)) else 1.0
    vehicle = attrs.get("vehicle_type")

    scores: list[float] = []
    for a in asked:
        if a in COLOUR_TERMS:
            if is_ir or not colours:
                continue  # cannot judge
            scores.append(max(conf, colour_threshold) if a in colours else 0.0)
        elif a in CARRY_TERMS:
            scores.append(1.0 if a in carrying else 0.0)
        elif a in VEHICLE_TYPES and vehicle:
            scores.append(1.0 if vehicle == a else 0.0)
    return sum(scores) / len(scores) if scores else None


def explain_attributes(wanted: Iterable[str], attrs: Mapping[str, object]) -> list[str]:
    out: list[str] = []
    conf = attrs.get("color_conf")
    for a in dict.fromkeys(wanted):
        if a in COLOUR_TERMS and not attrs.get("is_ir") and a in {attrs.get("color"), attrs.get("upper_color")}:
            out.append(f"colour {a} {float(conf):.2f}" if isinstance(conf, (int, float)) else f"colour {a}")
        elif a in CARRY_TERMS and a in (attrs.get("carrying") or []):
            out.append(f"carrying {a.replace('_', ' ')}")
        elif a in VEHICLE_TYPES and attrs.get("vehicle_type") == a:
            out.append(f"vehicle type {a}")
    return out


# --------------------------------------------------------------------- BM25
_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("_", " "))


class BM25:
    """Small BM25 over caption documents keyed by track id (no external dependency)."""

    def __init__(self, docs: Mapping[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        self._tf: dict[str, Counter[str]] = {d: Counter(tokenize(t)) for d, t in docs.items()}
        self._len = {d: sum(c.values()) for d, c in self._tf.items()}
        self._avg = (sum(self._len.values()) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for c in self._tf.values():
            df.update(c.keys())
        n = len(self._tf)
        self._idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> dict[str, float]:
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        out: dict[str, float] = {}
        for doc, tf in self._tf.items():
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = 1 - self._b + self._b * self._len[doc] / (self._avg or 1.0)
                    s += self._idf[t] * f * (self._k1 + 1) / (f + self._k1 * norm)
            if s > 0:
                out[doc] = s
        return out


def squash_bm25(score: float, half: float = 3.0) -> float:
    """BM25 is unbounded; map it to [0, 1] with s / (s + half)."""
    return score / (score + half)


# ------------------------------------------------------------------- blend
def blend(signals: TrackSignals, weights: Mapping[str, float] = DEFAULT_WEIGHTS) -> float:
    """Weighted mean over the signals that exist."""
    total = 0.0
    used = 0.0
    for name in ("image", "attributes", "caption", "scene"):
        value = getattr(signals, name)
        if value is None:
            continue
        w = weights.get(name, 0.0)
        total += w * value
        used += w
    return total / used if used > 0 else 0.0


def scene_support(track_cam: str, t0: float, t1: float, scenes: Iterable[tuple[str, float, float]],
                  pad_s: float = 1.0) -> float | None:
    """Best calibrated scene similarity among tiles of this camera that overlap [t0, t1] (+pad)."""
    best: float | None = None
    for cam, t, s in scenes:
        if cam == track_cam and t0 - pad_s <= t <= t1 + pad_s and (best is None or s > best):
            best = s
    return best


def group_by(items: Iterable[tuple[str, float]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for key, value in items:
        grouped[key].append(value)
    return dict(grouped)
