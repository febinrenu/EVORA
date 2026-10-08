"""Track attributes (contribution C2): colour naming, infrared detection, vehicle type, carrying.

Colour: k-means (k=3) in CIELAB on a masked region of the crop, dominant cluster mapped to one of 11
basic colour terms by nearest prototype with an achromatic rule (low chroma -> black / grey / white by L*).
A per-camera grey-world gain, estimated from background frames, is applied before naming.
Everything here is deterministic and works on plain numpy arrays so it can be tested without models.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

COLOURS = ("black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown")
VEHICLE_TYPES = ("car", "suv", "truck", "bus", "van", "motorcycle", "bicycle", "auto_rickshaw")
VEHICLE_PROMPTS = {
    "car": "a photo of a car",
    "suv": "a photo of an SUV",
    "truck": "a photo of a truck",
    "bus": "a photo of a bus",
    "van": "a photo of a van",
    "motorcycle": "a photo of a motorcycle",
    "bicycle": "a photo of a bicycle",
    "auto_rickshaw": "a photo of an auto rickshaw",
}
BAG_CLASSES = {"backpack": "backpack", "handbag": "handbag", "suitcase": "suitcase", "umbrella": "umbrella"}

# chromatic prototypes as sRGB; converted to Lab on import so they follow the same colour space as the pixels
_PROTO_RGB = {
    "red": (200, 30, 30), "orange": (240, 140, 20), "yellow": (240, 220, 40), "green": (40, 150, 60),
    "blue": (30, 70, 200), "purple": (130, 50, 160), "pink": (245, 150, 190), "brown": (120, 70, 35),
}
CHROMA_ACHROMATIC = 14.0     # Lab chroma below this is black / grey / white
BLACK_L, WHITE_L = 30.0, 75.0
IR_SATURATION = 0.05         # mean HSV saturation (0..1) below this means the frame is greyscale / infrared
MIN_REGION_PIXELS = 24


def _bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """(N,3) uint8 BGR -> (N,3) float32 Lab with L in 0..100 and a, b centred on 0."""
    px = np.ascontiguousarray(bgr.reshape(-1, 1, 3), dtype=np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1:] -= 128.0
    return lab


_PROTO_NAMES = tuple(_PROTO_RGB)
_PROTO_LAB = _bgr_to_lab(np.array([[c[2], c[1], c[0]] for c in _PROTO_RGB.values()], dtype=np.uint8))


def name_colour(lab: np.ndarray) -> tuple[str, float]:
    """Basic colour term for one Lab triple, with a 0..1 confidence from the margin to the runner-up."""
    L, a, b = (float(v) for v in lab)
    chroma = math.hypot(a, b)
    if chroma < CHROMA_ACHROMATIC:
        if L < BLACK_L:
            return "black", min(1.0, (BLACK_L - L) / BLACK_L + 0.5)
        if L > WHITE_L:
            return "white", min(1.0, (L - WHITE_L) / (100 - WHITE_L) + 0.5)
        return "grey", 0.7
    d = np.linalg.norm(_PROTO_LAB - np.array([L, a, b], dtype=np.float32), axis=1)
    order = np.argsort(d)
    best, second = float(d[order[0]]), float(d[order[1]])
    conf = 1.0 - best / max(best + second, 1e-6)
    return _PROTO_NAMES[int(order[0])], float(min(max(conf * 2.0 - 0.0, 0.0), 1.0))


def dominant_colour(region_bgr: np.ndarray, gains: np.ndarray | None = None, *, k: int = 3,
                    max_pixels: int = 800, seed: int = 0) -> tuple[str, float] | None:
    """Dominant colour term of a BGR region, or None when the region is too small to judge."""
    if region_bgr.size == 0:
        return None
    px = region_bgr.reshape(-1, 3)
    if px.shape[0] < MIN_REGION_PIXELS:
        return None
    if px.shape[0] > max_pixels:
        idx = np.random.default_rng(seed).choice(px.shape[0], max_pixels, replace=False)
        px = px[idx]
    if gains is not None:
        px = np.clip(px.astype(np.float32) * gains.astype(np.float32), 0, 255).astype(np.uint8)
    lab = _bgr_to_lab(px)
    kk = min(k, max(1, len(np.unique(lab, axis=0))))
    if kk == 1:
        centre, share = lab.mean(axis=0), 1.0
    else:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        cv2.setRNGSeed(seed)
        _, labels, centres = cv2.kmeans(lab, kk, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=kk)
        top = int(np.argmax(counts))
        centre, share = centres[top], float(counts[top]) / lab.shape[0]
    term, conf = name_colour(centre)
    return term, float(conf * (0.5 + 0.5 * share))


# ---------------------------------------------------------------- camera-level helpers

def white_balance_gains(frames_bgr: Iterable[np.ndarray], clamp: tuple[float, float] = (0.7, 1.4)) -> np.ndarray:
    """Grey-world gains (B, G, R) from the median of background frames; neutral (1, 1, 1) when there are none."""
    small = [cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA) for f in frames_bgr]
    if not small:
        return np.ones(3, dtype=np.float32)
    background = np.median(np.stack(small), axis=0)
    means = background.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1.0)
    return np.clip(gains, *clamp).astype(np.float32)


def saturation_of(frame_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(cv2.resize(frame_bgr, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) / 255.0


def is_ir_frame(frame_bgr: np.ndarray) -> bool:
    """Infrared / night-vision frames are grey: colour terms from them are unreliable."""
    return saturation_of(frame_bgr) < IR_SATURATION


# ---------------------------------------------------------------- regions

def person_regions(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper body (15-50% of the height) and lower body (50-90%), central 60% of the width."""
    h, w = crop.shape[:2]
    x1, x2 = int(w * 0.2), max(int(w * 0.8), int(w * 0.2) + 1)
    return crop[int(h * 0.15): int(h * 0.5), x1:x2], crop[int(h * 0.5): int(h * 0.9), x1:x2]


def vehicle_region(crop: np.ndarray) -> np.ndarray:
    """The central 60% of the crop, which avoids road and background at the edges."""
    h, w = crop.shape[:2]
    return crop[int(h * 0.2): int(h * 0.8), int(w * 0.2): int(w * 0.8)]


def aggregate_colour(votes: Sequence[tuple[str, float] | None]) -> tuple[str | None, float | None]:
    """Combine per-crop colour terms: the term with the largest summed confidence wins."""
    totals: dict[str, float] = {}
    n = 0
    for v in votes:
        if v is None:
            continue
        n += 1
        totals[v[0]] = totals.get(v[0], 0.0) + v[1]
    if not totals:
        return None, None
    term = max(totals, key=lambda t: totals[t])
    return term, float(totals[term] / n)


# ---------------------------------------------------------------- vehicle type

def classify_vehicle(crop_vectors: np.ndarray, type_vectors: np.ndarray, names: Sequence[str] = VEHICLE_TYPES, *,
                     temperature: float = 100.0) -> tuple[str, float] | None:
    """Zero-shot vehicle type from crop embeddings: mean of the crops against one text vector per name."""
    if crop_vectors.size == 0:
        return None
    v = crop_vectors.mean(axis=0)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    logits = temperature * (type_vectors @ v)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    i = int(np.argmax(p))
    return names[i], float(p[i])


# ---------------------------------------------------------------- carrying

@dataclass(frozen=True)
class Box:
    t: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def h(self) -> float:
        return self.y2 - self.y1


def _at(points: Sequence[Box], t: float, tol: float) -> Box | None:
    best = min(points, key=lambda p: abs(p.t - t))
    return best if abs(best.t - t) <= tol else None


def carried_by(person: Sequence[Box], bag: Sequence[Box], *, tol: float = 0.4, min_frames: int = 3,
               large_ratio: float = 0.35) -> tuple[bool, bool]:
    """(carried, large): the bag centre stays inside the (slightly grown) person box for >= `min_frames` samples."""
    hits, large_hits = 0, 0
    for b in bag:
        p = _at(person, b.t, tol)
        if p is None:
            continue
        w, h = p.x2 - p.x1, p.y2 - p.y1
        inside = (p.x1 - 0.15 * w <= b.cx <= p.x2 + 0.15 * w) and (p.y1 <= b.cy <= p.y2 + 0.1 * h)
        if inside:
            hits += 1
            if h > 0 and b.h / h >= large_ratio:
                large_hits += 1
    return hits >= min_frames, large_hits >= min_frames


def carrying_terms(bag_classes_and_large: Iterable[tuple[str, bool]]) -> list[str]:
    """Contract terms for TrackAttrs.carrying: a large bag is reported as large_bag in addition to its class."""
    out: list[str] = []
    for cls, large in bag_classes_and_large:
        term = BAG_CLASSES.get(cls)
        if term and term not in out:
            out.append(term)
        if large and cls in ("handbag", "suitcase", "backpack") and "large_bag" not in out:
            out.append("large_bag")
    return out
