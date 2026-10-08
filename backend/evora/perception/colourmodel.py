"""Colour naming for garments and vehicles, built to be measured and tuned against human labels.

Pipeline for one crop:
  1. choose the pixels that belong to the garment (person mask or a geometric fallback, central part only, optional
     skin and background removal),
  2. white-balance them with the camera's grey-world gains,
  3. find the dominant colour (k-means in CIELAB),
  4. name it: neutral colours by lightness thresholds, coloured ones by the nearest colours of the public xkcd
     colour-name survey (CC0, 949 names people gave to swatches), mapped to the 11 basic terms the search uses,
  5. add a shade ("dark" / "light") from where the colour sits among the survey colours of the same term.
Every choice is a field of `ColourParams`, so `scripts/colour_eval.py` can switch each one on and off and keep it only
when the score on held-out human labels improves. `config/colour_calibration.json` can override the defaults and holds the
confidence calibration.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from evora.core.config import REPO_ROOT
from evora.perception import attributes as at

log = logging.getLogger("evora.perception.colourmodel")

DATA = Path(__file__).parent / "data" / "xkcd_rgb.txt"
CALIBRATION = REPO_ROOT / "config" / "colour_calibration.json"
TERMS = at.COLOURS
# survey names that contain no basic colour word, assigned by hand; everything else is decided by the survey colour's own hue
NAME_TO_TERM = {
    "navy": "blue", "navy blue": "blue", "teal": "green", "maroon": "red", "burgundy": "red", "wine": "red", "crimson":
    "red", "beige": "brown", "tan": "brown", "khaki": "brown", "sand": "brown", "taupe": "brown", "chocolate": "brown",
    "coffee": "brown", "cream": "white", "ivory": "white", "bone": "white", "off white": "white", "silver": "grey",
    "charcoal": "grey", "slate": "grey", "olive": "green", "lime": "green", "mint": "green", "forest": "green", "violet":
    "purple", "lavender": "purple", "lilac": "purple", "magenta": "pink", "fuchsia": "pink", "rose": "pink", "salmon":
    "pink", "coral": "pink", "peach": "pink", "gold": "yellow", "mustard": "yellow", "indigo": "blue", "cyan": "blue",
    "aqua": "blue", "turquoise": "blue", "azure": "blue", "cobalt": "blue", "rust": "brown", "mahogany": "brown", "mauve":
    "purple", "plum": "purple", "eggplant": "purple", "tangerine": "orange", "terracotta": "brown", "lemon": "yellow",
    "dark": "black", "midnight": "black", "ebony": "black", "jet": "black"
}


@dataclass(frozen=True)
class ColourParams:
    # which pixels
    use_mask: bool = True            # person segmentation mask when one is available
    trim: float = 0.25               # drop this share of the width on each side (arms, background)
    skin_filter: bool = False        # remove skin-coloured pixels from the garment region
    background_filter: bool = False  # remove pixels that look like the crop border (background) when there is no mask
    min_pixels: int = 40
    # naming
    use_survey: bool = True          # coloured pixels named by the survey; False = original hand prototypes
    knn: int = 7
    achro_chroma: float = 12.0       # below this Lab chroma the colour is black / grey / white
    black_l: float = 24.0
    white_l: float = 80.0
    dark_floor_l: float = 14.0       # below this lightness only strongly coloured pixels are not black
    dark_floor_chroma: float = 28.0
    # thresholds that differ by slot (trousers are often dark denim; sunlit white cars look underexposed)
    lower_achro_chroma: float = 12.0
    lower_black_l: float = 24.0
    vehicle_achro_chroma: float = 12.0
    vehicle_black_l: float = 24.0
    vehicle_white_l: float = 80.0
    shade_low: float = 0.33          # shade quantiles among survey colours of the same term
    shade_high: float = 0.67
    # confidence calibration: (raw confidence, observed accuracy) points, interpolated
    calibration: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def with_overrides(self, **kw) -> ColourParams:
        return replace(self, **kw)


@dataclass(frozen=True)
class ColourResult:
    term: str | None
    conf: float | None
    name: str | None = None          # e.g. "dark blue"
    shade: str | None = None         # "dark" | "light" | None

    def pair(self) -> tuple[str | None, float | None]:
        return self.term, self.conf


@lru_cache(maxsize=1)
def load_params() -> ColourParams:
    """Defaults, overridden by config/colour_calibration.json when it exists."""
    if not CALIBRATION.is_file():
        return ColourParams()
    raw = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    cal = tuple(tuple(p) for p in raw.pop("calibration", []))
    known = {k: v for k, v in raw.items() if k in ColourParams.__dataclass_fields__ and k != "calibration"}
    return ColourParams(**known, calibration=cal)


# ---------------------------------------------------------------- the survey table

def _term_of(name: str, lab: np.ndarray) -> str:
    tokens = name.replace("-", " ").split()
    found = None
    for tok in tokens:
        base = tok[:-3] if tok.endswith("ish") and len(tok) > 5 else tok
        if base.endswith("dd"):
            base = base[:-1]
        base = {"gray": "grey"}.get(base, base)
        if base in TERMS:
            found = base                       # the last basic word is the head: "greenish blue" is blue
    if found:
        return found
    if name in NAME_TO_TERM:
        return NAME_TO_TERM[name]
    for tok in tokens[::-1]:
        if tok in NAME_TO_TERM:
            return NAME_TO_TERM[tok]
    return at.name_colour(lab)[0]               # no word to go by: the colour itself decides


class Survey:
    def __init__(self) -> None:
        names, rgb = [], []
        for line in DATA.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name, hexv = line.split("\t")[:2]
            names.append(name.strip())
            rgb.append([int(hexv[i:i + 2], 16) for i in (1, 3, 5)])
        bgr = np.array(rgb, dtype=np.uint8)[:, ::-1]
        self.lab = at._bgr_to_lab(bgr)
        self.names = names
        self.terms = np.array([_term_of(n, self.lab[i]) for i, n in enumerate(names)])
        self.term_l = {t: np.sort(self.lab[self.terms == t, 0]) for t in TERMS if (self.terms == t).any()}

    def name(self, lab: np.ndarray, k: int) -> tuple[str, float, str]:
        """(term, vote share of the term among the k nearest survey colours, nearest survey name)."""
        d = np.linalg.norm(self.lab - lab[None, :], axis=1)
        idx = np.argsort(d)[:k]
        w = 1.0 / (d[idx] + 2.0)
        votes: dict[str, float] = {}
        for i, wi in zip(idx, w, strict=True):
            votes[self.terms[i]] = votes.get(self.terms[i], 0.0) + float(wi)
        term = max(votes, key=lambda t: votes[t])
        return term, votes[term] / float(w.sum()), self.names[int(idx[0])]

    def shade(self, term: str, lightness: float, lo: float, hi: float) -> str | None:
        ls = self.term_l.get(term)
        if ls is None or len(ls) < 8:
            return None
        if lightness <= np.quantile(ls, lo):
            return "dark"
        if lightness >= np.quantile(ls, hi):
            return "light"
        return None


_survey: Survey | None = None
_survey_lock = threading.Lock()


def survey() -> Survey:
    global _survey
    with _survey_lock:
        if _survey is None:
            _survey = Survey()
        return _survey


# ---------------------------------------------------------------- naming one colour

def name_lab(lab: np.ndarray, params: ColourParams) -> tuple[str, float, str | None]:
    """(term, raw confidence, shade) for one Lab triple."""
    L, a, b = (float(v) for v in lab)
    chroma = float(np.hypot(a, b))
    if chroma < params.achro_chroma and not (L < params.dark_floor_l and chroma >= params.dark_floor_chroma):
        if L < params.black_l:
            return "black", min(1.0, 0.6 + (params.black_l - L) / params.black_l * 0.4), None
        if L > params.white_l:
            return "white", min(1.0, 0.6 + (L - params.white_l) / (100 - params.white_l) * 0.4), None
        shade = "dark" if L < 45 else "light" if L > 65 else None
        return "grey", 0.7, shade
    if L < params.dark_floor_l and chroma < params.dark_floor_chroma:
        return "black", 0.8, None
    if not params.use_survey:
        term, conf = at.name_colour(lab)
        return term, conf, None
    sv = survey()
    term, share, _ = sv.name(lab, params.knn)
    return term, share, sv.shade(term, L, params.shade_low, params.shade_high)


def calibrate(raw: float, params: ColourParams) -> float:
    """Map a raw confidence to the accuracy observed on labelled data (identity when there is no calibration)."""
    if not params.calibration:
        return float(raw)
    xs, ys = zip(*params.calibration, strict=True)
    return float(np.interp(raw, xs, ys))


# ---------------------------------------------------------------- choosing the garment pixels

def _bands(kind: str, h: int, w: int, trim: float) -> dict[str, tuple[slice, slice]]:
    x = slice(int(w * trim), max(int(w * (1 - trim)), int(w * trim) + 1))
    if kind == "person":
        return {"upper": (slice(int(h * 0.20), int(h * 0.50)), x), "lower": (slice(int(h * 0.55), int(h * 0.90)), x)}
    return {"color": (slice(int(h * 0.20), int(h * 0.80)), x)}


def _skin(bgr: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    return (ycc[:, 1] >= 135) & (ycc[:, 1] <= 173) & (ycc[:, 2] >= 80) & (ycc[:, 2] <= 125)


def garment_pixels(kind: str, crop: np.ndarray, mask: np.ndarray | None, params: ColourParams) -> dict[str, np.ndarray]:
    """{slot: (N, 3) BGR pixels} for the parts of the crop that are clothing (or body paint for vehicles)."""
    h, w = crop.shape[:2]
    box = (0, 0, w, h)
    if mask is not None and params.use_mask and mask.shape == (h, w) and mask.any():
        ys, xs = np.nonzero(mask)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    x0, y0, x1, y1 = box
    sub = crop[y0:y1, x0:x1]
    sub_mask = mask[y0:y1, x0:x1] if (mask is not None and params.use_mask and mask.shape == (h, w)) else None
    border = None
    if params.background_filter and sub_mask is None:
        parts = (sub[:3], sub[-3:], sub[:, :3], sub[:, -3:])
        edge = np.concatenate([q.reshape(-1, 3) for q in parts])
        border = np.median(edge, axis=0)
    out = {}
    for slot, (ys_, xs_) in _bands(kind, *sub.shape[:2], params.trim).items():
        region = sub[ys_, xs_]
        keep = np.ones(region.shape[:2], dtype=bool)
        if sub_mask is not None:
            keep &= sub_mask[ys_, xs_]
        px = region.reshape(-1, 3)
        k = keep.reshape(-1)
        if params.skin_filter and kind == "person":
            k &= ~_skin(px)
        if border is not None:
            k &= np.linalg.norm(px.astype(np.float32) - border.astype(np.float32), axis=1) > 28
        chosen = px[k]
        out[slot] = chosen if len(chosen) >= params.min_pixels else px      # too little left: use the whole band
    return out


def dominant_lab(pixels: np.ndarray, gains: np.ndarray | None, *, k: int = 3, max_pixels: int = 900, seed: int = 0):
    """(Lab centre of the largest cluster, its share of the pixels) or None."""
    if len(pixels) < 8:
        return None
    px = pixels
    if len(px) > max_pixels:
        px = px[np.random.default_rng(seed).choice(len(px), max_pixels, replace=False)]
    if gains is not None:
        px = np.clip(px.astype(np.float32) * gains.astype(np.float32), 0, 255).astype(np.uint8)
    lab = at._bgr_to_lab(px)
    kk = min(k, max(1, len(np.unique(lab, axis=0))))
    if kk == 1:
        return lab.mean(axis=0), 1.0
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centres = cv2.kmeans(lab, kk, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=kk)
    top = int(np.argmax(counts))
    return centres[top], float(counts[top]) / len(lab)


# ---------------------------------------------------------------- public API

def _for_slot(slot: str, params: ColourParams) -> ColourParams:
    """The thresholds that apply to one garment slot."""
    if slot == "lower":
        return replace(params, achro_chroma=params.lower_achro_chroma, black_l=params.lower_black_l)
    if slot == "color":
        return replace(params, achro_chroma=params.vehicle_achro_chroma, black_l=params.vehicle_black_l,
                       white_l=params.vehicle_white_l)
    return params


def predict_detail(kind: str, crop: np.ndarray, gains: np.ndarray | None = None, mask: np.ndarray | None = None,
                   params: ColourParams | None = None) -> dict[str, ColourResult]:
    """Colour of each garment slot of one crop: person -> upper, lower; vehicle -> color."""
    params = params or load_params()
    result: dict[str, ColourResult] = {}
    for slot, px in garment_pixels(kind, crop, mask, params).items():
        dom = dominant_lab(px, gains)
        if dom is None:
            result[slot] = ColourResult(None, None)
            continue
        lab, share = dom
        term, raw, shade = name_lab(lab, _for_slot(slot, params))
        conf = calibrate(raw * (0.5 + 0.5 * share), params)
        result[slot] = ColourResult(term, conf, f"{shade} {term}" if shade else term, shade)
    return result


def predict(kind: str, crop: np.ndarray, gains: np.ndarray | None = None, mask: np.ndarray | None = None,
            params: ColourParams | None = None) -> dict[str, tuple[str | None, float | None]]:
    """`predict_detail` without the shade, as (term, confidence) pairs; this is what scripts/colour_eval.py scores."""
    return {slot: r.pair() for slot, r in predict_detail(kind, crop, gains, mask, params).items()}
