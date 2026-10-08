"""Objects the index has no tracks for: chairs, carpets, "red objects".

The tracker follows a short list of classes (people, vehicles, bags). A question about anything else used to be
dropped, and "red objects" counted tracks that were never there. Here such a question is answered by looking at the
frames the workspace already stores: an open-vocabulary detector (YOLOE) is asked for the noun in a sample of frames
of the camera and time window, each detection's colour is read from the middle of its box, and the answer is what
is in view in those frames.

Counts are what is in view in one frame (the median over the sampled frames, with the range), not a total over time:
a chair is the same chair in every frame. Detections are cached per frame and noun list, so asking about colour
after asking about the object costs nothing.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from contracts.models import Target

from evora.query import fastpath

log = logging.getLogger("evora.query.objects")

INDEXED_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase", "umbrella"}
GENERIC_WORDS = {"object", "objects", "thing", "things", "item", "items", "stuff"}
# what "red objects" is searched among: everyday things in a room or on a street, not clothes and not people
COMMON_OBJECTS = (
    "chair", "table", "desk", "sofa", "bench", "bed", "shelf", "cabinet", "bag", "box", "bottle", "cup", "laptop", "phone",
    "book", "plant", "lamp", "tv", "monitor", "carpet", "pillow", "clock", "basket", "bin", "fan", "ball", "toy",
)
SYNONYMS: dict[str, tuple[str, ...]] = {
    "carpet": ("carpet", "rug"), "rug": ("rug", "carpet"), "sofa": ("sofa", "couch"), "couch": ("couch", "sofa"),
    "tv": ("tv", "television"), "television": ("television", "tv"), "table": ("table", "desk"), "desk": ("desk", "table"),
    "bin": ("bin", "trash can"), "plant": ("plant", "potted plant"), "monitor": ("monitor", "screen"),
    "screen": ("screen", "monitor"), "bag": ("bag", "handbag"), "cushion": ("cushion", "pillow"),
}
MIN_CONF = 0.3          # detections below this are mostly clutter at the sizes these cameras give
MIN_AREA = 0.0015       # share of the frame; smaller boxes are noise
DUPLICATE_IOU = 0.6     # two labels on the same thing count once
COLOUR_SHARE = 0.25     # a box is "red" when at least this share of its middle is red
SAMPLE_FRAMES = 12      # per camera
CACHE_LIMIT = 20000


def singular(word: str) -> str:
    w = word.strip().lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ves") and len(w) > 4:
        return w[:-3] + "f"
    if w.endswith(("ches", "shes", "xes", "sses")) and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def labels_for(target: Target) -> list[str] | None:
    """The noun(s) to look for when this target is not something the tracker follows; None when it is."""
    noun = singular(target.noun)
    if not noun:
        return None
    if set(target.cls) & INDEXED_CLASSES:
        return None
    tracked = INDEXED_CLASSES | fastpath.PERSON_GENERIC | fastpath.PERSON_SPECIFIC | set(fastpath.VEHICLES) | set(fastpath.CARRY)
    if noun in tracked or target.noun.strip().lower() in tracked:
        return None
    if noun in GENERIC_WORDS:
        return list(COMMON_OBJECTS)
    return list(SYNONYMS.get(noun, (noun,)))


@dataclass(frozen=True)
class Detection:
    label: str
    conf: float
    box: tuple[float, float, float, float]          # normalised x1, y1, x2, y2
    colours: dict[str, float] = field(default_factory=dict, hash=False, compare=False)   # colour term -> share of the box


@dataclass(frozen=True)
class FrameRef:
    camera_id: str
    t: float
    path: str                                        # relative to the workspace's media folder


@dataclass
class FrameResult:
    ref: FrameRef
    detections: list[Detection]


@dataclass
class Survey:
    camera_id: str
    labels: list[str]
    colour: str | None
    frames: list[FrameResult]

    def matching(self, frame: FrameResult) -> list[Detection]:
        if self.colour is None:
            return frame.detections
        return [d for d in frame.detections if d.colours.get(self.colour, 0.0) >= COLOUR_SHARE]

    def counts(self) -> list[int]:
        return [len(self.matching(f)) for f in self.frames]

    @property
    def typical(self) -> int:
        counts = self.counts()
        return int(round(statistics.median(counts))) if counts else 0

    @property
    def peak(self) -> int:
        return max(self.counts(), default=0)

    @property
    def least(self) -> int:
        return min(self.counts(), default=0)

    @property
    def seen_in(self) -> int:
        return sum(1 for c in self.counts() if c > 0)

    def best_frame(self) -> FrameResult | None:
        scored = [(len(self.matching(f)), -i, f) for i, f in enumerate(self.frames)]
        scored = [s for s in scored if s[0] > 0]
        return max(scored, key=lambda s: (s[0], s[1]))[2] if scored else None

    def breakdown(self, frame: FrameResult) -> Counter[str]:
        return Counter(d.label for d in self.matching(frame))


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_duplicates(boxes: Sequence[Detection], iou: float = DUPLICATE_IOU) -> list[Detection]:
    """Keep the most confident of any boxes that cover the same thing (a chair is also a 'seat')."""
    kept: list[Detection] = []
    for d in sorted(boxes, key=lambda d: -d.conf):
        if all(_iou(d.box, k.box) < iou for k in kept):
            kept.append(d)
    return kept


def box_colours(image_bgr: np.ndarray, box: tuple[float, float, float, float], k: int = 3) -> dict[str, float]:
    """Colour terms present in the middle of a box with their share of its pixels (a person in front shifts it)."""
    import cv2

    from evora.perception import colourmodel as cm

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1) * w, (y2 - y1) * h
    cx1, cx2 = int(x1 * w + 0.2 * bw), int(x2 * w - 0.2 * bw)
    cy1, cy2 = int(y1 * h + 0.2 * bh), int(y2 * h - 0.2 * bh)
    crop = image_bgr[max(cy1, 0):max(cy2, 0), max(cx1, 0):max(cx2, 0)]
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return {}
    small = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    kk = min(k, max(1, len(np.unique(lab, axis=0))))
    if kk == 1:
        centres, counts = lab.mean(axis=0, keepdims=True), np.array([len(lab)])
    else:
        cv2.setRNGSeed(0)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, centres = cv2.kmeans(lab, kk, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=kk)
    params = cm.load_params()
    out: dict[str, float] = {}
    for centre, count in zip(centres, counts, strict=True):
        term = cm.name_lab(centre, params)[0]
        out[term] = out.get(term, 0.0) + float(count) / len(lab)
    return out


DetectFn = Callable[[np.ndarray, Sequence[str], float], list[Any]]


class OpenObjectSurveyor:
    """Runs the detector over sampled frames and keeps what it saw (the detector is loaded on first use)."""

    def __init__(self, media_dir: Path, detect: DetectFn | None = None, conf: float = MIN_CONF) -> None:
        self._media = Path(media_dir)
        self._detect = detect
        self._conf = conf
        self._cache: dict[tuple[str, tuple[str, ...], float], list[Detection]] = {}
        self._lock = threading.Lock()
        self.unavailable: str | None = None

    def _detector(self) -> DetectFn | None:
        if self._detect is not None:
            return self._detect
        if self.unavailable is not None:
            return None
        try:
            from evora.perception.openvocab import OpenVocabDetector, OpenVocabUnavailable

            try:
                detector = OpenVocabDetector()
            except OpenVocabUnavailable as exc:
                self.unavailable = str(exc)
                log.warning("open-vocabulary detection unavailable: %s", exc)
                return None
        except ImportError as exc:  # perception stack not installed
            self.unavailable = f"the perception stack is not installed ({exc})"
            return None
        self._detect = lambda image, labels, conf: detector.detect(image, labels, conf=conf)
        return self._detect

    def ready(self) -> bool:
        return self._detector() is not None

    def survey_sync(self, camera_id: str, frames: Sequence[FrameRef], labels: Sequence[str], colour: str | None) -> Survey:
        import cv2

        detect = self._detector()
        results: list[FrameResult] = []
        for ref in frames:
            key = (ref.path, tuple(labels), self._conf)
            with self._lock:
                cached = self._cache.get(key)
            if cached is None:
                image = cv2.imread(str(self._media / ref.path))
                if image is None or detect is None:
                    continue
                found = [
                    Detection(b.label, float(b.conf), tuple(b.xyxy), box_colours(image, tuple(b.xyxy)))
                    for b in detect(image, labels, self._conf)
                    if (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]) >= MIN_AREA
                ]
                cached = merge_duplicates(found)
                with self._lock:
                    if len(self._cache) >= CACHE_LIMIT:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[key] = cached
            results.append(FrameResult(ref, cached))
        return Survey(camera_id, list(labels), colour, results)

    async def survey(self, camera_id: str, frames: Sequence[FrameRef], labels: Sequence[str], colour: str | None) -> Survey:
        return await asyncio.to_thread(self.survey_sync, camera_id, list(frames), list(labels), colour)


def sample_evenly(items: Sequence[Any], n: int) -> list[Any]:
    """At most n items spread over the whole sequence, first and last included."""
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]
