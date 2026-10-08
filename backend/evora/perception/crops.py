"""Per-track bookkeeping: best-K crops by quality, downsampled trajectory points, class vote.

quality = detector confidence x sharpness x relative size x (1 - truncation)
  sharpness  = v / (v + 100), v = variance of the Laplacian of the grey crop
  size       = min(1, box height / (0.4 x frame height))
  truncation = share of the box area that lies outside the frame
All times here are seconds from the start of the file; the pipeline adds `cam.t0`.
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from evora.perception.settings import IngestSettings
from evora.perception.track import TrackedBox

SHARP_SCALE = 100.0
SIZE_REF = 0.4


@dataclass(order=True)
class Crop:
    quality: float
    t: float = field(compare=False)
    bbox: tuple[float, float, float, float] = field(compare=False)   # normalized 0..1
    image: np.ndarray = field(compare=False, repr=False)             # BGR, longest side <= crop_max_side


@dataclass
class FinishedTrack:
    seq: int
    cls: str
    cls_conf: float
    t_start: float
    t_end: float
    n_obs: int
    points: list[tuple[float, float, float, float, float, float]]    # t, x1, y1, x2, y2, conf (normalized)
    crops: list[Crop]                                                # best first
    quality: float
    direction: str | None


def _unit(x: float) -> float:
    return min(max(x, 0.0), 1.0)


@dataclass
class ActiveTrack:
    seq: int
    cls: str
    cls_conf: float
    t_start: float
    t_end: float
    n_obs: int
    points: list[tuple[float, float, float, float, float, float]]    # t, x1, y1, x2, y2, conf (normalized)


def sharpness(grey: np.ndarray) -> float:
    v = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return v / (v + SHARP_SCALE)


def truncation(xyxy: tuple[float, float, float, float], width: int, height: int) -> float:
    x1, y1, x2, y2 = xyxy
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    cx1, cy1, cx2, cy2 = max(x1, 0.0), max(y1, 0.0), min(x2, float(width)), min(y2, float(height))
    inside = max(cx2 - cx1, 0.0) * max(cy2 - cy1, 0.0)
    return float(min(max(1.0 - inside / area, 0.0), 1.0))


def crop_quality(conf: float, grey_crop: np.ndarray, box_h: float, frame_h: int, trunc: float) -> float:
    size = min(1.0, (box_h / max(frame_h, 1)) / SIZE_REF)
    return float(conf * sharpness(grey_crop) * size * (1.0 - trunc))


def direction_of(points: list[tuple[float, float, float, float, float, float]]) -> str | None:
    """Coarse travel direction from the first and last box centres, or None when the track barely moves."""
    if len(points) < 2:
        return None
    _, ax1, ay1, ax2, ay2, _ = points[0]
    _, bx1, by1, bx2, by2, _ = points[-1]
    dx = (bx1 + bx2) / 2 - (ax1 + ax2) / 2
    dy = (by1 + by2) / 2 - (ay1 + ay2) / 2
    if max(abs(dx), abs(dy)) < 0.08:
        return None
    if abs(dx) >= abs(dy):
        return "left_to_right" if dx > 0 else "right_to_left"
    return "towards_camera" if dy > 0 else "away_from_camera"


@dataclass
class _State:
    seq: int
    t_start: float
    t_end: float
    last_point_t: float = float("-inf")
    n_obs: int = 0
    votes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    conf_sum: float = 0.0
    points: list[tuple[float, float, float, float, float, float]] = field(default_factory=list)
    heap: list[Crop] = field(default_factory=list)


class TrackBook:
    """Accumulates observations per tracker id and hands back finished tracks."""

    def __init__(self, cfg: IngestSettings, first_seq: int = 1):
        self.cfg = cfg
        self._states: dict[int, _State] = {}
        self._next_seq = first_seq

    def observe(self, pts_s: float, boxes: list[TrackedBox], frame: np.ndarray) -> None:
        cfg = self.cfg
        fh, fw = frame.shape[:2]
        for b in boxes:
            st = self._states.get(b.track_key)
            if st is None:
                st = self._states[b.track_key] = _State(seq=self._next_seq, t_start=pts_s, t_end=pts_s)
                self._next_seq += 1
            st.t_end = pts_s
            st.n_obs += 1
            st.votes[b.cls] += b.conf
            st.conf_sum += b.conf
            x1, y1, x2, y2 = b.xyxy
            nx = (_unit(x1 / fw), _unit(y1 / fh), _unit(x2 / fw), _unit(y2 / fh))
            if pts_s - st.last_point_t >= 1.0 / cfg.point_hz:
                st.points.append((pts_s, *nx, b.conf))
                st.last_point_t = pts_s
            self._consider_crop(st, pts_s, b, frame, nx)

    def _consider_crop(self, st: _State, pts_s: float, b: TrackedBox, frame: np.ndarray, nx: tuple[float, ...]) -> None:
        cfg = self.cfg
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = b.xyxy
        pad_x, pad_y = (x2 - x1) * cfg.crop_pad, (y2 - y1) * cfg.crop_pad
        ix1, iy1 = int(max(x1 - pad_x, 0)), int(max(y1 - pad_y, 0))
        ix2, iy2 = int(min(x2 + pad_x, fw)), int(min(y2 + pad_y, fh))
        if ix2 - ix1 < 4 or iy2 - iy1 < 4:
            return
        crop = frame[iy1:iy2, ix1:ix2]
        side = max(crop.shape[:2])
        if side > cfg.crop_max_side:
            scale = cfg.crop_max_side / side
            new_size = (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale)))
            crop = cv2.resize(crop, new_size, interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        q = crop_quality(b.conf, grey, y2 - y1, fh, truncation(b.xyxy, fw, fh))
        item = Crop(quality=q, t=pts_s, bbox=(nx[0], nx[1], nx[2], nx[3]), image=crop)
        if len(st.heap) < cfg.crop_k:
            heapq.heappush(st.heap, item)
        elif q > st.heap[0].quality:
            heapq.heapreplace(st.heap, item)

    def _finish(self, st: _State) -> FinishedTrack | None:
        if st.n_obs < self.cfg.min_track_obs or not st.heap:
            return None
        cls = max(st.votes.items(), key=lambda kv: kv[1])[0]
        crops = sorted(st.heap, key=lambda c: c.quality, reverse=True)
        return FinishedTrack(
            seq=st.seq, cls=cls, cls_conf=st.conf_sum / st.n_obs, t_start=st.t_start, t_end=st.t_end,
            n_obs=st.n_obs, points=st.points, crops=crops,
            quality=float(np.mean([c.quality for c in crops])), direction=direction_of(st.points),
        )

    def active(self) -> list[ActiveTrack]:
        """Tracks still being followed, for callers that need results before a track ends (live ingest)."""
        out = []
        for st in self._states.values():
            cls = max(st.votes.items(), key=lambda kv: kv[1])[0]
            out.append(ActiveTrack(st.seq, cls, st.conf_sum / max(st.n_obs, 1), st.t_start, st.t_end, st.n_obs, list(st.points)))
        return out

    def finalize_stale(self, now_s: float) -> list[FinishedTrack]:
        """Tracks unseen for `track_lost_s` are closed so memory stays bounded on long files."""
        done: list[FinishedTrack] = []
        for key in [k for k, s in self._states.items() if now_s - s.t_end > self.cfg.track_lost_s]:
            fin = self._finish(self._states.pop(key))
            if fin is not None:
                done.append(fin)
        return done

    def finalize_all(self) -> list[FinishedTrack]:
        done = [f for st in self._states.values() if (f := self._finish(st)) is not None]
        self._states.clear()
        return done


def save_jpeg(image: np.ndarray, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"could not write {path}")
