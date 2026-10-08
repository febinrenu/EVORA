"""B0: frame retrieval baseline (PLAN section 4).

Sample frames at about 1 fps, embed whole frames, rank by cosine similarity to the query
text, and merge adjacent hits into windows. It gets the same parsed time window and
camera filter as our system (fairness rule, section 9.4); only retrieval differs.

The embedder is injected so this module has no model dependency; production wires in
the same SigLIP2 model the main index uses.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, tzinfo
from typing import Any, Protocol

import numpy as np
from contracts.models import Evidence, QueryPlan

from evora.query.logic import instant_in_window

MERGE_GAP_S = 2.0
DEFAULT_TOP_FRAMES = 200


class Embedder(Protocol):
    def embed_text(self, text: str) -> np.ndarray: ...
    def embed_images(self, images: Sequence[Any]) -> np.ndarray: ...


class CameraLike(Protocol):
    id: str
    name: str
    t0: float


@dataclass(frozen=True)
class Window:
    camera_id: str
    t_start: float
    t_end: float
    t_peak: float
    score: float


def _unit(vectors: np.ndarray) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


class FrameIndex:
    """Whole-frame embeddings per camera, kept in memory."""

    def __init__(self) -> None:
        self._cam: list[str] = []
        self._t: list[float] = []
        self._chunks: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._t)

    def add(self, camera_id: str, times: Sequence[float], vectors: np.ndarray) -> None:
        vecs = _unit(vectors)
        if len(times) != len(vecs):
            raise ValueError("one vector per timestamp")
        self._cam.extend([camera_id] * len(times))
        self._t.extend(float(t) for t in times)
        self._chunks.append(vecs)
        self._matrix = None

    def add_frames(self, camera_id: str, frames: Iterable[tuple[float, Any]], embedder: Embedder,
                   batch: int = 32) -> int:
        """Embed (timestamp, image) pairs in batches and index them. Returns the frame count."""
        total = 0
        buf_t: list[float] = []
        buf_img: list[Any] = []

        def flush() -> None:
            nonlocal total
            if buf_t:
                self.add(camera_id, buf_t, embedder.embed_images(buf_img))
                total += len(buf_t)
                buf_t.clear()
                buf_img.clear()

        for t, image in frames:
            buf_t.append(t)
            buf_img.append(image)
            if len(buf_t) >= batch:
                flush()
        flush()
        return total

    def _all(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.vstack(self._chunks) if self._chunks else np.zeros((0, 1), dtype=np.float32)
        return self._matrix

    def search(self, text_vec: np.ndarray, k: int = DEFAULT_TOP_FRAMES, camera_ids: Sequence[str] = (),
               plan: QueryPlan | None = None, tz: tzinfo = UTC) -> list[tuple[str, float, float]]:
        """Top-k (camera_id, t, cosine) after the camera filter and the plan's time window."""
        if not self._t:
            return []
        matrix = self._all()
        scores = matrix @ _unit(text_vec)[0]
        window = plan.time if plan else None
        allowed = set(camera_ids)
        keep = [i for i in range(len(self._t))
                if (not allowed or self._cam[i] in allowed) and instant_in_window(self._t[i], window, tz)]
        keep.sort(key=lambda i: -scores[i])
        return [(self._cam[i], self._t[i], float(scores[i])) for i in keep[:k]]


def merge_hits(hits: Iterable[tuple[str, float, float]], gap_s: float = MERGE_GAP_S) -> list[Window]:
    """Merge hits of one camera that are within `gap_s` of each other into windows."""
    by_cam: dict[str, list[tuple[float, float]]] = {}
    for cam, t, score in hits:
        by_cam.setdefault(cam, []).append((t, score))
    windows: list[Window] = []
    for cam, items in by_cam.items():
        items.sort()
        start = prev = items[0][0]
        peak_t, peak_s = items[0]
        for t, s in items[1:]:
            if t - prev > gap_s:
                windows.append(Window(cam, start, prev, peak_t, peak_s))
                start, peak_t, peak_s = t, t, s
            elif s > peak_s:
                peak_t, peak_s = t, s
            prev = t
        windows.append(Window(cam, start, prev, peak_t, peak_s))
    return sorted(windows, key=lambda w: -w.score)


def b0_search(index: FrameIndex, embedder: Embedder, plan: QueryPlan, tz: tzinfo = UTC,
              top_frames: int = DEFAULT_TOP_FRAMES, limit: int | None = None) -> list[Window]:
    """Rank windows for a plan the baseline way: text -> frame cosine -> merged windows."""
    text = plan.targets[0].embed_text if plan.targets else ""
    if not text:
        return []
    hits = index.search(embedder.embed_text(text), top_frames, plan.camera_ids, plan, tz)
    windows = merge_hits(hits)
    return windows[: limit or plan.limit]


def windows_to_evidence(windows: Sequence[Window], cameras: Sequence[CameraLike]) -> list[Evidence]:
    by_id = {c.id: c for c in cameras}
    out: list[Evidence] = []
    for i, w in enumerate(windows, start=1):
        cam = by_id[w.camera_id]
        eid = f"b0_{i:03d}"
        out.append(Evidence(
            id=eid, camera_id=w.camera_id, camera_name=cam.name,
            t_start=w.t_start, t_end=w.t_end, t_peak=w.t_peak, offset_s=w.t_peak - cam.t0,
            thumb_url=f"/api/media/thumb/{eid}.jpg", clip_url=f"/api/media/clip/{eid}.mp4",
            score=w.score, why=[f"frame cosine {w.score:.2f}"],
        ))
    return out
