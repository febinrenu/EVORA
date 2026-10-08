"""Ingestion pipeline: layered, resumable by layer, one pass per layer.

L0  scene embeddings (full frame + 2x2 tiles every `scene_every_s`)  -> LanceDB `scenes`
L1  detect + track + best-K crops + crop embeddings                    -> SQLite `tracks`, `track_points`, LanceDB `crops`
L2  attributes (colour, vehicle type, carrying, infrared) and events from the stored tracks; ReID is added later
L3  captions (not implemented yet; skipped with a warning)

Paths stored in the database are relative to the workspace `media/` directory.
Times are epoch seconds UTC: `cam.t0 + seconds into the file`.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from contracts.models import CameraInfo, IngestJob

from evora.core.config import load_config
from evora.core.db import Database, open_db
from evora.core.vectors import dims_from_meta, ensure_tables, open_store
from evora.core.workspace import Workspace
from evora.core.workspace import create as create_workspace
from evora.perception.crops import FinishedTrack, TrackBook, save_jpeg
from evora.perception.decode import DecodeError, probe_video, read_frames
from evora.perception.detect import load_detector
from evora.perception.embed import SigLIP2Embedder, get_embedder
from evora.perception.l2 import run_l2
from evora.perception.locks import STORE_SETUP
from evora.perception.motion import AdaptiveSampler
from evora.perception.settings import IngestSettings, load_settings
from evora.perception.track import FrameTracker

log = logging.getLogger("evora.perception.pipeline")

ProgressFn = Callable[[IngestJob], None]
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")
_IMPLEMENTED = ("L0", "L1", "L2")


def resolve_workspace(slug: str | None = None) -> Workspace:
    """Same rule the API uses: explicit slug, else env evora_WORKSPACE, else the configured default."""
    import os

    cfg = load_config()
    return create_workspace(slug or os.environ.get("evora_WORKSPACE") or cfg["workspace"]["default"])


class _Reporter:
    """Throttled IngestJob callbacks with a running video-seconds-per-second rate."""

    def __init__(self, cam: CameraInfo, layer: str, duration_s: float | None, on_progress: ProgressFn):
        self.cam, self.layer, self.duration, self.cb = cam, layer, duration_s, on_progress
        self.started = time.monotonic()
        self._last = 0.0

    def tick(self, pts_s: float, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < 0.5:
            return
        self._last = now
        frac = min(pts_s / self.duration, 0.99) if self.duration else 0.0
        self._emit(frac, pts_s)

    def done(self, pts_s: float) -> None:
        self._emit(1.0, pts_s)

    def _emit(self, frac: float, pts_s: float) -> None:
        wall = max(time.monotonic() - self.started, 1e-6)
        self.cb(IngestJob(
            id="", camera_id=self.cam.id, state="running", layer=self.layer,  # type: ignore[arg-type]
            progress=frac, video_s_per_s=pts_s / wall,
        ))


class _RowBuffer:
    """Collects vector rows and writes them in few large commits: every LanceDB add creates a new table version,
    and many small adds from several cameras at once contend on the version files."""

    def __init__(self, table, limit: int = 4000):
        self.table, self.limit, self.rows = table, limit, []

    def add(self, rows: list[dict]) -> None:
        self.rows += rows
        if len(self.rows) >= self.limit:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            self.table.add(self.rows)
            self.rows = []


def _check_id(cam_id: str) -> str:
    if not _SAFE_ID.match(cam_id):
        raise ValueError(f"unsafe camera id: {cam_id!r}")
    return cam_id


def _rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _tiles(img: np.ndarray) -> dict[str, np.ndarray]:
    h, w = img.shape[:2]
    hh, hw = h // 2, w // 2
    return {
        "full": img,
        "tl": img[:hh, :hw], "tr": img[:hh, hw:], "bl": img[hh:, :hw], "br": img[hh:, hw:],
    }


def _scene_image(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(round(h * width / w)) // 2 * 2), interpolation=cv2.INTER_AREA)


def _run_l0(cam: CameraInfo, path: Path, ws: Workspace, store, st: IngestSettings, embedder: SigLIP2Embedder,
            duration: float | None, on_progress: ProgressFn) -> int:
    cid = _check_id(cam.id)
    table = store.open_table("scenes")
    table.delete(f"camera_id = '{cid}'")
    buffer = _RowBuffer(table)
    shutil.rmtree(ws.media_dir / "scenes" / cid, ignore_errors=True)
    rep = _Reporter(cam, "L0", duration, on_progress)
    pending: list[tuple[float, str, dict[str, np.ndarray]]] = []
    written = 0
    next_t, last_pts = 0.0, 0.0

    def flush() -> int:
        if not pending:
            return 0
        imgs: list[np.ndarray] = []
        meta: list[tuple[float, str, str]] = []
        for t, frame_path, tiles in pending:
            for name, tile in tiles.items():
                imgs.append(tile)
                meta.append((t, name, frame_path))
        vecs = embedder.embed_images(imgs)
        buffer.add([
            {"vector": v.tolist(), "camera_id": cid, "t": cam.t0 + t, "tile": name, "frame_path": fp}
            for v, (t, name, fp) in zip(vecs, meta, strict=True)
        ])
        n = len(meta)
        pending.clear()
        return n

    for frame in read_frames(path, st.max_width):
        last_pts = frame.pts_s
        if frame.pts_s + 1e-9 < next_t:
            continue
        next_t = (int(frame.pts_s / st.scene_every_s) + 1) * st.scene_every_s
        out = ws.media_dir / "scenes" / cid / f"{int(round(frame.pts_s * 1000)):09d}.jpg"
        save_jpeg(_scene_image(frame.image, st.scene_jpeg_width), out, st.crop_jpeg_quality)
        pending.append((frame.pts_s, _rel(out, ws.media_dir), _tiles(frame.image)))
        if len(pending) >= 8:
            written += flush()
        rep.tick(frame.pts_s)
    written += flush()
    buffer.flush()
    rep.done(last_pts)
    return written


def _delete_l1(db: Database, store, ws: Workspace, cid: str) -> None:
    with db.write() as c:
        c.execute("DELETE FROM track_points WHERE track_id IN (SELECT id FROM tracks WHERE camera_id=?)", (cid,))
        c.execute("DELETE FROM tracks WHERE camera_id=?", (cid,))
    store.open_table("crops").delete(f"camera_id = '{cid}'")
    shutil.rmtree(ws.media_dir / "crops" / cid, ignore_errors=True)


def _persist(db: Database, buffer: _RowBuffer, ws: Workspace, cam: CameraInfo, tracks: list[FinishedTrack],
             st: IngestSettings, embedder: SigLIP2Embedder) -> int:
    if not tracks:
        return 0
    cid = _check_id(cam.id)
    flat = [(t, i, crop) for t in tracks for i, crop in enumerate(t.crops)]
    vecs = embedder.embed_images([c.image for _, _, c in flat])
    crop_rows, paths = [], {}
    for v, (trk, i, crop) in zip(vecs, flat, strict=True):
        tid = f"{cid}:t{trk.seq:06d}"
        out = ws.media_dir / "crops" / cid / f"t{trk.seq:06d}_{i}.jpg"
        save_jpeg(crop.image, out, st.crop_jpeg_quality)
        rel = _rel(out, ws.media_dir)
        paths[(trk.seq, i)] = rel
        crop_rows.append({"vector": v.tolist(), "track_id": tid, "camera_id": cid, "cls": trk.cls,
                          "t": cam.t0 + crop.t, "quality": float(crop.quality), "crop_path": rel})
    with db.write() as c:
        for trk in tracks:
            tid = f"{cid}:t{trk.seq:06d}"
            best = trk.crops[0]
            c.execute(
                "INSERT INTO tracks(id,camera_id,cls,cls_conf,t_start,t_end,n_obs,best_crop,best_t,best_bbox,"
                "attrs,direction,global_id,quality)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tid, cid, trk.cls, trk.cls_conf, cam.t0 + trk.t_start, cam.t0 + trk.t_end, trk.n_obs,
                 paths[(trk.seq, 0)], cam.t0 + best.t, json.dumps([round(x, 4) for x in best.bbox]), "{}",
                 trk.direction, None, trk.quality),
            )
            c.executemany(
                "INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,?,?,?,?,?)",
                [(tid, cam.t0 + p[0], p[1], p[2], p[3], p[4], p[5]) for p in trk.points],
            )
    buffer.add(crop_rows)
    return len(tracks)


def _run_l1(cam: CameraInfo, path: Path, ws: Workspace, db: Database, store, st: IngestSettings,
            embedder: SigLIP2Embedder, duration: float | None, on_progress: ProgressFn) -> tuple[int, int]:
    cid = _check_id(cam.id)
    _delete_l1(db, store, ws, cid)
    buffer = _RowBuffer(store.open_table("crops"))
    det = load_detector(st)
    tracker, book, sampler = FrameTracker(det, st), TrackBook(st), AdaptiveSampler(st)
    rep = _Reporter(cam, "L1", duration, on_progress)
    n_tracks = n_frames = 0
    last_pts = 0.0
    for frame in read_frames(path, st.max_width):
        last_pts = frame.pts_s
        if not sampler.should_process(frame.pts_s, frame):
            continue
        n_frames += 1
        book.observe(frame.pts_s, tracker.update(frame.image), frame.image)
        n_tracks += _persist(db, buffer, ws, cam, book.finalize_stale(frame.pts_s), st, embedder)
        rep.tick(frame.pts_s)
    n_tracks += _persist(db, buffer, ws, cam, book.finalize_all(), st, embedder)
    buffer.flush()
    rep.done(last_pts)
    return n_tracks, n_frames


def ingest(
    cam: CameraInfo, profile: str, layers: set[str], on_progress: ProgressFn, *,
    ws: Workspace | None = None, settings: IngestSettings | None = None,
) -> None:
    """Entry point called by the job runner (PLAN.md section 5.6).

    Finished layers are reported with `progress == 1.0`. Layers that are not implemented yet are
    skipped with a warning and never reported as finished.
    """
    if cam.kind != "file":
        raise NotImplementedError("live ingest is handled by live_ingest (P2.17)")
    cfg = load_config(profile)
    st = settings or load_settings(cfg)
    ws = ws or resolve_workspace()
    path = Path(cam.source_uri)
    if not path.is_file():
        raise DecodeError(f"source file missing: {path}")
    skipped = sorted(set(layers) - set(_IMPLEMENTED))
    if skipped:
        log.warning("layers %s are not implemented yet; skipping for %s", skipped, cam.id)
    todo = [ly for ly in _IMPLEMENTED if ly in layers]
    if not todo:
        return
    duration = cam.duration_s or probe_video(path).duration_s
    db = open_db(ws.db_path)
    embedder = get_embedder(st)
    store = open_store(ws.vectors_dir)
    with STORE_SETUP:
        db.set_meta("embed_dim_image", str(embedder.dim))
        db.set_meta("embed_model_image", st.image_model)
        ensure_tables(store, dims_from_meta(db), only={"crops", "scenes"})
    started = time.monotonic()
    if "L0" in todo:
        n = _run_l0(cam, path, ws, store, st, embedder, duration, on_progress)
        log.info("%s L0: %d scene embeddings", cam.id, n)
    if "L1" in todo:
        n_tracks, n_frames = _run_l1(cam, path, ws, db, store, st, embedder, duration, on_progress)
        log.info("%s L1: %d tracks from %d sampled frames", cam.id, n_tracks, n_frames)
    if "L2" in todo:
        stats = run_l2(cam, ws, db, store, st, embedder, on_progress)
        log.info("%s L2: %s", cam.id, stats)
    log.info("%s ingest took %.1fs for %.1fs of video", cam.id, time.monotonic() - started, duration or 0.0)
