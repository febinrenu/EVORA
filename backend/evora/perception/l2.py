"""Layer L2: track attributes and events, computed from what L0 and L1 stored.

Reads the crops and track points of one camera, writes `tracks.attrs` (TrackAttrs JSON), the camera's
`ir_fraction`, and the events for appear / disappear and every stored zone. No video is decoded.
ReID features are a later part of this layer (P2.14).
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from contracts.models import CameraInfo, IngestJob, TrackAttrs

from evora.core.db import Database
from evora.core.workspace import Workspace
from evora.perception import attributes as at
from evora.perception.events import recompute_events, zones_of
from evora.perception.settings import IngestSettings

log = logging.getLogger("evora.perception.l2")

ProgressFn = Callable[[IngestJob], None]
VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}
# which vehicle types a detector class may be refined to by the zero-shot step
CLASS_TYPES = {
    "car": ("car", "suv", "van", "auto_rickshaw"), "truck": ("truck", "van"), "bus": ("bus",),
    "motorcycle": ("motorcycle",), "bicycle": ("bicycle",),
}
MAX_SCENE_SAMPLES = 120


def _scene_samples(ws: Workspace, cam: CameraInfo) -> list[tuple[float, np.ndarray]]:
    """(epoch time, BGR frame) for evenly spread L0 scene frames, used for white balance and IR detection."""
    files = sorted((ws.media_dir / "scenes" / cam.id).glob("*.jpg"))
    if len(files) > MAX_SCENE_SAMPLES:
        files = [files[int(i)] for i in np.linspace(0, len(files) - 1, MAX_SCENE_SAMPLES)]
    out = []
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            out.append((cam.t0 + int(f.stem) / 1000.0, img))
    return out


def _nearest(times: list[float], t: float) -> int:
    return int(np.argmin(np.abs(np.asarray(times) - t)))


def _load_boxes(db: Database, camera_id: str) -> dict[str, list[at.Box]]:
    with db.read() as c:
        rows = c.execute(
            "SELECT p.track_id, p.t, p.x1, p.y1, p.x2, p.y2 FROM track_points p JOIN tracks t ON t.id=p.track_id "
            "WHERE t.camera_id=? ORDER BY p.track_id, p.t", (camera_id,)).fetchall()
    out: dict[str, list[at.Box]] = defaultdict(list)
    for r in rows:
        out[r["track_id"]].append(at.Box(r["t"], r["x1"], r["y1"], r["x2"], r["y2"]))
    return out


def _bag_owners(persons: dict[str, list[at.Box]], bags: dict[str, tuple[str, list[at.Box]]]) -> dict[str, list[tuple[str, bool]]]:
    """person track id -> [(bag class, large)] where each bag goes to the closest person that carries it."""
    owned: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for _, (cls, bag_boxes) in bags.items():
        best: tuple[float, str, bool] | None = None
        for pid, p_boxes in persons.items():
            if p_boxes[-1].t < bag_boxes[0].t - 1.0 or p_boxes[0].t > bag_boxes[-1].t + 1.0:
                continue
            carried, large = at.carried_by(p_boxes, bag_boxes)
            if not carried:
                continue
            dists = []
            for b in bag_boxes:
                p = min(p_boxes, key=lambda q: abs(q.t - b.t))
                dists.append(abs(p.cx - b.cx) + abs(p.cy - b.cy))
            score = float(np.mean(dists))
            if best is None or score < best[0]:
                best = (score, pid, large)
        if best is not None:
            owned[best[1]].append((cls, best[2]))
    return owned


def run_l2(cam: CameraInfo, ws: Workspace, db: Database, store, st: IngestSettings, embedder,
           on_progress: ProgressFn) -> dict[str, int]:
    """Compute attributes and events for every track of `cam`; returns small counters for logging."""
    started = time.monotonic()
    cid = cam.id

    def report(frac: float) -> None:
        on_progress(IngestJob(id="", camera_id=cid, state="running", layer="L2", progress=frac,  # type: ignore[arg-type]
                              video_s_per_s=None))

    report(0.0)
    scenes = _scene_samples(ws, cam)
    scene_t = [t for t, _ in scenes]
    ir_flags = [at.is_ir_frame(img) for _, img in scenes]
    colour_frames = [img for (_, img), ir in zip(scenes, ir_flags, strict=True) if not ir] or [img for _, img in scenes]
    gains = at.white_balance_gains(colour_frames)
    ir_fraction = float(np.mean(ir_flags)) if ir_flags else None
    with db.write() as c:
        c.execute("UPDATE cameras SET ir_fraction=? WHERE id=?", (ir_fraction, cid))
    report(0.1)

    with db.read() as c:
        tracks = c.execute("SELECT id, cls, best_t FROM tracks WHERE camera_id=? ORDER BY id", (cid,)).fetchall()
    crop_rows = store.open_table("crops").search().where(f"camera_id = '{cid}'").limit(10_000_000).to_arrow().to_pylist()
    crops_by_track: dict[str, list[dict]] = defaultdict(list)
    for r in crop_rows:
        crops_by_track[r["track_id"]].append(r)
    boxes = _load_boxes(db, cid)
    type_vecs = embedder.embed_texts([at.VEHICLE_PROMPTS[t] for t in at.VEHICLE_TYPES])

    persons = {t["id"]: boxes[t["id"]] for t in tracks if t["cls"] == "person" and boxes.get(t["id"])}
    bags = {t["id"]: (t["cls"], boxes[t["id"]]) for t in tracks if t["cls"] in at.BAG_CLASSES and boxes.get(t["id"])}
    carrying = _bag_owners(persons, bags)

    updates: list[tuple[str, str]] = []
    for i, t in enumerate(tracks):
        tid, cls = t["id"], t["cls"]
        attrs = TrackAttrs()
        pts = boxes.get(tid, [])
        if pts:
            attrs.size_rel = round(statistics.median(p.h for p in pts), 4)
        if scene_t and t["best_t"] is not None:
            attrs.is_ir = bool(ir_flags[_nearest(scene_t, t["best_t"])])
        images = []
        for r in sorted(crops_by_track.get(tid, []), key=lambda r: -r["quality"]):
            img = cv2.imread(str(Path(ws.media_dir) / r["crop_path"]))
            if img is not None:
                images.append((img, np.asarray(r["vector"], dtype=np.float32)))
        if images and not attrs.is_ir:
            if cls == "person":
                up = at.aggregate_colour([at.dominant_colour(at.person_regions(im)[0], gains) for im, _ in images])
                lo = at.aggregate_colour([at.dominant_colour(at.person_regions(im)[1], gains) for im, _ in images])
                attrs.upper_color, attrs.lower_color = up[0], lo[0]
                attrs.color, attrs.color_conf = up[0], up[1]
            elif cls in VEHICLES:
                col = at.aggregate_colour([at.dominant_colour(at.vehicle_region(im), gains) for im, _ in images])
                attrs.color, attrs.color_conf = col
        if cls in VEHICLES and images:
            allowed = CLASS_TYPES[cls]
            if len(allowed) == 1:
                attrs.vehicle_type = allowed[0]
            else:
                idx = [at.VEHICLE_TYPES.index(a) for a in allowed]
                guess = at.classify_vehicle(np.stack([v for _, v in images]), type_vecs[idx], allowed)
                attrs.vehicle_type = guess[0] if guess else allowed[0]
        if cls == "person":
            attrs.carrying = at.carrying_terms(carrying.get(tid, []))
        updates.append((json.dumps(attrs.model_dump(mode="json", exclude_none=True)), tid))
        if i % 25 == 0:
            report(0.1 + 0.75 * i / max(len(tracks), 1))

    with db.write() as c:
        c.executemany("UPDATE tracks SET attrs=? WHERE id=?", [(a, tid) for a, tid in updates])
    report(0.9)
    n_events = recompute_events(cid, zones_of(db, cid), db=db, settings=st)
    report(1.0)
    stats = {"tracks": len(tracks), "events": n_events, "ir_scenes": int(sum(ir_flags)),
             "carrying": sum(1 for v in carrying.values() if v)}
    log.info("%s L2 in %.1fs: %s", cid, time.monotonic() - started, stats)
    return stats
