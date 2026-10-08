"""Appearance (ReID) features per track, from the best crops kept at ingest.

OSNet (BoxMOT, `osnet_x0_25_msmt17`) turns each crop into a 512-d vector. A track's feature is the mean of
its crops' vectors, L2-normalised, stored in the LanceDB `reid` table. The same network is used for
vehicles in this version; colour and type attributes carry most of the weight when matching them.
Weights come from `models/boxmot/` (`scripts/models_download.py --only boxmot`).
"""
from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from evora.core.config import REPO_ROOT
from evora.core.db import Database
from evora.core.vectors import dims_from_meta, ensure_tables
from evora.core.workspace import Workspace
from evora.perception.detect import resolve_device
from evora.perception.locks import STORE_SETUP
from evora.perception.settings import IngestSettings

log = logging.getLogger("evora.reid.features")


class ReidUnavailable(RuntimeError):
    """The ReID weights or the BoxMOT package are missing."""


def weights_path(name: str) -> Path:
    root = Path(os.environ.get("EVORA_MODELS_DIR", REPO_ROOT / "models"))
    return root / "boxmot" / name


class ReidEncoder:
    def __init__(self, cfg: IngestSettings):
        path = weights_path(cfg.reid_weights)
        if not path.is_file():
            raise ReidUnavailable(f"ReID weights missing: {path} (run scripts/models_download.py --only boxmot)")
        try:
            from boxmot.reid.core.runtime import ReID
        except ImportError as exc:
            raise ReidUnavailable("boxmot is not installed") from exc
        device = resolve_device(cfg.device)
        self._runtime = ReID(weights=path, device="cuda:0" if device == "cuda" else device, half=device == "cuda")
        self._lock = threading.Lock()
        self.dim = 0
        self.dim = int(self.encode([np.full((128, 64, 3), 127, dtype=np.uint8)]).shape[1])

    def encode(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """(N, D) float32, L2-normalised."""
        if not crops_bgr:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            return np.asarray(self._runtime(crops_bgr), dtype=np.float32)


_shared: ReidEncoder | None = None
_shared_lock = threading.Lock()


def get_encoder(cfg: IngestSettings) -> ReidEncoder:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = ReidEncoder(cfg)
        return _shared


def track_feature(crop_features: np.ndarray) -> np.ndarray:
    """Mean of a track's crop features, L2-normalised."""
    v = crop_features.mean(axis=0)
    return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)


def compute_reid(cam_id: str, ws: Workspace, db: Database, store, st: IngestSettings, encoder: ReidEncoder | None = None) -> int:
    """Write one ReID vector per track of a camera; returns the number of tracks embedded."""
    enc = encoder or get_encoder(st)
    with STORE_SETUP:
        db.set_meta("embed_dim_reid", str(enc.dim))
        db.set_meta("embed_model_reid", st.reid_weights)
        ensure_tables(store, dims_from_meta(db), only={"reid"})
    table = store.open_table("reid")
    table.delete(f"camera_id = '{cam_id}'")
    with db.read() as c:
        tracks = c.execute("SELECT id, cls, t_start, t_end FROM tracks WHERE camera_id=? ORDER BY id", (cam_id,)).fetchall()
    crop_rows = store.open_table("crops").search().where(f"camera_id = '{cam_id}'").limit(10_000_000).to_arrow().to_pylist()
    by_track: dict[str, list[dict]] = defaultdict(list)
    for r in crop_rows:
        by_track[r["track_id"]].append(r)
    rows = []
    for t in tracks:
        crops = []
        for r in sorted(by_track.get(t["id"], []), key=lambda r: -r["quality"])[: st.reid_crops_per_track]:
            img = cv2.imread(str(ws.media_dir / r["crop_path"]))
            if img is not None:
                crops.append(img)
        if not crops:
            continue
        rows.append({"vector": track_feature(enc.encode(crops)).tolist(), "track_id": t["id"], "camera_id": cam_id,
                     "cls": t["cls"], "t_start": t["t_start"], "t_end": t["t_end"]})
    if rows:
        table.add(rows)
    log.info("%s: %d ReID vectors (dim %d)", cam_id, len(rows), enc.dim)
    return len(rows)
