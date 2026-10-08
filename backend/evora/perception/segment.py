"""Person and vehicle masks on saved crops, so colour is read from the object and not from the background.

A small YOLO26 segmentation model (`models/ultralytics/yolo26n-seg.pt`, fetched by `scripts/models_download.py --only yolo`)
runs on each saved crop (at most a few per track, so the cost is small). `mask(crop, kind)` returns a boolean array of the
crop's size, or None when nothing was found, in which case the colour code falls back to geometric regions.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from evora.core.config import REPO_ROOT
from evora.perception.detect import resolve_device

log = logging.getLogger("evora.perception.segment")

COCO_PERSON = 0
COCO_VEHICLES = (1, 2, 3, 5, 7)     # bicycle, car, motorcycle, bus, truck
MIN_SIDE = 160                      # smaller crops are enlarged first, the model does poorly on tiny images


class SegmentUnavailable(RuntimeError):
    """The segmentation checkpoint is missing."""


class Segmenter:
    def __init__(self, weights: str = "yolo26n-seg.pt", device: str = "auto"):
        path = Path(os.environ.get("EVORA_MODELS_DIR", REPO_ROOT / "models")) / "ultralytics" / weights
        if not path.is_file():
            raise SegmentUnavailable(f"missing {path} (run scripts/models_download.py --only yolo)")
        from ultralytics import YOLO

        self._model = YOLO(str(path))
        self.device = resolve_device(device)
        self._lock = threading.Lock()

    def mask(self, crop_bgr: np.ndarray, kind: str = "person", conf: float = 0.2) -> np.ndarray | None:
        """Boolean mask (crop height x width) of the main object in the crop, or None."""
        h, w = crop_bgr.shape[:2]
        scale = max(1.0, MIN_SIDE / max(h, w))
        img = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else crop_bgr
        classes = [COCO_PERSON] if kind == "person" else list(COCO_VEHICLES)
        with self._lock:
            res = self._model.predict(img, classes=classes, conf=conf, device=self.device, verbose=False,
                                      imgsz=int(np.ceil(max(img.shape[:2]) / 32) * 32))[0]
        if res.masks is None or len(res.masks.data) == 0:
            return None
        masks = res.masks.data.cpu().numpy() > 0.5                     # (N, mh, mw) at the model's mask resolution
        masks = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]).astype(bool)
        # the object the crop was made for is the one in the middle, not a neighbour at the edge
        centre = np.zeros((h, w), dtype=bool)
        centre[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)] = True
        overlap = (masks & centre).sum(axis=(1, 2))
        best = int(np.argmax(overlap))
        if overlap[best] < 0.1 * centre.sum():
            return None
        return masks[best]


_shared: Segmenter | None = None
_shared_lock = threading.Lock()


def get_segmenter(device: str = "auto") -> Segmenter | None:
    """The process-wide segmenter, or None when the checkpoint is not installed."""
    global _shared
    with _shared_lock:
        if _shared is None:
            try:
                _shared = Segmenter(device=device)
            except (SegmentUnavailable, ImportError) as exc:
                log.warning("person masks unavailable, colour falls back to fixed regions: %s", exc)
                return None
        return _shared
