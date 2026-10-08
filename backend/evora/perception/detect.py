"""Detector loading and class mapping (Ultralytics YOLO26).

The model is loaded once per process and shared by tracking. Class ids are looked up by name from
the model's own label map, never hard-coded, so any COCO-style checkpoint works.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from evora.core.config import REPO_ROOT
from evora.perception.settings import IngestSettings

log = logging.getLogger("evora.perception.detect")

PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
BAG_CLASSES = {"backpack", "handbag", "suitcase", "umbrella"}


@dataclass
class LoadedDetector:
    model: object            # ultralytics.YOLO
    class_ids: list[int]     # ids to keep, in the order of settings.classes
    names: dict[int, str]    # id -> label
    device: str
    half: bool
    lock: threading.Lock = field(default_factory=threading.Lock)   # one predict at a time per model object


_lock = threading.Lock()
_cache: dict[tuple[str, str, int], LoadedDetector] = {}


def resolve_device(requested: str) -> str:
    """auto -> cuda if available, else mps, else cpu. Explicit values pass through."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def weights_path(name: str) -> str:
    """Prefer a weight already downloaded by scripts/models_download.py, else let Ultralytics fetch it."""
    root = Path(os.environ.get("EVORA_MODELS_DIR", REPO_ROOT / "models"))
    local = root / "ultralytics" / name
    return str(local) if local.is_file() else name


def load_detector(cfg: IngestSettings) -> LoadedDetector:
    device = resolve_device(cfg.device)
    # one model object per thread: the 6 MB detector is cheap, and workers no longer queue behind one shared model
    key = (cfg.detector, device, threading.get_ident())
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        from ultralytics import YOLO

        model = YOLO(weights_path(cfg.detector))
        names = {int(k): str(v) for k, v in model.names.items()}
        by_name = {v: k for k, v in names.items()}
        missing = [c for c in cfg.classes if c not in by_name]
        if missing:
            log.warning("detector %s has no classes %s", cfg.detector, missing)
        ids = [by_name[c] for c in cfg.classes if c in by_name]
        if not ids:
            raise RuntimeError(f"detector {cfg.detector} knows none of the configured classes")
        det = LoadedDetector(model=model, class_ids=ids, names=names, device=device, half=device == "cuda")
        _cache[key] = det
        log.info("detector %s on %s, %d classes", cfg.detector, device, len(ids))
        return det
