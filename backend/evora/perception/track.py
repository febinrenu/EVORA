"""Detection plus multi-object tracking on the sampled frames.

Wraps `model.track(..., persist=True)` so detector and tracker share one forward pass. One
`FrameTracker` per camera: tracker state is per instance, the model is shared and guarded by a lock.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from evora.perception.detect import LoadedDetector
from evora.perception.settings import IngestSettings

_model_lock = threading.Lock()


@dataclass(frozen=True)
class TrackedBox:
    track_key: int                       # tracker id, unique within one camera pass
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]   # pixels in the processed frame


class FrameTracker:
    def __init__(self, det: LoadedDetector, cfg: IngestSettings):
        self.det, self.cfg = det, cfg
        self._fresh = True

    def update(self, bgr: np.ndarray) -> list[TrackedBox]:
        """Run detection and tracking on one frame; boxes without a tracker id are discarded."""
        with _model_lock:
            results = self.det.model.track(
                bgr, persist=not self._fresh, tracker=self.cfg.tracker, conf=self.cfg.det_conf,
                imgsz=self.cfg.det_imgsz, classes=self.det.class_ids, device=self.det.device,
                half=self.det.half, verbose=False,
            )
        self._fresh = False
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []
        ids = boxes.id.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        return [
            TrackedBox(int(i), self.det.names[int(c)], float(p), (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            for i, c, p, b in zip(ids, clss, confs, xyxy, strict=True)
        ]
