"""Detection plus multi-object tracking on the sampled frames.

The detector is shared and guarded by a lock; every camera owns its own tracker object, so cameras
ingested at the same time cannot mix up each other's track ids.
"""
from __future__ import annotations

import itertools
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


def build_tracker(tracker_cfg: str):
    """A fresh Ultralytics tracker (ByteTrack by default) from a tracker yaml name."""
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_cfg)))
    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)
    # Ultralytics numbers tracks from one counter shared by every tracker in the process, and creating a tracker
    # resets it. Give this tracker its own counter so cameras ingested at the same time cannot collide.
    ids = itertools.count(1)

    class CameraTrack(tracker.track_class):
        @staticmethod
        def next_id() -> int:
            return next(ids)

    tracker.track_class = CameraTrack
    return tracker, cfg


class FrameTracker:
    def __init__(self, det: LoadedDetector, cfg: IngestSettings):
        self.det, self.cfg = det, cfg
        self._tracker, tracker_cfg = build_tracker(cfg.tracker)
        # ByteTrack uses low-confidence boxes to keep existing tracks alive, so the detector must report them
        self._conf = min(cfg.det_conf, float(getattr(tracker_cfg, "track_low_thresh", cfg.det_conf)))

    def update(self, bgr: np.ndarray) -> list[TrackedBox]:
        """Run detection and tracking on one frame; only boxes that belong to a track are returned."""
        with _model_lock:
            results = self.det.model.predict(
                bgr, conf=self._conf, imgsz=self.cfg.det_imgsz, classes=self.det.class_ids, device=self.det.device,
                quantize=16 if self.det.half else None, verbose=False,
            )
        boxes = results[0].boxes
        tracks = self._tracker.update(boxes.cpu().numpy(), bgr) if boxes is not None else np.empty((0, 8))
        out = []
        for x1, y1, x2, y2, tid, score, cls, _ in np.asarray(tracks).reshape(-1, 8)[:, :8].tolist():
            out.append(TrackedBox(int(tid), self.det.names[int(cls)], float(score), (x1, y1, x2, y2)))
        return out
