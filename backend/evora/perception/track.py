"""Detection plus multi-object tracking on the sampled frames.

The detector is shared and guarded by a lock; every camera owns its own tracker object, so cameras
ingested at the same time cannot mix up each other's track ids.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from evora.perception.detect import LoadedDetector
from evora.perception.settings import IngestSettings


@dataclass(frozen=True)
class TrackedBox:
    track_key: int                       # tracker id, unique within one camera pass
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]   # pixels in the processed frame


def tracker_overrides(cfg: IngestSettings) -> dict[str, float | int]:
    """Our own start and buffer values on top of the stock yaml.

    The stock file starts a track only from a box scoring 0.25, which drops small, distant people (they score 0.15 to
    0.25). The buffer is counted in processed frames: it must last at least `track_lost_s` at the busiest sampling rate.
    """
    return {
        "track_high_thresh": cfg.track_high_thresh,
        "new_track_thresh": cfg.track_new_thresh,
        "track_buffer": max(30, int(math.ceil(cfg.track_lost_s * cfg.fps_ceil))),
    }


def build_tracker(tracker_cfg: str, overrides: dict[str, float | int] | None = None):
    """A fresh Ultralytics tracker (ByteTrack by default) from a tracker yaml name, with optional value overrides."""
    from evora.perception.lap_shim import install_if_missing

    install_if_missing()          # before Ultralytics imports `lap` (it would try to pip install it)
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**{**YAML.load(check_yaml(tracker_cfg)), **(overrides or {})})
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


def detector_size(width: int, cfg: IngestSettings) -> int:
    """Detector input size: the configured one, else the frame width rounded up to a multiple of 32, between 640 and the cap.

    At 640 pixels a 1080p scene shrinks cars and people to a few pixels and many are missed (on the MEVA hospital clip 6
    cars were found in 12 frames at 640 against 25 at 1280); small frames gain nothing from a larger size.
    """
    if cfg.det_imgsz:
        return cfg.det_imgsz
    return int(min(cfg.det_imgsz_max, max(640, -(-width // 32) * 32)))


class FrameTracker:
    def __init__(self, det: LoadedDetector, cfg: IngestSettings):
        self.det, self.cfg = det, cfg
        self._tracker, tracker_cfg = build_tracker(cfg.tracker, tracker_overrides(cfg))
        # ByteTrack uses low-confidence boxes to keep existing tracks alive, so the detector must report them
        self._conf = min(cfg.det_conf, float(getattr(tracker_cfg, "track_low_thresh", cfg.det_conf)))

    def update(self, bgr: np.ndarray) -> list[TrackedBox]:
        """Run detection and tracking on one frame; only boxes that belong to a track are returned."""
        with self.det.lock:
            results = self.det.model.predict(
                bgr, conf=self._conf, imgsz=detector_size(bgr.shape[1], self.cfg), classes=self.det.class_ids,
                device=self.det.device,
                quantize=16 if self.det.half else None, verbose=False,
            )
        boxes = results[0].boxes
        tracks = self._tracker.update(boxes.cpu().numpy(), bgr) if boxes is not None else np.empty((0, 8))
        out = []
        for x1, y1, x2, y2, tid, score, cls, _ in np.asarray(tracks).reshape(-1, 8)[:, :8].tolist():
            out.append(TrackedBox(int(tid), self.det.names[int(cls)], float(score), (x1, y1, x2, y2)))
        return out
