"""Motion gate: adaptive sampling rate between a floor and a ceiling (contribution C6).

Activity is the fraction of pixels that changed between consecutive 160 px grey proxies.
Quiet scenes are sampled near `fps_floor`; busy scenes near `fps_ceil`. With the gate off,
a fixed rate (`fixed_fps`) is used, which is the ablation baseline.
"""
from __future__ import annotations

import cv2
import numpy as np

from evora.perception.decode import Frame
from evora.perception.settings import IngestSettings


class AdaptiveSampler:
    """Decide, frame by frame in presentation order, which frames to process."""

    def __init__(self, cfg: IngestSettings):
        self.cfg = cfg
        self._prev: np.ndarray | None = None
        self._next_t = float("-inf")
        self.activity = 0.0

    def _proxy(self, frame: np.ndarray | Frame) -> np.ndarray:
        if isinstance(frame, Frame):
            grey = frame.proxy(self.cfg.motion_proxy_px)      # no full-size conversion
        else:
            h, w = frame.shape[:2]
            scale = self.cfg.motion_proxy_px / max(w, 1)
            small = cv2.resize(frame, (self.cfg.motion_proxy_px, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
            grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(grey, (5, 5), 0)

    def _measure(self, frame: np.ndarray | Frame) -> float:
        proxy = self._proxy(frame)
        if self._prev is None or self._prev.shape != proxy.shape:
            self._prev = proxy
            return 0.0
        diff = cv2.absdiff(proxy, self._prev)
        self._prev = proxy
        return float(np.count_nonzero(diff > self.cfg.motion_pixel_delta)) / diff.size

    def rate_for(self, activity: float) -> float:
        cfg = self.cfg
        if not cfg.motion_gate:
            return cfg.fixed_fps
        level = min(1.0, activity / max(cfg.motion_active_frac, 1e-6))
        return cfg.fps_floor + (cfg.fps_ceil - cfg.fps_floor) * level

    def should_process(self, pts_s: float, frame: np.ndarray | Frame) -> bool:
        """True when this frame is due. Motion is measured on every decoded frame so bursts are not missed.

        Passing a `Frame` keeps the measurement cheap: only a 160 px grey copy is made, never the full image.
        """
        self.activity = self._measure(frame) if self.cfg.motion_gate else 0.0
        if pts_s + 1e-9 < self._next_t:
            return False
        period = 1.0 / self.rate_for(self.activity)
        # advance from the previous due time (not from this frame) so quantisation does not lower the long-run rate
        base = self._next_t if pts_s - self._next_t < period else pts_s
        self._next_t = base + period
        return True
