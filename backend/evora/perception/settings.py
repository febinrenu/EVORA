"""Perception settings: the `ingest:` block of config/default.yaml, with safe defaults.

The defaults here only keep the pipeline runnable until the block lands in the shared config;
every value can be overridden from `config/default.yaml` or a profile.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestSettings(BaseModel):
    # sampling (contribution C6)
    motion_gate: bool = True
    fps_floor: float = 1.0
    fps_ceil: float = 8.0
    fixed_fps: float = 4.0            # used when motion_gate is off
    motion_proxy_px: int = 160
    motion_pixel_delta: int = 18      # grey-level change that counts as "moved"
    motion_active_frac: float = 0.02  # fraction of proxy pixels changed that maps to fps_ceil
    # decode
    max_width: int = 1280
    # L0 scene embeddings
    scene_every_s: float = 2.0
    scene_jpeg_width: int = 640
    # L1 detection and tracking
    detector: str = "yolo26n.pt"
    det_conf: float = 0.25
    det_imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    classes: list[str] = Field(default_factory=lambda: [
        "person", "bicycle", "car", "motorcycle", "bus", "truck",
        "backpack", "handbag", "suitcase", "umbrella",
    ])
    track_lost_s: float = 5.0         # a track unseen this long is finalised
    min_track_obs: int = 3            # shorter tracks are noise and are dropped
    # crops
    crop_k: int = 4
    crop_pad: float = 0.08
    crop_jpeg_quality: int = 88
    crop_max_side: int = 256
    point_hz: float = 4.0             # track_points downsample rate
    # embeddings
    image_model: str = "google/siglip2-base-patch16-224"
    embed_batch: int = 32
    # device: "auto" picks from the profile (cuda / mps / cpu)
    device: str = "auto"


def load_settings(cfg: dict[str, Any] | None = None) -> IngestSettings:
    """Build settings from a loaded config dict (`cfg["ingest"]` if present) plus the profile device."""
    block = dict((cfg or {}).get("ingest") or {})
    device = (cfg or {}).get("device")
    if device and "device" not in block:
        block["device"] = device
    return IngestSettings(**block)
