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
    det_imgsz: int = 0                # detector input size; 0 = automatic from the frame width (see track.detector_size)
    det_imgsz_max: int = 1280         # largest automatic size: small, distant objects need it, a 4x cost per frame
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
    # colour
    # "v2" = person masks, survey naming and shades (default: it agrees better with two independent opinions and removes the
    # phantom purple / brown / pink people of the original). "legacy" = the original method, kept for comparison.
    # Without the segmentation checkpoint v2 falls back to fixed regions by itself.
    colour_engine: str = "v2"
    # A colour claim needs this calibrated confidence, otherwise the colour is left unknown. On held-out human labels, predictions
    # below 0.6 were right only about 40% of the time (coin flips), those at 0.8 or more 95-100%. 0 disables the floor.
    colour_min_conf: float = 0.6
    # L3 captions
    l3_max_tracks: int = 20           # best tracks per camera to caption (captions are a bonus; they must never delay indexing)
    l3_budget_s: float = 45.0         # time limit per camera; the layer stops and keeps what it has
    l3_max_tokens: int = 512          # reasoning vision models spend tokens thinking before they answer
    # events
    dwell_s: float = 20.0             # continuous presence in a zone that counts as dwelling
    line_hysteresis: float = 0.01     # normalised distance beyond a line before a crossing counts
    zone_debounce: int = 2            # samples a zone enter/exit must persist
    # appearance (ReID) features
    reid_weights: str = "osnet_x0_25_msmt17.pt"
    reid_crops_per_track: int = 4
    # cross-camera linking (contribution C5)
    reid_topology: bool = True        # False = appearance and attributes only (ablation)
    reid_max_gap_s: float = 180.0     # longest plausible time between leaving one camera and appearing in another
    reid_min_track_s: float = 1.0     # shorter tracks are too noisy to link
    reid_bootstrap_thr: float = 0.80  # appearance cosine for the confident matches that teach the topology
    reid_accept_thr: float = 0.55     # combined score needed to link two tracks
    reid_w_appearance: float = 0.6
    reid_w_attributes: float = 0.2
    reid_w_topology: float = 0.2
    reid_topology_prior: float = 0.3  # topology score for a camera pair with no learned link
    reid_std_floor_s: float = 5.0
    reid_cross_min_cos: float = 0.72  # appearance floor for a cross-camera link (a wrong merge costs more than a missed one)
    reid_veto_conf: float = 0.6       # two clearly different, confidently read colours veto a link
    reid_min_z: float = 3.0           # a match must stand this many std above the camera pair's background similarity
    reid_margin: float = 0.05         # and beat the runner-up by this much to teach the topology
    reid_topology_p: float = 0.05     # matched gaps must be more concentrated than chance at this p-value
    # within-camera stitching: one person broken into several tracks becomes one identity
    reid_stitch: bool = True
    reid_stitch_max_gap_s: float = 8.0       # longest pause between two fragments of one person
    reid_stitch_min_track_s: float = 0.5     # fragments this short can still be stitched
    reid_stitch_min_cos: float = 0.70        # appearance must be at least this similar ...
    reid_stitch_min_z: float = 2.0           # ... and stand out from the camera's background similarity
    reid_stitch_reach: float = 0.06          # allowed jump (normalised) plus this much per second of gap
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
