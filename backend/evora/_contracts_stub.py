"""Temporary copy of the v1 models from PLAN section 5.4.

Used only until `contracts/models.py` lands (cp0). At that point every import of
this module is switched to the real one and this file is deleted.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Epoch = float  # seconds since Unix epoch, UTC


class TrackAttrs(BaseModel):
    color: str | None = None
    color_conf: float | None = None
    upper_color: str | None = None
    lower_color: str | None = None
    vehicle_type: str | None = None
    carrying: list[str] = []
    size_rel: float | None = None
    is_ir: bool = False


class CameraInfo(BaseModel):
    id: str
    name: str
    kind: Literal["file", "rtsp"]
    source_uri: str
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    t0: Epoch
    t0_source: Literal["filename", "metadata", "osd", "slate", "manual", "live"]
    duration_s: float | None = None
    site_xy: tuple[float, float] | None = None
    status: Literal["pending", "ingesting", "ready", "live", "error"] = "pending"
    layers: list[Literal["L0", "L1", "L2", "L3"]] = []
    ir_fraction: float | None = None


class Zone(BaseModel):
    id: str
    camera_id: str
    kind: Literal["line", "polygon", "frame"]
    points: list[tuple[float, float]] = []
    direction: Literal["any", "a_to_b", "b_to_a"] = "any"


class TimeWindow(BaseModel):
    start: Epoch | None = None
    end: Epoch | None = None
    phrase: str | None = None
    tod_after: str | None = None
    tod_before: str | None = None


class Target(BaseModel):
    noun: str
    cls: list[str] = []
    attributes: list[str] = []
    embed_text: str
    example_track_id: str | None = None


class Referent(BaseModel):
    text: str
    role: Literal["place", "object", "time"]


class QueryPlan(BaseModel):
    intent: Literal["exists", "list", "count", "first", "last", "path", "describe", "standing"]
    targets: list[Target] = []
    place: Referent | None = None
    action: Literal["any", "pass_through", "enter", "exit", "dwell", "appear"] = "any"
    time: TimeWindow | None = None
    camera_ids: list[str] = []
    limit: int = 10
    unresolved: list[Referent] = []
    source: Literal["fastpath", "llm", "local_llm", "cache"] = "llm"


class Evidence(BaseModel):
    id: str
    camera_id: str
    camera_name: str
    t_start: Epoch
    t_end: Epoch
    t_peak: Epoch
    offset_s: float
    track_id: str | None = None
    global_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    thumb_url: str
    clip_url: str
    score: float
    verified: bool | None = None
    why: list[str] = []


class PathHop(BaseModel):
    camera_id: str
    camera_name: str
    t_in: Epoch
    t_out: Epoch
    evidence_id: str


class Answer(BaseModel):
    query_id: str
    text: str
    verdict: Literal["yes", "no", "found", "not_found", "partial", "count"]
    count: int | None = None
    evidence: list[Evidence] = []
    path: list[PathHop] = []
    nearest_miss: Evidence | None = None
    confidence: float
    plan: QueryPlan
    timings_ms: dict[str, float] = {}
    notes: list[str] = []


class StreamEvent(BaseModel):
    type: Literal["plan", "clarify", "evidence", "answer", "verified", "note", "error", "done"]
    data: dict
