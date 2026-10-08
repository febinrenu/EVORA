from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

Epoch = float  # seconds since Unix epoch, UTC

class TrackAttrs(BaseModel):
    color: str | None = None; color_conf: float | None = None          # one of 11 basic colour terms
    upper_color: str | None = None; lower_color: str | None = None       # persons
    vehicle_type: str | None = None                                     # car|suv|truck|bus|motorcycle|bicycle|auto_rickshaw|van
    carrying: list[str] = []                                            # backpack|handbag|suitcase|large_bag|umbrella
    size_rel: float | None = None                                       # bbox height / frame height (median)
    is_ir: bool = False                                                 # colour unreliable

class CameraInfo(BaseModel):
    id: str; name: str; kind: Literal["file", "rtsp"]; source_uri: str
    fps: float | None = None; width: int | None = None; height: int | None = None
    t0: Epoch; t0_source: Literal["filename", "metadata", "osd", "slate", "manual", "live"]
    duration_s: float | None = None
    site_xy: tuple[float, float] | None = None
    status: Literal["pending", "ingesting", "ready", "live", "error"] = "pending"
    layers: list[Literal["L0", "L1", "L2", "L3"]] = []
    ir_fraction: float | None = None

class Zone(BaseModel):
    id: str; camera_id: str; kind: Literal["line", "polygon", "frame"]
    points: list[tuple[float, float]] = []
    direction: Literal["any", "a_to_b", "b_to_a"] = "any"

class TimeWindow(BaseModel):
    start: Epoch | None = None; end: Epoch | None = None
    phrase: str | None = None
    tod_after: str | None = None; tod_before: str | None = None   # "20:00"

class Target(BaseModel):
    noun: str
    cls: list[str] = []
    attributes: list[str] = []
    embed_text: str
    example_track_id: str | None = None

class Referent(BaseModel):
    text: str; role: Literal["place", "object", "time"]

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
    id: str; camera_id: str; camera_name: str
    t_start: Epoch; t_end: Epoch; t_peak: Epoch
    offset_s: float                      # seconds into the camera's own file: the second timestamp we always show
    track_id: str | None = None; global_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    thumb_url: str; clip_url: str
    score: float; verified: bool | None = None
    why: list[str] = []                  # provenance, e.g. "siglip 0.31", "colour red 0.92", "crossed Main gate 09:14:03"

class PathHop(BaseModel):
    camera_id: str; camera_name: str; t_in: Epoch; t_out: Epoch; evidence_id: str

class Answer(BaseModel):
    query_id: str; text: str
    verdict: Literal["yes", "no", "found", "not_found", "partial", "count"]
    count: int | None = None
    evidence: list[Evidence] = []
    path: list[PathHop] = []
    nearest_miss: Evidence | None = None
    confidence: float
    plan: QueryPlan
    timings_ms: dict[str, float] = {}
    notes: list[str] = []

class CameraOption(BaseModel):
    camera_id: str; camera_name: str; thumb_url: str

class ClarifyRequest(BaseModel):
    query_id: str; referent: Referent; question: str
    kind: Literal["choose_camera", "choose_known", "choose_track", "time_range"]
    options: list[CameraOption] = []
    known_candidates: list[str] = []     # fact ids when disambiguating between known facts
    allow_region: bool = True

class ClarifyResponse(BaseModel):
    query_id: str
    camera_id: str | None = None; zone: Zone | None = None
    fact_id: str | None = None; track_id: str | None = None
    tod_after: str | None = None; tod_before: str | None = None
    text: str | None = None              # typed answer like "camera 2"; parsed server-side

class MemoryFact(BaseModel):
    id: str; kind: Literal["place", "object", "time"]
    canonical: str; aliases: list[str] = []
    binding: dict
    source: Literal["clarification", "statement", "correction", "import"]
    created_at: Epoch; last_used_at: Epoch | None = None; use_count: int = 0
    superseded_by: str | None = None

class StandingQuery(BaseModel):
    id: str; text: str; rule: dict; active: bool = True; created_at: Epoch

class Alert(BaseModel):
    id: str; standing_query_id: str; t: Epoch; camera_id: str
    evidence: Evidence; acknowledged: bool = False

class IngestJob(BaseModel):
    id: str; camera_id: str
    state: Literal["queued", "running", "done", "error"]
    layer: Literal["L0", "L1", "L2", "L3"] | None = None
    progress: float = 0.0; video_s_per_s: float | None = None; error: str | None = None

class StreamEvent(BaseModel):
    type: Literal["plan", "clarify", "evidence", "answer", "verified", "note", "error", "done"]
    data: dict
