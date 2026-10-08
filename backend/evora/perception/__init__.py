"""Perception (M2): clock detection, decoding, tracking, crops and embeddings.

`detect_clock`, `ingest`, `live_ingest`, `blur_faces` and `recompute_events` are the entry points the platform
calls (PLAN.md section 5.6).
"""
from evora.perception.clock import detect_clock
from evora.perception.events import recompute_events
from evora.perception.faces import blur_faces
from evora.perception.live import live_ingest
from evora.perception.pipeline import ingest

__all__ = ["blur_faces", "detect_clock", "ingest", "live_ingest", "recompute_events"]
