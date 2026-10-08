"""Perception (M2): clock detection, decoding, tracking, crops and embeddings.

`detect_clock`, `ingest` and `blur_faces` are the entry points the platform calls (PLAN.md section 5.6).
"""
from evora.perception.clock import detect_clock
from evora.perception.faces import blur_faces
from evora.perception.pipeline import ingest

__all__ = ["blur_faces", "detect_clock", "ingest"]
