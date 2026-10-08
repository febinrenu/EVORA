"""Perception (M2): clock detection, decoding, tracking, crops and embeddings.

`detect_clock` and `ingest` are the entry points the platform calls (PLAN.md section 5.6).
"""
from evora.perception.clock import detect_clock
from evora.perception.pipeline import ingest

__all__ = ["detect_clock", "ingest"]
