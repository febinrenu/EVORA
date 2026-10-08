"""Perception (M2): clock detection, decoding, tracking, crops and embeddings.

`detect_clock` is the entry point the platform calls (PLAN.md section 5.6); `ingest` is added once the
pipeline is verified end to end.
"""
from evora.perception.clock import detect_clock

__all__ = ["detect_clock"]
