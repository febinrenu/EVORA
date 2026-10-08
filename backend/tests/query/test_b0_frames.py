from types import SimpleNamespace

import numpy as np
import pytest
from contracts.models import QueryPlan, Target, TimeWindow

from evora.baseline.b0_frames import FrameIndex, b0_search, merge_hits, windows_to_evidence


class ToyEmbedder:
    """Two concepts: axis 0 is 'red car', axis 1 is 'blue truck', axis 2 is background."""

    DIM = 3

    def embed_text(self, text):
        v = np.zeros(self.DIM, dtype=np.float32)
        v[0 if "red car" in text else 1 if "blue truck" in text else 2] = 1.0
        return v

    def embed_images(self, images):
        return np.array(images, dtype=np.float32)


RED, BLUE, BG = [1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]


def plan(text="a photo of a red car", cams=(), window=None, limit=10):
    return QueryPlan(intent="list", targets=[Target(noun="car", cls=["car"], embed_text=text)],
                     camera_ids=list(cams), time=window, limit=limit)


def build():
    idx = FrameIndex()
    emb = ToyEmbedder()
    # cam_01: red car at t=10..12 and again at t=100, blue truck at t=50
    frames1 = [(float(t), BG) for t in range(0, 120)]
    frames1 = [(t, RED if 10 <= t <= 12 or t == 100 else BLUE if t == 50 else v) for t, v in frames1]
    idx.add_frames("cam_01", frames1, emb, batch=16)
    idx.add_frames("cam_02", [(float(t), RED if t == 30 else BG) for t in range(0, 60)], emb)
    return idx, emb


def test_adjacent_hits_merge_into_one_window_with_the_best_peak():
    hits = [("c", 10.0, 0.5), ("c", 11.0, 0.9), ("c", 12.0, 0.6), ("c", 100.0, 0.7), ("d", 11.0, 0.4)]
    ws = merge_hits(hits)
    assert [(w.camera_id, w.t_start, w.t_end, w.t_peak, w.score) for w in ws] == [
        ("c", 10.0, 12.0, 11.0, 0.9), ("c", 100.0, 100.0, 100.0, 0.7), ("d", 11.0, 11.0, 11.0, 0.4)]


def test_gap_threshold_is_inclusive():
    assert len(merge_hits([("c", 0.0, 1.0), ("c", 2.0, 1.0)])) == 1
    assert len(merge_hits([("c", 0.0, 1.0), ("c", 2.5, 1.0)])) == 2


def test_search_finds_the_right_frames_across_cameras():
    idx, emb = build()
    ws = b0_search(idx, emb, plan(), top_frames=5)  # exactly five red frames exist
    spans = {(w.camera_id, w.t_start, w.t_end) for w in ws}
    assert ("cam_01", 10.0, 12.0) in spans and ("cam_01", 100.0, 100.0) in spans and ("cam_02", 30.0, 30.0) in spans
    assert not any(w.camera_id == "cam_01" and w.t_start == 50.0 for w in ws)  # the blue truck is not a red car


def test_same_camera_filter_as_our_system():
    idx, emb = build()
    ws = b0_search(idx, emb, plan(cams=["cam_02"]), top_frames=6)
    assert {w.camera_id for w in ws} == {"cam_02"}


def test_same_time_window_as_our_system():
    idx, emb = build()
    ws = b0_search(idx, emb, plan(window=TimeWindow(start=90.0, end=110.0)), top_frames=6)
    assert [(w.camera_id, w.t_peak) for w in ws if w.score > 0.9] == [("cam_01", 100.0)]
    assert all(90.0 <= w.t_start and w.t_end <= 110.0 for w in ws)


def test_limit_and_empty_cases():
    idx, emb = build()
    assert len(b0_search(idx, emb, plan(limit=1), top_frames=6)) == 1
    assert b0_search(FrameIndex(), emb, plan()) == []
    assert b0_search(idx, emb, QueryPlan(intent="describe")) == []


def test_vectors_are_normalised_and_lengths_checked():
    idx = FrameIndex()
    idx.add("c", [0.0], np.array([[3.0, 4.0, 0.0]]))
    assert idx.search(np.array([3.0, 4.0, 0.0]))[0][2] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        idx.add("c", [0.0, 1.0], np.array([[1.0, 0.0, 0.0]]))


def test_add_frames_batches_and_counts():
    idx = FrameIndex()
    n = idx.add_frames("c", [(float(t), BG) for t in range(37)], ToyEmbedder(), batch=16)
    assert n == 37 and len(idx) == 37


def test_windows_become_evidence_with_file_offsets():
    idx, emb = build()
    ws = b0_search(idx, emb, plan(cams=["cam_02"]), top_frames=3)
    cams = [SimpleNamespace(id="cam_02", name="Lobby", t0=1000.0)]
    ev = windows_to_evidence(ws, cams)[0]
    assert ev.camera_name == "Lobby" and ev.offset_s == ev.t_peak - 1000.0 and ev.id == "b0_001"
    assert ev.why[0].startswith("frame cosine")
