"""FrameTracker with a scripted detector: tracker state belongs to the camera, not to the shared model."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ultralytics")
torch = pytest.importorskip("torch")

from ultralytics.engine.results import Boxes  # noqa: E402

from evora.perception.settings import IngestSettings  # noqa: E402
from evora.perception.track import FrameTracker  # noqa: E402


class ScriptedModel:
    """predict() returns the boxes of whichever scripted frame the caller says is current."""

    def __init__(self):
        self.next_boxes = []

    def predict(self, img, **kwargs):
        rows = torch.tensor(self.next_boxes, dtype=torch.float32).reshape(-1, 6)
        return [SimpleNamespace(boxes=Boxes(rows, img.shape[:2]))]


def _detector(model):
    return SimpleNamespace(model=model, class_ids=[0], names={0: "person"}, device="cpu", half=False)


def _walk(t, x0):
    x = x0 + 12 * t
    return [x, 40.0, x + 40, 160.0, 0.9, 0.0]


def test_two_cameras_keep_independent_track_ids():
    model = ScriptedModel()
    cfg = IngestSettings(tracker="bytetrack.yaml")
    a, b = FrameTracker(_detector(model), cfg), FrameTracker(_detector(model), cfg)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    ids_a, ids_b = [], []
    for t in range(8):
        model.next_boxes = [_walk(t, 10)]            # camera a: one person
        ids_a += [x.track_key for x in a.update(frame)]
        model.next_boxes = [_walk(t, 200), _walk(t, 60)]  # camera b: two people, updated in between a's frames
        ids_b += [x.track_key for x in b.update(frame)]
    assert len(set(ids_a)) == 1                        # one person stays one track
    assert len(set(ids_b)) == 2
    assert ids_a[0] == min(ids_b) == 1                 # each camera counts from 1: nothing is shared


def test_unconfirmed_first_frames_and_empty_detections_are_handled():
    model = ScriptedModel()
    tracker = FrameTracker(_detector(model), IngestSettings())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    model.next_boxes = []
    assert tracker.update(frame) == []
    model.next_boxes = [_walk(0, 10)]
    out = tracker.update(frame)
    assert all(o.cls == "person" and o.conf == pytest.approx(0.9, abs=0.2) for o in out)


def test_creating_a_tracker_does_not_reset_a_running_cameras_ids():
    model = ScriptedModel()
    cfg = IngestSettings()
    a = FrameTracker(_detector(model), cfg)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    seen = set()
    for t in range(6):
        model.next_boxes = [_walk(t, 10), _walk(t, 150)]
        seen |= {x.track_key for x in a.update(frame)}
    FrameTracker(_detector(model), cfg)               # another camera starts while a is running
    ids: list[int] = []
    for t in range(6, 9):                              # a third person appears in a; it is confirmed on its second frame
        model.next_boxes = [_walk(t, 10), _walk(t, 150), [250.0 + 6 * t, 40.0, 290.0 + 6 * t, 160.0, 0.9, 0.0]]
        ids = [x.track_key for x in a.update(frame)]
    assert len(ids) == len(set(ids)) == 3
    assert max(ids) not in seen and max(ids) == 3      # continues from 3; it was not reset to 1 by the other tracker


@pytest.mark.parametrize(
    "width, expected", [(320, 640), (360, 640), (640, 640), (800, 800), (1000, 1024), (1280, 1280), (1920, 1280)])
def test_detector_size_follows_the_frame_width(width, expected):
    from evora.perception.track import detector_size

    assert detector_size(width, IngestSettings()) == expected
    assert detector_size(width, IngestSettings(det_imgsz=512)) == 512        # an explicit setting always wins


def test_tracker_starts_small_distant_people_and_keeps_them_for_the_lost_time():
    from evora.perception.settings import IngestSettings
    from evora.perception.track import build_tracker, tracker_overrides

    cfg = IngestSettings()
    tracker, args = build_tracker(cfg.tracker, tracker_overrides(cfg))
    assert args.new_track_thresh == cfg.track_new_thresh and args.track_high_thresh == cfg.track_high_thresh
    assert args.track_buffer >= cfg.track_lost_s * cfg.fps_ceil          # frames at the busiest rate cover track_lost_s
    stock, stock_args = build_tracker(cfg.tracker)
    assert stock_args.new_track_thresh == 0.25
