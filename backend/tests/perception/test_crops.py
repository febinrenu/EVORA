import numpy as np
import pytest

pytest.importorskip("cv2")

from evora.perception import crops  # noqa: E402
from evora.perception.settings import IngestSettings  # noqa: E402
from evora.perception.track import TrackedBox  # noqa: E402


def _textured(h=120, w=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w), dtype=np.uint8)


def test_sharp_beats_blurry():
    import cv2

    sharp = _textured()
    blurry = cv2.GaussianBlur(sharp, (15, 15), 0)
    assert crops.sharpness(sharp) > 5 * crops.sharpness(blurry)


def test_truncation():
    assert crops.truncation((10, 10, 50, 50), 100, 100) == 0
    assert crops.truncation((-20, 10, 20, 50), 100, 100) == pytest.approx(0.5)
    assert crops.truncation((200, 200, 220, 220), 100, 100) == 1


def test_quality_penalises_truncation_and_small_boxes():
    g = _textured()
    full = crops.crop_quality(0.9, g, 100, 240, 0.0)
    assert crops.crop_quality(0.9, g, 100, 240, 0.5) == pytest.approx(full / 2)
    assert crops.crop_quality(0.9, g, 20, 240, 0.0) < full


def test_direction():
    pts = [(0.0, 0.1, 0.4, 0.2, 0.6, 0.9), (2.0, 0.7, 0.4, 0.8, 0.6, 0.9)]
    assert crops.direction_of(pts) == "left_to_right"
    assert crops.direction_of(pts[:1]) is None
    still = [(0.0, 0.4, 0.4, 0.5, 0.6, 0.9), (2.0, 0.41, 0.4, 0.51, 0.6, 0.9)]
    assert crops.direction_of(still) is None


def _frame_with_person(rng, sharp=True):
    img = np.full((240, 320, 3), 40, dtype=np.uint8)
    patch = rng.integers(0, 255, (100, 50, 3), dtype=np.uint8)
    if not sharp:
        import cv2

        patch = cv2.GaussianBlur(patch, (21, 21), 0)
    img[60:160, 100:150] = patch
    return img


def test_trackbook_keeps_best_k_and_drops_short_tracks():
    cfg = IngestSettings(crop_k=2, min_track_obs=3)
    book = crops.TrackBook(cfg)
    rng = np.random.default_rng(1)
    box = (100.0, 60.0, 150.0, 160.0)
    for i in range(6):
        frame = _frame_with_person(rng, sharp=i % 2 == 0)
        book.observe(i * 0.5, [TrackedBox(7, "person", 0.9, box)], frame)
    book.observe(0.0, [TrackedBox(9, "car", 0.9, box)], _frame_with_person(rng))  # a 1-observation track
    done = book.finalize_all()
    assert [t.seq for t in done] == [1]
    track = done[0]
    assert track.cls == "person" and track.n_obs == 6
    assert len(track.crops) == 2
    assert track.crops[0].quality >= track.crops[1].quality
    assert {c.t for c in track.crops} <= {0.0, 1.0, 2.0}  # the sharp frames won


def test_stale_tracks_are_finalised_for_bounded_memory():
    cfg = IngestSettings(track_lost_s=2.0, min_track_obs=1)
    book = crops.TrackBook(cfg)
    rng = np.random.default_rng(2)
    book.observe(0.0, [TrackedBox(1, "person", 0.9, (100.0, 60.0, 150.0, 160.0))], _frame_with_person(rng))
    assert book.finalize_stale(1.0) == []
    assert len(book.finalize_stale(5.0)) == 1
    assert book.finalize_all() == []
