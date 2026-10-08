import numpy as np
import pytest

pytest.importorskip("cv2")

from evora.perception.motion import AdaptiveSampler  # noqa: E402
from evora.perception.settings import IngestSettings  # noqa: E402


def _frame(shift: int = 0) -> np.ndarray:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[100:160, 40 + shift : 120 + shift] = 255
    return img


def _count(settings: IngestSettings, moving: bool, seconds: int = 20, src_fps: int = 25) -> int:
    s = AdaptiveSampler(settings)
    kept = 0
    for i in range(seconds * src_fps):
        frame = _frame((i * 13) % 150 if moving else 0)
        kept += s.should_process(i / src_fps, frame)
    return kept


def test_static_scene_stays_near_the_floor():
    kept = _count(IngestSettings(fps_floor=1, fps_ceil=8), moving=False)
    assert 18 <= kept <= 22


def test_busy_scene_approaches_the_ceiling():
    kept = _count(IngestSettings(fps_floor=1, fps_ceil=8), moving=True)
    assert kept >= 20 * 6


def test_gate_off_uses_the_fixed_rate():
    kept = _count(IngestSettings(motion_gate=False, fixed_fps=4), moving=False)
    assert 78 <= kept <= 82


def test_first_frame_is_always_processed():
    assert AdaptiveSampler(IngestSettings()).should_process(0.0, _frame()) is True
