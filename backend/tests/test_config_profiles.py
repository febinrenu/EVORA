from evora.core.config import load_config
from evora.perception.settings import IngestSettings
from evora.perception.track import detector_size


def detector_input(profile: str, frame_width: int = 1280) -> int:
    ingest = load_config(profile)["ingest"]
    return detector_size(frame_width, IngestSettings(det_imgsz=ingest["det_imgsz"]))


def test_the_gpu_profile_lets_the_detector_see_the_frame_at_full_decoded_size():
    assert detector_input("gpu") == 1280, "a 1080p person must not be shrunk to a third before detection"
    assert detector_input("mps") == 1280


def test_the_cpu_profile_keeps_the_fast_input_size():
    assert detector_input("cpu") == 640


def test_small_frames_are_not_enlarged():
    assert detector_input("gpu", frame_width=360) == 640
