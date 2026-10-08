import numpy as np
import pytest

pytest.importorskip("cv2")

from evora.perception import attributes as at  # noqa: E402

SWATCHES_RGB = {
    "red": (200, 30, 30), "orange": (240, 140, 20), "yellow": (240, 220, 40), "green": (40, 150, 60),
    "blue": (30, 70, 200), "purple": (130, 50, 160), "pink": (245, 150, 190), "brown": (120, 70, 35),
    "black": (12, 12, 12), "white": (245, 245, 245), "grey": (128, 128, 128),
}


def _patch(rgb, h=60, w=40, noise=6, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array(rgb[::-1], dtype=np.int16)  # BGR
    img = base + rng.integers(-noise, noise + 1, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", list(SWATCHES_RGB))
def test_basic_colour_terms(name):
    term, conf = at.dominant_colour(_patch(SWATCHES_RGB[name]))
    assert term == name
    assert 0.3 < conf <= 1.0


def test_all_eleven_terms_are_covered():
    assert set(SWATCHES_RGB) == set(at.COLOURS)


def test_dominant_cluster_wins_over_background():
    img = _patch(SWATCHES_RGB["red"], h=60, w=40)
    img[:, :10] = _patch(SWATCHES_RGB["green"], h=60, w=10)  # a quarter of the pixels are background
    assert at.dominant_colour(img)[0] == "red"


def test_tiny_region_is_not_judged():
    assert at.dominant_colour(np.zeros((2, 3, 3), dtype=np.uint8)) is None


def test_white_balance_removes_a_colour_cast():
    cast = np.array([1.35, 1.0, 0.8], dtype=np.float32)  # blue-ish camera: B high, R low
    background = [np.clip(np.full((72, 128, 3), 120, np.float32) * cast, 0, 255).astype(np.uint8) for _ in range(3)]
    gains = at.white_balance_gains(background)
    assert gains[0] < 1.0 < gains[2]
    grey_garment = np.clip(_patch((170, 170, 170)).astype(np.float32) * cast, 0, 255).astype(np.uint8)
    assert at.dominant_colour(grey_garment)[0] != "grey"          # the cast makes it look coloured
    assert at.dominant_colour(grey_garment, gains)[0] == "grey"   # the gains undo it


def test_white_balance_is_neutral_without_frames_and_bounded():
    assert np.array_equal(at.white_balance_gains([]), np.ones(3, dtype=np.float32))
    extreme = [np.dstack([np.full((72, 128), v, np.uint8) for v in (250, 120, 5)])]
    g = at.white_balance_gains(extreme)
    assert g.min() >= 0.7 and g.max() <= 1.4


def test_infrared_frame_detection():
    grey = np.dstack([np.full((72, 128), 90, np.uint8)] * 3)
    colour = np.dstack([np.full((72, 128), v, np.uint8) for v in (30, 150, 200)])
    assert at.is_ir_frame(grey) is True
    assert at.is_ir_frame(colour) is False


def test_person_and_vehicle_regions():
    crop = np.zeros((100, 50, 3), dtype=np.uint8)
    upper, lower = at.person_regions(crop)
    assert upper.shape[0] == 35 and lower.shape[0] == 40 and upper.shape[1] == 30
    assert at.vehicle_region(crop).shape == (60, 30, 3)


def test_aggregate_colour_votes_by_summed_confidence():
    term, conf = at.aggregate_colour([("red", 0.9), ("blue", 0.4), ("red", 0.6), None])
    assert term == "red" and conf == pytest.approx(1.5 / 3)
    assert at.aggregate_colour([None, None]) == (None, None)


def test_vehicle_type_zero_shot():
    eye = np.eye(len(at.VEHICLE_TYPES), dtype=np.float32)
    crops = np.stack([eye[at.VEHICLE_TYPES.index("truck")] * 0.9 + eye[0] * 0.1] * 3)
    kind, conf = at.classify_vehicle(crops, eye)
    assert kind == "truck" and conf > 0.9
    assert at.classify_vehicle(np.zeros((0, 8), dtype=np.float32), eye) is None
    subset = at.classify_vehicle(crops, eye[[2, 4]], ("truck", "van"))   # names follow the vectors that were passed
    assert subset is not None and subset[0] in ("truck", "van")


def _walk(x0, dx, n=8, w=0.1, h=0.4, y=0.3):
    return [at.Box(t=i * 0.25, x1=x0 + dx * i, y1=y, x2=x0 + dx * i + w, y2=y + h) for i in range(n)]


def test_bag_moving_with_a_person_is_carried_and_large_is_flagged():
    person = _walk(0.2, 0.02)
    bag = [at.Box(b.t, b.x1 + 0.02, b.y1 + 0.15, b.x2 - 0.02, b.y1 + 0.35) for b in person]  # 0.2 tall vs 0.4 person
    assert at.carried_by(person, bag) == (True, True)
    small = [at.Box(b.t, b.x1 + 0.03, b.y1 + 0.2, b.x2 - 0.03, b.y1 + 0.28) for b in person]
    assert at.carried_by(person, small) == (True, False)


def test_a_bag_left_behind_is_not_carried():
    person = _walk(0.1, 0.05)
    bag = [at.Box(b.t, 0.8, 0.5, 0.85, 0.6) for b in person]
    assert at.carried_by(person, bag) == (False, False)


def test_too_few_frames_is_not_carried():
    person = _walk(0.2, 0.0, n=8)
    bag = [at.Box(b.t, 0.22, 0.4, 0.26, 0.5) for b in person[:2]]
    assert at.carried_by(person, bag)[0] is False


def test_carrying_terms_follow_the_contract():
    assert at.carrying_terms([("backpack", False)]) == ["backpack"]
    assert at.carrying_terms([("suitcase", True), ("umbrella", False)]) == ["suitcase", "large_bag", "umbrella"]
    assert at.carrying_terms([("backpack", False), ("backpack", True)]) == ["backpack", "large_bag"]
