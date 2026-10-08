import numpy as np
import pytest

pytest.importorskip("cv2")

from evora.perception import colourmodel as cm  # noqa: E402

P = cm.ColourParams()


def lab(rgb):
    return cm.at._bgr_to_lab(np.array([[rgb[2], rgb[1], rgb[0]]], dtype=np.uint8))[0]


def patch(rgb, h=120, w=60, noise=5, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array(rgb[::-1], dtype=np.int16)
    return np.clip(base + rng.integers(-noise, noise + 1, (h, w, 3)), 0, 255).astype(np.uint8)


def test_the_survey_is_loaded_and_every_basic_term_is_represented():
    sv = cm.survey()
    assert len(sv.names) == 949
    assert set(sv.terms) == set(cm.TERMS)
    assert sv.terms[sv.names.index("dark blue")] == "blue" and sv.terms[sv.names.index("pale pink")] == "pink"
    assert sv.terms[sv.names.index("greenish blue")] == "blue"          # the last basic word is the head
    assert sv.terms[sv.names.index("reddish brown")] == "brown"


@pytest.mark.parametrize(
    "rgb, term",
    [((20, 30, 90), "blue"), ((20, 22, 26), "black"), ((60, 62, 66), "grey"), ((245, 245, 245), "white"),
     ((215, 190, 150), "brown"),
     ((110, 115, 40), "green"), ((240, 150, 190), "pink"), ((20, 70, 30), "green"), ((120, 180, 230), "blue"),
     ((240, 130, 20), "orange"), ((110, 70, 40), "brown"), ((200, 30, 30), "red"), ((240, 220, 40), "yellow"),
     ((130, 50, 160), "purple")],
)
def test_naming_of_known_colours(rgb, term):
    assert cm.name_lab(lab(rgb), P)[0] == term


def test_shades_follow_lightness_within_a_term():
    assert cm.name_lab(lab((20, 30, 90)), P)[2] == "dark"
    assert cm.name_lab(lab((120, 180, 230)), P)[2] == "light"
    assert cm.name_lab(lab((60, 62, 66)), P)[2] == "dark"             # dark grey
    assert cm.name_lab(lab((200, 30, 30)), P)[2] is None


def test_the_dark_floor_keeps_near_black_clothes_black_but_lets_strong_colour_through():
    assert cm.name_lab(lab((8, 8, 12)), P)[0] == "black"
    assert cm.name_lab(lab((0, 0, 70)), P)[0] == "blue"


def test_thresholds_are_parameters_not_constants():
    mid_grey = lab((120, 120, 120))
    assert cm.name_lab(mid_grey, P)[0] == "grey"
    assert cm.name_lab(mid_grey, P.with_overrides(black_l=70.0))[0] == "black"
    assert cm.name_lab(mid_grey, P.with_overrides(white_l=40.0))[0] == "white"


def test_calibration_maps_raw_confidence_to_observed_accuracy():
    p = P.with_overrides(calibration=((0.0, 0.1), (0.5, 0.4), (1.0, 0.95)))
    assert cm.calibrate(0.5, p) == pytest.approx(0.4) and cm.calibrate(0.75, p) == pytest.approx(0.675)
    assert cm.calibrate(0.7, P) == 0.7


def test_person_regions_use_the_mask_and_ignore_the_background():
    crop = patch((200, 40, 40), h=160, w=80)                              # a red top...
    crop[:, :16] = patch((20, 180, 40), h=160, w=16)                      # ...with a green wall behind the left edge
    mask = np.zeros(crop.shape[:2], dtype=bool)
    mask[10:150, 16:80] = True
    with_mask = cm.predict("person", crop, mask=mask, params=P)
    assert with_mask["upper"][0] == "red"
    pixels = cm.garment_pixels("person", crop, mask, P)["upper"]
    assert len(pixels) > 100 and pixels[:, 1].mean() < 120              # no green wall pixels


def test_skin_pixels_are_removed_when_asked():
    top = patch((30, 60, 160), h=40, w=40)                              # blue top
    skin = patch((224, 172, 140), h=40, w=40)                           # bare arms
    crop = np.vstack([top[:20], skin[:20], top[20:], skin[20:]])
    off = cm.garment_pixels("person", crop, None, P.with_overrides(use_mask=False, skin_filter=False, trim=0.0))["upper"]
    on = cm.garment_pixels("person", crop, None, P.with_overrides(use_mask=False, skin_filter=True, trim=0.0))["upper"]
    assert len(on) < len(off) and on[:, 0].mean() > off[:, 0].mean()    # remaining pixels are bluer (BGR: blue channel first)


def test_background_filter_without_a_mask_removes_border_coloured_pixels():
    crop = patch((90, 90, 90), h=120, w=60)
    crop[30:100, 12:48] = patch((200, 30, 30), h=70, w=36)
    kept = cm.garment_pixels("person", crop, None, P.with_overrides(use_mask=False, background_filter=True, trim=0.0))["upper"]
    assert (kept[:, 2] > 150).mean() > 0.9                              # almost only the red top is left


def test_vehicle_slot_and_tiny_regions_fall_back_safely():
    car = patch((230, 230, 235), h=80, w=120)
    assert cm.predict("vehicle", car, params=P)["color"][0] == "white"
    tiny = np.full((6, 4, 3), 120, dtype=np.uint8)
    out = cm.predict("person", tiny, params=P)
    assert set(out) == {"upper", "lower"}                              # no crash, a slot either way


def test_predict_detail_gives_a_readable_name():
    res = cm.predict_detail("person", patch((20, 30, 90), h=160, w=80), params=P)["upper"]
    assert res.term == "blue" and res.shade == "dark" and res.name == "dark blue" and 0.3 < res.conf <= 1.0


def test_defaults_can_be_overridden_by_the_calibration_file(tmp_path, monkeypatch):
    import json

    f = tmp_path / "colour_calibration.json"
    f.write_text(json.dumps({"black_l": 30.0, "calibration": [[0, 0], [1, 0.9]], "not_a_field": 1}))
    monkeypatch.setattr(cm, "CALIBRATION", f)
    cm.load_params.cache_clear()
    try:
        p = cm.load_params()
        assert p.black_l == 30.0 and p.calibration == ((0, 0), (1, 0.9)) and p.knn == 7
    finally:
        cm.load_params.cache_clear()
