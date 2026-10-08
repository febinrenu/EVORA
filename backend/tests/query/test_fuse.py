import pytest

from evora.query.fuse import (
    BM25,
    Calibration,
    TrackSignals,
    aggregate_crops,
    asked_garments,
    attribute_score,
    blend,
    caption_supports_colour,
    explain_attributes,
    scene_support,
    squash_bm25,
    tokenize,
)


def test_calibration_is_monotonic_and_centred():
    cal = Calibration(midpoint=0.2, scale=0.05)
    assert cal(0.2) == pytest.approx(0.5)
    assert cal(0.05) < cal(0.15) < cal(0.2) < cal(0.3) < cal(0.5)
    assert 0.0 < cal(-5) < 1e-6 and 1 - 1e-6 < cal(5) < 1.0  # no overflow at the extremes


def test_aggregate_max_mean_beats_one_lucky_crop():
    lucky = aggregate_crops([0.40, 0.05, 0.04, 0.03])
    steady = aggregate_crops([0.30, 0.29, 0.28, 0.27])
    assert steady > lucky  # a single high crop does not carry the track
    assert aggregate_crops([0.4, 0.1], unit="frame") == 0.4  # frame-level: the best crop alone
    assert aggregate_crops([0.3]) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        aggregate_crops([])


def test_attribute_score_colours_carrying_and_vehicle_type():
    person = {"upper_color": "red", "color_conf": 0.9, "carrying": ["backpack"], "is_ir": False}
    assert attribute_score(["red"], person) == pytest.approx(0.9)
    assert attribute_score(["blue"], person) == 0.0
    assert attribute_score(["red", "backpack"], person) == pytest.approx(0.95)
    assert attribute_score(["red", "umbrella"], person) == pytest.approx(0.45)
    car = {"color": "white", "color_conf": 0.8, "vehicle_type": "suv"}
    assert attribute_score(["white", "suv"], car) == pytest.approx(0.9)
    assert attribute_score(["van"], car) == 0.0


def test_attribute_score_is_neutral_when_it_cannot_judge():
    assert attribute_score([], {"color": "red"}) is None            # nothing asked
    assert attribute_score(["red"], {}) is None                      # attributes not computed yet
    assert attribute_score(["red"], {"color": "red", "is_ir": True}) is None  # infrared: colour is no evidence
    assert attribute_score(["red"], {"carrying": ["backpack"]}) is None       # no colour stored
    assert attribute_score(["van"], {"color": "red", "color_conf": 1.0}) is None  # no vehicle type stored
    # infrared still judges non-colour attributes
    assert attribute_score(["red", "backpack"], {"is_ir": True, "carrying": ["backpack"], "color": "red"}) == 1.0


def test_explain_attributes_lists_only_matches():
    attrs = {"upper_color": "red", "color_conf": 0.92, "carrying": ["large_bag"], "vehicle_type": "van"}
    assert explain_attributes(["red", "blue", "large_bag", "van", "suv"], attrs) == [
        "colour red 0.92", "carrying large bag", "vehicle type van"]
    assert explain_attributes(["red"], {"color": "red", "is_ir": True}) == []


def test_tokenize_splits_underscores():
    assert tokenize("A person carrying a Large_Bag!") == ["a", "person", "carrying", "a", "large", "bag"]


def test_bm25_ranks_by_term_match_and_rarity():
    docs = {"t1": "a man in a red jacket walks", "t2": "a woman in a blue coat", "t3": "a man in a red jacket and red hat",
            "t4": "an empty corridor"}
    scores = BM25(docs).scores("red jacket")
    assert set(scores) == {"t1", "t3"} and scores["t1"] > 0
    assert BM25(docs).scores("zebra") == {}
    assert BM25({}).scores("red") == {}
    assert BM25(docs).scores("man")["t1"] > BM25(docs).scores("man").get("t2", 0)


def test_squash_bm25_is_bounded_and_monotonic():
    assert squash_bm25(0) == 0.0
    assert 0 < squash_bm25(1) < squash_bm25(5) < squash_bm25(50) < 1.0


def test_blend_renormalises_over_available_signals():
    weights = {"image": 0.5, "attributes": 0.3, "caption": 0.1, "scene": 0.1}
    full = TrackSignals("t", image=0.8, attributes=1.0, caption=0.5, scene=0.2)
    assert blend(full, weights) == pytest.approx(0.5 * 0.8 + 0.3 * 1.0 + 0.1 * 0.5 + 0.1 * 0.2)
    only_image = TrackSignals("t", image=0.8)
    assert blend(only_image, weights) == pytest.approx(0.8)  # not dragged down by absent signals
    assert blend(TrackSignals("t"), weights) == 0.0
    assert blend(TrackSignals("t", image=0.4, attributes=0.0), weights) == pytest.approx((0.5 * 0.4) / 0.8)


def test_scene_support_needs_same_camera_and_overlap():
    scenes = [("cam_01", 10.0, 0.4), ("cam_01", 20.0, 0.9), ("cam_02", 10.0, 0.99)]
    assert scene_support("cam_01", 8.0, 12.0, scenes) == 0.4
    assert scene_support("cam_01", 8.0, 25.0, scenes) == 0.9
    assert scene_support("cam_01", 30.0, 40.0, scenes) is None
    assert scene_support("cam_01", 12.5, 15.0, scenes, pad_s=1.0) is None


def test_attribute_score_reads_each_colour_slot_and_treats_unknown_slots_as_no_evidence():
    attrs = {"upper_color": "red", "upper_color_conf": 0.8, "lower_color": "blue", "lower_color_conf": 0.7,
             "color": "red", "color_conf": 0.8}
    assert attribute_score(["blue"], attrs) == pytest.approx(0.7)       # trousers count, with their own confidence
    assert attribute_score(["red"], attrs) == pytest.approx(0.8)
    assert attribute_score(["green"], attrs) == 0.0                     # a stored colour that differs is a mismatch
    # a slot left unknown is not a mismatch, and a track with nothing known is not penalised
    partial = {"upper_color": "red", "upper_color_conf": 0.8, "lower_color_conf": 0.3, "colour_unsure": ["lower"]}
    assert attribute_score(["red"], partial) == pytest.approx(0.8)
    unknown = {"upper_color_conf": 0.2, "lower_color_conf": 0.2, "colour_unsure": ["upper", "lower"], "carrying": []}
    assert attribute_score(["red"], unknown) is None
    assert explain_attributes(["blue"], attrs) == ["colour blue 0.70"]


def test_caption_colour_must_belong_to_the_garment_asked_for():
    jacket = asked_garments("a photo of a person wearing a red jacket")
    assert jacket == {"jacket"}
    assert caption_supports_colour("a person in a red long sleeved jacket", ["red"], jacket)
    assert not caption_supports_colour("a person in a pink top and dark blue trousers carrying a red bag", ["red"], jacket)
    assert not caption_supports_colour("a person in a red top", ["red"], jacket)       # red, but not the jacket asked for
    assert not caption_supports_colour("a person in a blue jacket and red hat", ["red"], jacket)
    # no garment asked: a colour on something carried is still not the person's colour
    assert caption_supports_colour("a person in a red coat", ["red"], set())
    assert not caption_supports_colour("a person carrying a red bag", ["red"], set())


def test_a_brown_shirt_is_found_in_a_caption_that_says_brown_top():
    shirt = asked_garments("a person wearing a brown shirt")
    assert caption_supports_colour("a person in a brown top and beige trousers carrying a black bag", ["brown"], shirt)
    # brown trousers are not a brown shirt
    assert not caption_supports_colour("a person in a black top and brown trousers", ["brown"], shirt)
