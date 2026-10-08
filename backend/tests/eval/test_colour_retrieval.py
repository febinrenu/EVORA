import pytest
from eval import colour_retrieval as cr


def person(tid, cam, upper, lower):
    return {"id": tid, "camera": cam, "cls": "person", "kind": "person", "label_upper": upper, "label_lower": lower}


def car(tid, cam, colour, cls="car"):
    return {"id": tid, "camera": cam, "cls": cls, "kind": "vehicle", "label_color": colour}


def test_a_track_is_scored_only_when_its_labelled_slots_are_known():
    row = person("a", "cam_01", "red", "black")
    assert cr.judge(row, "red") is True and cr.judge(row, "black") is True       # either slot counts
    assert cr.judge(row, "blue") is False                                          # both known, neither matches
    assert cr.judge(person("b", "cam_01", "grey", "unsure"), "blue") is None       # a slot is unsure: cannot say
    assert cr.judge(person("b", "cam_01", "grey", "unsure"), "grey") is True       # a match is a match
    assert cr.judge(car("c", "cam_08", "white"), "white") is True and cr.judge(car("c", "cam_08", "white"), "red") is False
    assert cr.judge(car("d", "cam_08", None), "white") is None


def test_queries_need_a_scoreable_positive_and_negative_and_keep_the_camera_split():
    items = [person("a", "cam_05", "red", "black"), person("b", "cam_05", "grey", "black"),
             person("c", "cam_05", "grey", "unsure"), person("d", "cam_02", "blue", "blue")]
    queries = cr.build_queries(items)
    by = {(q["camera"], q["colour"]): q for q in queries}
    assert ("cam_05", "red") in by and by[("cam_05", "red")]["positives"] == ["a"]
    assert by[("cam_05", "red")]["judged"] == ["a", "b"]           # c has an unsure slot and a colour that is not red
    assert by[("cam_05", "red")]["split"] == "dev"
    assert ("cam_05", "black") not in by          # every scoreable track has it: nothing to rank against
    assert not [q for q in queries if q["camera"] == "cam_02"]      # a single track cannot be ranked
    vehicles = cr.build_queries([car("x", "cam_08", "white"), car("y", "cam_08", "red", cls="truck")])
    assert vehicles[0]["classes"] == ["car", "truck"]


def test_rank_ignores_tracks_without_a_label():
    judged, pos = {"a", "b", "c"}, {"c"}
    assert cr.first_positive_rank(["u1", "a", "u2", "b", "c"], pos, judged) == 3     # unlabelled ones are skipped
    assert cr.first_positive_rank(["c", "a"], pos, judged) == 1
    assert cr.first_positive_rank(["a", "b"], pos, judged) is None                    # the positive was not retrieved


def test_chance_is_exact_for_a_random_order():
    hit1, rr = cr.chance(2, 1)
    assert hit1 == pytest.approx(0.5) and rr == pytest.approx(0.75)           # 1/2 * 1 + 1/2 * 1/2
    hit1, rr = cr.chance(3, 3)
    assert hit1 == 1.0 and rr == 1.0
    assert cr.chance(4, 1)[1] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)


def test_summary_averages_over_queries_and_counts_a_miss_as_zero():
    rows = [{"attributes_on": 1, "attributes_off": 2, "chance_hit1": 0.5, "chance_rr": 0.75},
            {"attributes_on": None, "attributes_off": 1, "chance_hit1": 0.25, "chance_rr": 0.5}]
    out = cr.summarise(rows)
    assert out["n"] == 2
    assert out["attributes_on"] == {"hit@1": 0.5, "mrr": 0.5}
    assert out["attributes_off"] == {"hit@1": 0.5, "mrr": 0.75}
    assert out["chance"] == {"hit@1": 0.375, "mrr": 0.625}
