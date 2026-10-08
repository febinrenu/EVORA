import pytest
from eval import colour_verification as cv


def test_the_check_is_scored_only_on_labelled_tracks_it_decided_on():
    checked = {"a": True, "b": False, "c": True, "d": None, "e": True, "f": False}
    counts = cv.verifier_counts(checked, positives={"a", "b"}, judged={"a", "b", "c", "d"})
    assert counts == {"tp": 1, "fn": 1, "tn": 0, "fp": 1, "undecided": 1, "unlabelled": 2}


def test_rates_report_each_side_with_its_n_and_none_when_a_side_is_empty():
    out = cv.rates({"tp": 3, "fn": 1, "tn": 0, "fp": 0, "undecided": 0, "unlabelled": 0})
    assert out == {"yes_on_positives": 0.75, "n_positives": 4, "no_on_negatives": None, "n_negatives": 0}


def test_labelled_view_ignores_unlabelled_evidence_and_reads_hit_and_precision():
    view = cv.labelled_view(["u1", "a", "b", "c"], positives={"b", "c"}, judged={"a", "b", "c"})
    assert view == {"n": 3, "hit1": False, "precision": pytest.approx(2 / 3)}
    assert cv.labelled_view(["u1"], {"a"}, {"a"}) == {"n": 0, "hit1": None, "precision": None}


def test_effect_compares_before_and_after_and_counts_queries_the_check_emptied():
    rows = [{"before": {"n": 3, "hit1": False, "precision": 0.3333}, "after": {"n": 1, "hit1": True, "precision": 1.0}},
            {"before": {"n": 2, "hit1": True, "precision": 0.5}, "after": {"n": 2, "hit1": True, "precision": 0.5}},
            {"before": {"n": 2, "hit1": False, "precision": 0.0}, "after": {"n": 0, "hit1": None, "precision": None}}]
    out = cv.effect(rows)
    assert out["n"] == 2 and out["n_queries"] == 3 and out["lost_all_labelled"] == 1
    assert out["hit1_before"] == 0.5 and out["hit1_after"] == 1.0
    assert out["precision_before"] == pytest.approx(0.4167, abs=1e-3) and out["precision_after"] == pytest.approx(0.75)


def test_questions_name_the_camera_and_the_colour():
    person = {"kind": "person", "colour": "red"}
    assert cv.question(person, "G419") == "Was there a person wearing red on G419?"
    assert cv.question({"kind": "vehicle", "colour": "white"}, "G424") == "Was there a white car on G424?"
