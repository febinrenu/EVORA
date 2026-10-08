from datetime import datetime
from types import SimpleNamespace

import pytest
from contracts.models import Evidence, PathHop, QueryPlan, Referent, Target, TimeWindow

from evora.query.compose import (
    Sentence,
    UngroundedAnswer,
    clock,
    compose,
    compose_checked,
    make_notes,
    offset_label,
    pluralize,
    validate,
)
from evora.query.timeparse import parse_tz

IST = parse_tz("+05:30")


def t(h, m=0, s=0, day=9):
    return datetime(2026, 10, day, h, m, s, tzinfo=IST).timestamp()


def evidence(eid="ev_1", cam="cam_01", name="Main gate", peak=None, offset=723.0, score=0.91):
    peak = peak if peak is not None else t(9, 14, 3)
    return Evidence(id=eid, camera_id=cam, camera_name=name, t_start=peak - 3, t_end=peak + 3, t_peak=peak,
                    offset_s=offset, thumb_url=f"/api/media/thumb/{eid}.jpg", clip_url=f"/api/media/clip/{eid}.mp4",
                    score=score)


def plan(intent="exists", noun="car", embed="a photo of a red car", attrs=("red",), place="main gate",
         action="pass_through", phrase="in the last hour", cams=()):
    return QueryPlan(
        intent=intent,
        targets=[Target(noun=noun, cls=[noun], attributes=list(attrs), embed_text=embed)],
        place=Referent(text=place, role="place") if place else None,
        action=action,
        time=TimeWindow(phrase=phrase) if phrase else None,
        camera_ids=list(cams),
    )


SRC = {"cam_01": "gate.mp4"}
KW = {"tz": IST, "source_names": SRC, "reference_now": t(10)}


def test_yes_has_dual_timestamps_and_cites_the_evidence():
    out = compose(plan(), [evidence()], **KW)
    assert out.verdict == "yes"
    assert out.text == ("Yes. A red car passed through the main gate in the last hour: "
                        "09:14:03 on Main gate (12:03 into gate.mp4).")
    assert out.sentences[0].evidence == ("ev_1",)


def test_more_matches_are_mentioned_and_cited():
    out = compose(plan(), [evidence("a"), evidence("b"), evidence("c")], **KW)
    assert "2 more matches found." in out.text
    assert out.sentences[1].evidence == ("b", "c")
    assert "1 more match found." in compose(plan(), [evidence("a"), evidence("b")], **KW).text


def test_negative_with_nearest_miss():
    miss = evidence("ev_9", peak=t(9, 41, 10), offset=2470.0, score=0.31)
    out = compose(plan(), [], nearest_miss=miss, **KW)
    assert out.verdict == "no"
    assert out.text == ("No red car passed through the main gate in the last hour. "
                        "Closest: 09:41:10 on Main gate (41:10 into gate.mp4), match score 0.31.")
    assert out.sentences[0].kind == "negative" and out.sentences[1].evidence == ("ev_9",)


def test_negative_without_nearest_miss_stays_honest():
    out = compose(plan(), [], **KW)
    assert out.text == "No red car passed through the main gate in the last hour."
    assert out.verdict == "no"


def test_not_found_for_list_like_intents():
    for intent in ("list", "first", "last"):
        assert compose(plan(intent=intent), [], **KW).verdict == "not_found"


def test_list_found():
    out = compose(plan("list"), [evidence("a"), evidence("b")], **KW)
    assert out.verdict == "found"
    assert out.text.startswith("Found 2 matches for red car passing through the main gate in the last hour.")
    assert "Best match: 09:14:03 on Main gate" in out.text
    assert "Found 1 match for" in compose(plan("list"), [evidence()], **KW).text


def test_first_and_last():
    first = compose(plan("first", phrase=None), [evidence()], **KW)
    assert first.text == ("The first match for red car passing through the main gate was at "
                          "09:14:03 on Main gate (12:03 into gate.mp4).")
    assert compose(plan("last", phrase=None), [evidence()], **KW).text.startswith("The last match")


def test_count_zero_one_many():
    p = plan("count", noun="person", embed="a photo of a person", attrs=(), place="lobby", action="enter",
             phrase="after 8pm")
    many = compose(p, [evidence("a", peak=t(20, 5)), evidence("b", peak=t(20, 1))], count=4, **KW)
    assert many.verdict == "count" and many.count == 4
    assert many.text.startswith("Counted 4 matching people entering the lobby after 8pm.")
    assert many.sentences[1].evidence == ("b",) and "First at 20:01:00" in many.text  # earliest evidence
    assert "Counted 1 matching person entering" in compose(p, [evidence()], count=1, **KW).text
    zero = compose(p, [], count=0, **KW)
    assert zero.count == 0 and zero.text == "Counted 0 matching people entering the lobby after 8pm."
    assert zero.sentences[0].kind == "negative"


def test_count_defaults_to_the_evidence_length():
    assert compose(plan("count"), [evidence("a"), evidence("b")], **KW).count == 2


def test_path_lists_hops_in_order_with_their_evidence():
    hops = [PathHop(camera_id="cam_01", camera_name="Main gate", t_in=t(9, 14), t_out=t(9, 15), evidence_id="e1"),
            PathHop(camera_id="cam_02", camera_name="Lobby", t_in=t(9, 16, 20), t_out=t(9, 17), evidence_id="e2")]
    out = compose(plan("path", place=None, action="any", phrase=None), [evidence("e1"), evidence("e2")],
                  path=hops, **KW)
    assert out.text == "Path of red car: Main gate 09:14:00 → Lobby 09:16:20."
    assert out.sentences[0].evidence == ("e1", "e2")


def test_path_without_hops_is_not_found_or_falls_back_to_evidence():
    assert compose(plan("path"), [], **KW).verdict == "not_found"


def test_partial_when_results_are_incomplete():
    assert compose(plan(), [evidence()], partial=True, **KW).verdict == "partial"
    assert compose(plan("list"), [evidence()], partial=True, **KW).verdict == "partial"


def test_place_and_camera_phrasing():
    by_cam = plan(place=None, cams=("cam_02",), action="any", phrase="today")
    out = compose(by_cam, [], cameras=[SimpleNamespace(id="cam_02", name="Lobby", layers=["L1"], ir_fraction=None)],
                  **KW)
    assert out.text == "No red car was seen at the Lobby camera today."
    bare = compose(plan(place=None, action="enter", phrase=None), [], **KW)
    assert bare.text == "No red car entered."
    assert compose(plan(place="the parking", action="dwell"), [], **KW).text.startswith(
        "No red car stayed near the parking")


def test_dates_appear_only_when_needed():
    other_day = evidence(peak=t(9, 14, 3, day=8))
    assert "8 Oct 09:14:03" in compose(plan(), [other_day], **KW).text  # not the reference day
    assert "9 Oct" not in compose(plan(), [evidence()], **KW).text  # same day as reference_now
    mixed = compose(plan("list"), [evidence("a"), other_day], **KW).text
    assert "Best match: 9 Oct 09:14:03" in mixed  # evidence spans two days, so even the reference day is dated


def test_source_name_falls_back_to_camera_name():
    out = compose(plan(), [evidence()], tz=IST, reference_now=t(10))
    assert "(12:03 into Main gate)" in out.text


def test_notes_for_unindexed_and_infrared_cameras():
    cams = [SimpleNamespace(id="cam_01", name="Main gate", layers=["L0"], ir_fraction=0.8),
            SimpleNamespace(id="cam_02", name="Lobby", layers=["L0", "L1"], ir_fraction=0.0),
            SimpleNamespace(id="cam_03", name="Roof", layers=[], ir_fraction=None)]
    notes = make_notes(plan(cams=("cam_01", "cam_02")), cams, [])
    assert any("Main gate is still being indexed" in n for n in notes)
    assert any("Main gate has night or infrared footage" in n for n in notes)
    assert not any("Lobby" in n for n in notes) and not any("Roof" in n for n in notes)  # not part of the query
    colourless = plan(attrs=(), embed="a photo of a car", cams=("cam_01",))
    assert not any("infrared" in n for n in make_notes(colourless, cams, []))
    all_cams = make_notes(plan(cams=()), cams, [])
    assert any("Roof is still being indexed" in n for n in all_cams)  # no filter: every camera is in scope


def test_compose_returns_notes_with_the_answer():
    cams = [SimpleNamespace(id="cam_01", name="Main gate", layers=["L0"], ir_fraction=None)]
    assert compose(plan(), [evidence()], cameras=cams, **KW).notes


# ---------------------------------------------------------------- validator
def test_validate_accepts_grounded_and_exempts_negatives_and_notes():
    validate([Sentence("a", ("e1",)), Sentence("No cars.", (), "negative"), Sentence("n", (), "note")], {"e1"})


def test_validate_rejects_ungrounded_and_unknown_citations():
    with pytest.raises(UngroundedAnswer, match="cites no evidence"):
        validate([Sentence("A car did something.")], {"e1"})
    with pytest.raises(UngroundedAnswer, match="unknown evidence"):
        validate([Sentence("A car did something.", ("e9",))], {"e1"})


@pytest.mark.parametrize("intent", ["exists", "list", "count", "first", "last"])
def test_composed_answers_always_pass_the_validator(intent):
    miss = evidence("ev_miss", score=0.2)
    compose_checked(plan(intent), [evidence("a"), evidence("b")], **KW)
    compose_checked(plan(intent), [], nearest_miss=miss, **KW)
    compose_checked(plan(intent), [], **KW)


def test_compose_checked_requires_path_hops_to_cite_held_evidence():
    hops = [PathHop(camera_id="cam_01", camera_name="Main gate", t_in=t(9), t_out=t(9, 1), evidence_id="ev_1")]
    compose_checked(plan("path"), [evidence("ev_1")], path=hops, **KW)
    bad = [PathHop(camera_id="cam_01", camera_name="Main gate", t_in=t(9), t_out=t(9, 1), evidence_id="ghost")]
    with pytest.raises(UngroundedAnswer):
        compose_checked(plan("path"), [evidence("ev_1")], path=bad, **KW)


def test_helpers():
    assert offset_label(723.4) == "12:03" and offset_label(3723) == "1:02:03" and offset_label(-5) == "00:00"
    assert clock(t(9, 14, 3), IST) == "09:14:03" and clock(t(9, 14, 3), IST, True) == "9 Oct 09:14:03"
    names = ("person", "car", "bus", "van", "woman")
    assert [pluralize(n) for n in names] == ["people", "cars", "buses", "vans", "women"]


def counted(**kw):
    return compose(plan(intent="count", noun="person", attrs=(), place="the room", action="any", phrase=None),
                   [evidence("a"), evidence("b")], **{**KW, **kw})


def test_a_count_of_people_in_a_room_is_how_many_are_in_view_with_appearances_as_context():
    out = counted(count=38, concurrent={"cam_01": {"typical": 4, "peak": 6, "seconds": 150}}, appearances=38)
    assert out.verdict == "count" and out.count == 4
    assert out.text.startswith("About 4 people were in view of the room at the same time (up to 6 at once).")
    assert any("38 separate appearances" in n and "at the same time" in n for n in out.notes)


def test_one_person_and_no_peak_wording():
    out = counted(count=1, concurrent={"cam_01": {"typical": 1, "peak": 1, "seconds": 9}}, appearances=1)
    assert out.text == "About 1 person was in view of the room at the same time."
    assert not any("separate appearances" in n for n in out.notes)


def test_several_cameras_give_the_largest_number_and_say_why():
    cams = {"cam_01": {"typical": 4, "peak": 5, "seconds": 9}, "cam_02": {"typical": 3, "peak": 4, "seconds": 9}}
    out = counted(concurrent=cams, appearances=10)
    assert out.count == 4 and "Per camera:" in out.text
    assert any("may show the same place" in n for n in out.notes)


def test_a_colour_nobody_could_read_is_not_counted_as_zero():
    out = compose(plan(intent="count", noun="person", attrs=("red",), place=None, action="any", phrase=None), [],
                  count=0, unreadable=30, **KW)
    assert out.verdict == "partial" and out.count is None
    assert out.text.startswith("I can't tell how many people are wearing red: the colour could not be read for 30 of them")
    assert any("unknown, not absent" in n for n in out.notes)


def test_without_concurrency_the_old_count_is_unchanged():
    out = counted(count=3)
    assert out.text.startswith("Counted 3 matching people")


def test_a_mixed_per_camera_line_says_which_number_is_only_the_detectors():
    from evora.query.compose import compose_objects
    rows = [{"camera_id": "cam_01", "camera_name": "Gate", "typical": 5, "peak": 6, "least": 5, "frames": 12, "seen_in": 12,
             "breakdown": {}, "vision_counts": [5, 5, 6, 5], "detected": 2},
            {"camera_id": "cam_02", "camera_name": "Hall", "typical": 1, "peak": 2, "least": 1, "frames": 12, "seen_in": 8,
             "breakdown": {}}]
    out = compose_objects(plan(intent="count", noun="chair", attrs=("red",), place=None, action="any", phrase=None), rows,
                          [evidence()], noun="chair", colour="red")
    assert "Per camera: Gate 5, Hall 1 (detector only)." in out.text
    assert out.count == 5 and any("Counted by the local vision model in 4 frames (5, 5, 6, 5)" in n for n in out.notes)
