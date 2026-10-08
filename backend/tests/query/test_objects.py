from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from contracts.models import Evidence, QueryPlan, Target

from evora.query import objects as ob
from evora.query.compose import compose_objects, validate


def target(noun, cls=(), attrs=()):
    return Target(noun=noun, cls=list(cls), attributes=list(attrs), embed_text=f"a photo of a {noun}")


def test_only_what_the_tracker_does_not_follow_goes_to_the_open_search():
    assert ob.labels_for(target("chair")) == ["chair"]
    assert ob.labels_for(target("chairs")) == ["chair"]                      # plurals are looked up as the singular
    assert ob.labels_for(target("carpet")) == ["carpet", "rug"]
    assert ob.labels_for(target("shelves")) == ["shelf"]
    assert ob.labels_for(target("object", attrs=["red"])) == list(ob.COMMON_OBJECTS)
    assert "jacket" not in ob.COMMON_OBJECTS and "person" not in ob.COMMON_OBJECTS
    for tracked in (target("person", ["person"]), target("people"), target("man"), target("car", ["car"]),
                    target("van"), target("backpack"), target("suv", ["car"])):
        assert ob.labels_for(tracked) is None, tracked.noun
    assert ob.labels_for(target("")) is None


def test_singular_forms():
    assert [ob.singular(w) for w in ("chairs", "boxes", "shelves", "bodies", "glass", "bus", "tv")] == [
        "chair", "box", "shelf", "body", "glass", "bus", "tv"]


def test_sampling_spreads_over_the_whole_sequence():
    assert ob.sample_evenly(list(range(10)), 4) == [0, 3, 6, 9]
    assert ob.sample_evenly([1, 2], 5) == [1, 2] and ob.sample_evenly([], 3) == [] and ob.sample_evenly([1], 0) == []
    assert ob.sample_evenly([1, 2, 3], 1) == [2]


def det(label="chair", conf=0.8, box=(0.1, 0.1, 0.3, 0.4), colours=None):
    return ob.Detection(label, conf, box, colours or {})


def test_two_labels_on_the_same_thing_count_once_but_neighbours_stay_separate():
    boxes = [det("chair", 0.9, (0.1, 0.1, 0.3, 0.4)), det("seat", 0.5, (0.11, 0.1, 0.3, 0.41)),
             det("chair", 0.7, (0.6, 0.1, 0.8, 0.4))]
    kept = ob.merge_duplicates(boxes)
    assert [(d.label, d.conf) for d in kept] == [("chair", 0.9), ("chair", 0.7)]


def frame_with(colour_bgr, size=(240, 320)):
    img = np.full((*size, 3), 90, np.uint8)
    img[60:180, 100:220] = colour_bgr
    return img


def test_colour_is_read_from_the_middle_of_the_box():
    box = (100 / 320, 60 / 240, 220 / 320, 180 / 240)
    assert ob.box_colours(frame_with((0, 0, 200)), box).get("red", 0) > 0.6
    blue = ob.box_colours(frame_with((200, 40, 0)), box)
    assert blue.get("blue", 0) > 0.6 and blue.get("red", 0) < 0.1
    assert ob.box_colours(frame_with((0, 0, 200)), (0.5, 0.5, 0.5, 0.5)) == {}      # an empty box has no colour


def survey_of(detections_per_frame, colour=None):
    frames = [ob.FrameResult(ob.FrameRef("cam_01", float(i), f"f{i}.jpg"), dets) for i, dets in enumerate(detections_per_frame)]
    return ob.Survey("cam_01", ["chair"], colour, frames)


def test_a_survey_reports_the_typical_count_and_its_range_not_a_total():
    sv = survey_of([[det()] * 3, [det()] * 5, [det()] * 4, [det()] * 4, []])
    assert (sv.typical, sv.peak, sv.least, sv.seen_in) == (4, 5, 0, 4)
    assert len(sv.matching(sv.best_frame())) == 5 and sv.best_frame().ref.path == "f1.jpg"
    assert survey_of([]).typical == 0 and survey_of([]).best_frame() is None


def test_colour_filter_keeps_boxes_that_are_at_least_a_quarter_that_colour():
    red, mixed, blue = det(colours={"red": 0.8}), det(colours={"red": 0.3, "grey": 0.7}), det(colours={"blue": 0.9})
    assert survey_of([[red, mixed, blue]], colour="red").counts() == [2]
    assert survey_of([[red, mixed, blue]], colour=None).counts() == [3]
    assert survey_of([[det(colours={"red": 0.2})]], colour="red").counts() == [0]


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def __call__(self, image, labels, conf):
        self.calls += 1
        return [SimpleNamespace(label=labels[0], conf=0.9, xyxy=(100 / 320, 60 / 240, 220 / 320, 180 / 240)),
                SimpleNamespace(label=labels[0], conf=0.5, xyxy=(0.0, 0.0, 0.0001, 0.0001))]   # tiny: dropped as noise


@pytest.mark.asyncio
async def test_the_surveyor_detects_once_per_frame_and_noun_and_reuses_it_for_a_colour_question(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    for i in range(3):
        cv2.imwrite(str(media / f"f{i}.jpg"), frame_with((0, 0, 200)))
    fake = FakeDetector()
    surveyor = ob.OpenObjectSurveyor(media, detect=fake)
    refs = [ob.FrameRef("cam_01", float(i), f"f{i}.jpg") for i in range(3)]
    plain = await surveyor.survey("cam_01", refs, ["chair"], None)
    assert plain.counts() == [1, 1, 1] and fake.calls == 3
    red = await surveyor.survey("cam_01", refs, ["chair"], "red")
    assert red.counts() == [1, 1, 1] and fake.calls == 3                       # cached: asking about colour is free
    blue = await surveyor.survey("cam_01", refs, ["chair"], "blue")
    assert blue.counts() == [0, 0, 0] and fake.calls == 3
    await surveyor.survey("cam_01", refs, ["table"], None)
    assert fake.calls == 6                                                      # another noun is another search
    missing = await surveyor.survey("cam_01", [ob.FrameRef("cam_01", 9.0, "gone.jpg")], ["chair"], None)
    assert missing.frames == []                                                 # an unreadable frame is skipped


def test_the_surveyor_says_why_it_cannot_run_when_the_detector_is_missing(tmp_path, monkeypatch):
    from evora.perception import openvocab

    def boom(*args, **kwargs):
        raise openvocab.OpenVocabUnavailable("no weights here")

    monkeypatch.setattr(openvocab, "OpenVocabDetector", boom)
    surveyor = ob.OpenObjectSurveyor(tmp_path)
    assert surveyor.ready() is False and "no weights here" in surveyor.unavailable


def object_plan(intent, noun="chair", colour=None):
    return QueryPlan(intent=intent, targets=[target(noun, attrs=[colour] if colour else [])], action="any")


SUMMARY = [{"camera_id": "cam_01", "camera_name": "Gate", "typical": 5, "peak": 7, "least": 4, "frames": 12, "seen_in": 12,
            "breakdown": {"chair": 7}}]


def evidence_for(eid="e1"):
    return Evidence(id=eid, camera_id="cam_01", camera_name="Gate", t_start=1.0, t_end=6.0, t_peak=3.0, offset_s=3.0,
                    bbox=(0.1, 0.1, 0.3, 0.3), thumb_url="t", clip_url="c", score=0.8)


def test_object_answers_name_the_count_range_and_the_limits_and_pass_the_validator():
    out = compose_objects(object_plan("count"), SUMMARY, [evidence_for()], noun="chair", colour=None)
    assert out.verdict == "count" and out.count == 5
    assert out.text.startswith("About 5 chairs were in view (between 4 and 7 depending on the frame).")
    assert any("open-vocabulary detector" in n and "not a total over time" in n for n in out.notes)
    validate(out.sentences, {"e1"})


def test_a_colour_question_reads_the_colour_and_says_how():
    out = compose_objects(object_plan("count", "chair", "red"), [{**SUMMARY[0], "typical": 2, "peak": 3, "least": 1}],
                          [evidence_for()], noun="chair", colour="red")
    assert out.text.startswith("About 2 red chairs were in view")
    assert any("middle of each detected object" in n for n in out.notes)


def test_generic_objects_list_what_they_were():
    out = compose_objects(object_plan("count", "object", "red"), [{**SUMMARY[0], "breakdown": {"chair": 3, "bag": 1}}],
                          [evidence_for()], noun="object", colour="red")
    assert "In the clearest frame: chair 3, bag 1." in out.text and "everyday objects" in out.notes[0]
    validate(out.sentences, {"e1"})


def test_nothing_found_is_a_negative_and_exists_says_yes_or_no():
    none = [{**SUMMARY[0], "typical": 0, "peak": 0, "least": 0, "seen_in": 0, "breakdown": {}}]
    out = compose_objects(object_plan("count"), none, [], noun="chair", colour=None)
    assert out.verdict == "count" and out.count == 0 and "No chairs were found" in out.text
    no = compose_objects(object_plan("exists", "plant"), none, [], noun="plant", colour=None)
    assert no.verdict == "no" and no.text == "No plant was found in the 12 frames looked at."
    yes = compose_objects(object_plan("exists", "carpet"), SUMMARY, [evidence_for()], noun="carpet", colour=None)
    assert yes.verdict == "yes" and yes.text.startswith("Yes. A carpet was in view:")
    assert "seen in 12 of 12 frames" in yes.text
    validate(yes.sentences, {"e1"})
    validate(no.sentences, set())


def test_several_cameras_give_the_largest_count_and_the_per_camera_line_passes_the_validator():
    two = [SUMMARY[0], {**SUMMARY[0], "camera_id": "cam_02", "camera_name": "Hall", "typical": 3}]
    out = compose_objects(object_plan("count"), two, [evidence_for()], noun="chair", colour=None)
    assert out.count == 5 and "Per camera: Gate 5, Hall 3." in out.text
    validate(out.sentences, {"e1"})


def test_colour_from_the_outline_ignores_what_is_behind_the_object():
    image = np.full((240, 320, 3), (200, 40, 0), np.uint8)               # a blue room ...
    image[100:140, 140:180] = (0, 0, 200)                                 # ... with a small red seat in the middle
    box = (100 / 320, 60 / 240, 220 / 320, 180 / 240)                      # a loose box around it
    seat = [(140 / 320, 100 / 240), (180 / 320, 100 / 240), (180 / 320, 140 / 240), (140 / 320, 140 / 240)]
    assert ob.box_colours(image, box, polygon=seat).get("red", 0) > 0.8     # only the object's own pixels
    loose = ob.box_colours(image, box)
    assert loose.get("blue", 0) > loose.get("red", 0)                       # the box is mostly background
    assert ob.box_colours(image, box, polygon=[(0.1, 0.1), (0.1, 0.1)]).get("blue", 0) > 0.5   # too few points: the box


def test_an_outline_that_is_too_small_to_read_falls_back_to_the_box():
    image = frame_with((0, 0, 200))
    tiny = [(0.5, 0.5), (0.5005, 0.5), (0.5005, 0.5005)]
    assert ob._outline_pixels(image, tiny) is None
