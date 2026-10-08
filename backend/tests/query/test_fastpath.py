from types import SimpleNamespace

import pytest
from contracts.models import QueryPlan

from evora.query.fastpath import parse

CAMS = [
    SimpleNamespace(id="cam_01", name="Gate"),
    SimpleNamespace(id="cam_02", name="Lobby"),
    SimpleNamespace(id="cam_03", name="Back Door"),
    SimpleNamespace(id="cam_04", name="Parking"),
]


def summary(plan: QueryPlan) -> dict:
    t = plan.targets[0]
    return {
        "intent": plan.intent, "noun": t.noun, "cls": t.cls, "attrs": t.attributes,
        "place": plan.place.text if plan.place else None, "action": plan.action, "cams": plan.camera_ids,
        "phrase": plan.time.phrase if plan.time else None,
        "after": plan.time.tod_after if plan.time else None,
        "before": plan.time.tod_before if plan.time else None,
        "limit": plan.limit, "unresolved": [r.text for r in plan.unresolved],
    }


def want(**kw) -> dict:
    base = {"attrs": [], "place": None, "action": "any", "cams": [], "phrase": None, "after": None,
            "before": None, "limit": 10, "unresolved": []}
    base.update(kw)
    return base


PERSON = {"noun": "person", "cls": ["person"]}
CAR = {"noun": "car", "cls": ["car"]}

CASES = [
    # appendix A example, verbatim
    ("did a red car pass through the main gate in the last hour?",
     want(intent="exists", **CAR, attrs=["red"], place="main gate", action="pass_through",
          phrase="in the last hour", unresolved=["main gate"])),
    # existence, many phrasings
    ("Was there a blue truck near the loading dock?",
     want(intent="exists", noun="truck", cls=["truck"], attrs=["blue"], place="loading dock",
          unresolved=["loading dock"])),
    ("is there anyone at the gate",
     want(intent="exists", **PERSON, cams=["cam_01"])),
    ("Was there anybody in the lobby today?",
     want(intent="exists", **PERSON, cams=["cam_02"], phrase="today")),
    ("did someone enter the server room after 8pm",
     want(intent="exists", **PERSON, place="server room", action="enter", phrase="after 8pm", after="20:00",
          unresolved=["server room"])),
    ("did anyone enter the lobby after 8 p.m.?",
     want(intent="exists", **PERSON, cams=["cam_02"], action="enter", phrase="after 8pm", after="20:00")),
    ("did a person leave the building before 9:30am",
     want(intent="exists", **PERSON, place="building", action="exit", phrase="before 9:30am", before="09:30",
          unresolved=["building"])),
    ("Did a white van park near the gate", None),  # "park" is not understood: let the planner handle it
    ("was a green bus at the gate yesterday",
     want(intent="exists", noun="bus", cls=["bus"], attrs=["green"], cams=["cam_01"], phrase="yesterday")),
    ("were there any red cars in the parking this morning",
     want(intent="exists", **CAR, attrs=["red"], cams=["cam_04"], phrase="this morning")),
    ("did a motorcycle pass the main gate",
     want(intent="exists", noun="motorcycle", cls=["motorcycle"], place="main gate", action="pass_through",
          unresolved=["main gate"])),
    ("was a black motorbike in the parking last night",
     want(intent="exists", noun="motorcycle", cls=["motorcycle"], attrs=["black"], cams=["cam_04"],
          phrase="last night")),
    ("did a bicycle go through the gate",
     want(intent="exists", noun="bicycle", cls=["bicycle"], cams=["cam_01"], action="pass_through")),
    ("was a silver sedan at the back door",
     want(intent="exists", **CAR, attrs=["grey"], cams=["cam_03"])),
    ("was there a gray hatchback near the lobby",
     want(intent="exists", **CAR, attrs=["grey"], cams=["cam_02"])),
    ("did a white SUV go past the main entrance",
     want(intent="exists", noun="suv", cls=["car"], attrs=["white", "suv"], place="main entrance",
          action="pass_through", unresolved=["main entrance"])),
    ("was there a yellow taxi at the gate",
     want(intent="exists", **CAR, attrs=["yellow"], cams=["cam_01"])),
    ("did an orange truck cross the gate between 9 and 10 am",
     want(intent="exists", noun="truck", cls=["truck"], attrs=["orange"], cams=["cam_01"], action="pass_through",
          phrase="between 9 and 10 am", after="09:00", before="10:00")),
    ("was there a purple vehicle at the gate",
     want(intent="exists", noun="vehicle", cls=["car", "motorcycle", "bus", "truck"], attrs=["purple"],
          cams=["cam_01"])),
    ("was there a woman at the lobby",
     want(intent="exists", noun="woman", cls=["person"], cams=["cam_02"])),
    ("did a man enter the back door",
     want(intent="exists", noun="man", cls=["person"], cams=["cam_03"], action="enter")),
    ("did a child go through the gate",
     want(intent="exists", noun="child", cls=["person"], cams=["cam_01"], action="pass_through")),
    ("was there a rickshaw near the gate",
     want(intent="exists", noun="rickshaw", cls=["car", "motorcycle"], attrs=["auto_rickshaw"], cams=["cam_01"])),
    ("did an auto-rickshaw enter the parking",
     want(intent="exists", noun="rickshaw", cls=["car", "motorcycle"], attrs=["auto_rickshaw"], cams=["cam_04"],
          action="enter")),
    # carrying
    ("did someone carrying a large bag go through the main entrance last night",
     want(intent="exists", **PERSON, attrs=["large_bag"], place="main entrance", action="pass_through",
          phrase="last night", unresolved=["main entrance"])),
    ("was there a person with a backpack at the gate",
     want(intent="exists", **PERSON, attrs=["backpack"], cams=["cam_01"])),
    ("did a man carrying an umbrella enter the lobby",
     want(intent="exists", noun="man", cls=["person"], attrs=["umbrella"], cams=["cam_02"], action="enter")),
    ("was a woman holding a suitcase at the gate",
     want(intent="exists", noun="woman", cls=["person"], attrs=["suitcase"], cams=["cam_01"])),
    ("did a person carrying a handbag leave the lobby",
     want(intent="exists", **PERSON, attrs=["handbag"], cams=["cam_02"], action="exit")),
    # tricky time forms
    ("did anyone enter the gate between 11 and 1 pm",
     want(intent="exists", **PERSON, cams=["cam_01"], action="enter", phrase="between 11 and 1 pm",
          after="11:00", before="13:00")),
    ("was there a car at the gate between 14:00 and 14:30",
     want(intent="exists", **CAR, cams=["cam_01"], phrase="between 14:00 and 14:30", after="14:00", before="14:30")),
    ("was there a car at the gate between 9am and 5pm",
     want(intent="exists", **CAR, cams=["cam_01"], phrase="between 9am and 5pm", after="09:00", before="17:00")),
    ("did a car pass the gate in the last 2 hours",
     want(intent="exists", **CAR, cams=["cam_01"], action="pass_through", phrase="in the last 2 hours")),
    ("did a car pass the gate in the past ten minutes",
     want(intent="exists", **CAR, cams=["cam_01"], action="pass_through", phrase="in the past ten minutes")),
    ("did a car pass the gate within the last day",
     want(intent="exists", **CAR, cams=["cam_01"], action="pass_through", phrase="within the last day")),
    ("was there a car at the gate yesterday evening",
     want(intent="exists", **CAR, cams=["cam_01"], phrase="yesterday evening")),
    ("was there a car at the gate tonight",
     want(intent="exists", **CAR, cams=["cam_01"], phrase="tonight")),
    ("was there a car at the gate yesterday after 8pm",
     want(intent="exists", **CAR, cams=["cam_01"], phrase="yesterday after 8pm", after="20:00")),
    ("was there a car at the gate after 20:00 and before 22:00",
     want(intent="exists", **CAR, cams=["cam_01"], phrase="after 20:00 before 22:00", after="20:00", before="22:00")),
    # politeness and punctuation
    ("Please tell me if a red car passed through the main gate.",
     want(intent="exists", **CAR, attrs=["red"], place="main gate", action="pass_through", unresolved=["main gate"])),
    ("Can you show me every person near the gate?",
     want(intent="list", **PERSON, cams=["cam_01"])),
    ("  DID   A   RED   CAR   PASS   THROUGH   THE   GATE  ",
     want(intent="exists", **CAR, attrs=["red"], cams=["cam_01"], action="pass_through")),
    # camera references
    ("was there a car at camera 2",
     want(intent="exists", **CAR, cams=["cam_02"])),
    ("was there a car at the Back Door camera",
     want(intent="exists", **CAR, cams=["cam_03"])),
    # list
    ("show me everyone who entered the lobby after 8pm",
     want(intent="list", **PERSON, cams=["cam_02"], action="enter", phrase="after 8pm", after="20:00")),
    ("show me all red cars at the gate",
     want(intent="list", **CAR, attrs=["red"], cams=["cam_01"])),
    ("find every white van near the loading dock",
     want(intent="list", noun="van", cls=["car", "truck"], attrs=["white", "van"], place="loading dock",
          unresolved=["loading dock"])),
    ("list all people in the parking today",
     want(intent="list", **PERSON, cams=["cam_04"], phrase="today")),
    ("give me any person who waited at the back door",
     want(intent="list", **PERSON, cams=["cam_03"], action="dwell")),
    # count
    ("how many people walked past the loading dock between 9 and 10 am?",
     want(intent="count", **PERSON, place="loading dock", action="pass_through", phrase="between 9 and 10 am",
          after="09:00", before="10:00", unresolved=["loading dock"])),
    ("how many cars entered the parking today",
     want(intent="count", **CAR, cams=["cam_04"], action="enter", phrase="today")),
    ("how many red cars went through the gate",
     want(intent="count", **CAR, attrs=["red"], cams=["cam_01"], action="pass_through")),
    ("how many people with backpacks were at the gate",
     want(intent="count", **PERSON, attrs=["backpack"], cams=["cam_01"])),
    # first / last
    ("when did the first person arrive at the back door today?",
     want(intent="first", **PERSON, cams=["cam_03"], action="appear", phrase="today", limit=1)),
    ("when was the first red car at the gate",
     want(intent="first", **CAR, attrs=["red"], cams=["cam_01"], limit=1)),
    ("when was a white van last seen near the loading dock?",
     want(intent="last", noun="van", cls=["car", "truck"], attrs=["white", "van"], place="loading dock",
          unresolved=["loading dock"], limit=1)),
    ("when did the last person leave the lobby",
     want(intent="last", **PERSON, cams=["cam_02"], action="exit", limit=1)),
    ("when did the last bus leave the parking yesterday",
     want(intent="last", noun="bus", cls=["bus"], cams=["cam_04"], action="exit", phrase="yesterday", limit=1)),
    # dwell
    ("did anyone loiter at the gate",
     want(intent="exists", **PERSON, cams=["cam_01"], action="dwell")),
    ("was someone standing near the back door tonight",
     want(intent="exists", **PERSON, cams=["cam_03"], action="dwell", phrase="tonight")),
]

# shapes the fast path must refuse so the language model handles them
REFUSED = [
    "where did the woman in the green jacket go after the lobby?",
    "what happened at the gate yesterday evening?",
    "alert me if anyone enters the server room after hours",
    "show me my car leaving the lot this morning",
    "did my car pass the gate",
    "was there a red car and a blue truck at the gate",
    "did a person wearing a red jacket enter the lobby",
    "was a person in red at the gate",
    "has anyone been loitering near the back door for more than a few minutes today?",
    "was there a car at the gate after hours",
    "was there a car at the gate last week",
    "was there a car at the gate on 3 October",
    "was there a car at the gate after 5",
    "was there a car at the gate between 10 and 9 am",
    "when did a red car arrive at the gate",
    "when was the gate opened",
    "did a spaceship pass the gate",
    "hello",
    "",
    "   ",
    "red car",
    "show me",
    "did a car",
    "did someone carrying something enter the lobby",
    "did a car pass the gate and the lobby",
    "was there a car from the lobby to the gate",
]


@pytest.mark.parametrize("text,expected", CASES, ids=[c[0][:60] for c in CASES])
def test_recognised_shapes(text, expected):
    plan = parse(text, CAMS)
    if expected is None:
        assert plan is None
        return
    assert plan is not None, text
    assert plan.source == "fastpath"
    assert summary(plan) == expected


@pytest.mark.parametrize("text", REFUSED)
def test_unrecognised_shapes_fall_through(text):
    assert parse(text, CAMS) is None


def test_table_is_large_enough():
    assert len(CASES) + len(REFUSED) >= 60


def test_every_plan_is_a_valid_queryplan_with_embed_text():
    for text, expected in CASES:
        if expected is None:
            continue
        plan = QueryPlan.model_validate(parse(text, CAMS).model_dump())
        assert all(t.embed_text.startswith("a photo of a") for t in plan.targets)
        assert plan.place is None or plan.place in plan.unresolved


def test_embed_text_forms():
    assert parse("did a red car pass the gate", CAMS).targets[0].embed_text == "a photo of a red car"
    assert parse("did someone carrying a large bag enter the lobby", CAMS).targets[0].embed_text == (
        "a photo of a person carrying a large bag"
    )
    assert parse("was a man in red", CAMS) is None
    assert parse("was a woman with a backpack at the gate", CAMS).targets[0].embed_text == (
        "a photo of a woman carrying a backpack"
    )


def test_place_matching_a_camera_name_becomes_a_camera_filter_not_a_referent():
    plan = parse("was there a car at the lobby", CAMS)
    assert plan.camera_ids == ["cam_02"] and plan.place is None and plan.unresolved == []


def test_unknown_place_without_cameras_is_unresolved():
    plan = parse("was there a car at the lobby")
    assert plan.camera_ids == [] and plan.place.text == "lobby" and plan.unresolved == [plan.place]


def test_time_is_never_resolved_to_epoch_by_the_parser():
    plan = parse("did a car pass the gate in the last hour", CAMS)
    assert plan.time.start is None and plan.time.end is None


def test_no_time_words_means_no_time_window():
    assert parse("did a car pass the gate", CAMS).time is None
