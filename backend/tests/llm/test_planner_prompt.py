from contracts.models import QueryPlan

from evora.llm.prompts import (
    EXAMPLE_CAMERAS,
    build_planner_messages,
    load_examples,
    planner_system_prompt,
)

ALLOWED_CLS = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase", "umbrella"}
EXAMPLE_CAM_IDS = {c["id"] for c in EXAMPLE_CAMERAS}


def test_examples_cover_every_intent():
    shots = load_examples()
    assert len(shots) == 16
    intents = {QueryPlan.model_validate(s["plan"]).intent for s in shots}
    assert intents == {"exists", "list", "count", "first", "last", "path", "describe", "standing"}


def test_every_example_validates_and_follows_the_rules():
    for shot in load_examples():
        plan = QueryPlan.model_validate(shot["plan"])
        assert plan.source == "llm"
        for target in plan.targets:
            assert set(target.cls) <= ALLOWED_CLS, shot["question"]
            assert target.embed_text
        assert set(plan.camera_ids) <= EXAMPLE_CAM_IDS
        if plan.place:  # a known place must be listed as unresolved for the memory check
            assert plan.place in plan.unresolved, shot["question"]
        if plan.time:
            assert plan.time.start is None and plan.time.end is None  # the server resolves times
            assert plan.time.phrase


def test_camera_names_become_camera_ids_not_places():
    by_question = {s["question"]: QueryPlan.model_validate(s["plan"]) for s in load_examples()}
    lobby = by_question["show me everyone who entered the lobby after 8pm"]
    assert lobby.camera_ids == ["cam_02"] and lobby.place is None and lobby.unresolved == []


def test_appendix_a_example_matches_the_plan_document():
    first = QueryPlan.model_validate(load_examples()[0]["plan"])
    assert first.intent == "exists" and first.action == "pass_through"
    assert first.place.text == "main gate" and first.time.phrase == "in the last hour"


def test_system_prefix_is_byte_identical_across_requests():
    a = build_planner_messages([{"id": "cam_01", "name": "Gate"}], "did a red car pass?")
    b = build_planner_messages([{"id": "cam_09", "name": "Dock"}, {"id": "cam_10", "name": "Roof"}], "who is there?")
    assert a[0]["content"] == b[0]["content"]
    assert a[0]["content"] is planner_system_prompt()
    assert a[1]["content"] != b[1]["content"]


def test_camera_list_and_question_are_only_in_the_user_message():
    msgs = build_planner_messages([{"id": "cam_77", "name": "Zebra", "extra": "dropped"}], "any wolves?")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "cam_77" in msgs[1]["content"] and "any wolves?" in msgs[1]["content"]
    assert "extra" not in msgs[1]["content"]
    assert "cam_77" not in msgs[0]["content"] and "any wolves?" not in msgs[0]["content"]


def test_system_prompt_carries_rules_shape_and_examples():
    text = planner_system_prompt()
    assert "Never invent camera ids" in text
    assert "PLAN SHAPE:" in text and '"unresolved"' in text and "$defs" not in text
    assert text.count("QUESTION:") == 16


def test_objects_the_tracker_does_not_follow_are_still_targets():
    text = planner_system_prompt()
    assert "is still a target" in text and "cls = []" in text
    shots = {s["question"]: QueryPlan.model_validate(s["plan"]) for s in load_examples()}
    chair = shots["how many chairs are there in the room?"]
    assert chair.intent == "count" and chair.targets[0].noun == "chair" and chair.targets[0].cls == []
    red = shots["how many red objects are there in the room?"].targets[0]
    assert red.noun == "object" and red.attributes == ["red"]
