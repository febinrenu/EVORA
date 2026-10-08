"""Questions about actions the system cannot recognise are answered with who was there, never with a claimed action."""
import pytest
from contracts.models import QueryPlan, Referent, Target

from evora.query.actions import unsupported_action
from evora.query.compose import ACTION_NOTE, compose_checked
from tests.query.test_compose import evidence, plan, t
from tests.query.test_router import FakeGateway, collect, make_router, of, red_car_crossing, ws  # noqa: F401

# ---- what counts as an action we cannot see ------------------------------------------------------------------------------

UNSUPPORTED = [
    ("Did a person put something down on 9 March between 10:10 and 10:15?", "putting something down"),
    ("did someone pick the bag up", "picking something up"),
    ("did a person drop a bag", "dropping something"),
    ("did anyone get out of a car", "getting out of a vehicle"),
    ("did a person get into the van", "getting into a vehicle"),
    ("did a person exit the car", "getting out of a vehicle"),
    ("did a man open the car door", "opening or closing something"),
    ("a person closed the gate", "opening or closing something"),
    ("did a car turn left", "turning left"),
    ("was there a u-turn", "turning around"),
    ("were boxes loaded", "loading or unloading"),
    ("did someone hand over a package", "handing something over"),
    ("did someone give a bag to another person", "handing something over"),
    ("were two people talking", "talking"),
    ("did people hug", "greeting someone"),
    ("did anyone fight", "fighting"),
    ("did someone run", "running"),
    ("was anyone sitting", "sitting or lying down"),
    ("was someone smoking", "smoking"),
    ("was someone on the phone", "using a phone"),
    ("did he fall", "falling"),
]

SUPPORTED_OR_PLAIN = [
    "Was there a person on G300 between 10:10 and 10:11?",
    "did a person walk through the main gate",
    "Did a person walk out through a building entrance on 9 March between 10:10 and 10:15?",
    "Did a person carry something heavy on 9 March between 10:10 and 10:15?",
    "a person carrying a large bag",
    "did anyone enter the lobby",
    "did someone leave the building",
    "how many people loitered near the gate",
    "who stood near the door",
    "was a red car parked",
    "people at the loading bay",
    "anyone near the loading dock",
    "at the drop-off zone",
    "a person close to the gate",
    "people near the open gate",
    "during rush hour",
    "the place down the road",
    "give me the cars between 10 and 11",
    "did a car pass the main entrance",
]


@pytest.mark.parametrize("text,label", UNSUPPORTED)
def test_actions_without_a_recogniser_are_named(text, label):
    assert unsupported_action(text) == label


@pytest.mark.parametrize("text", SUPPORTED_OR_PLAIN)
def test_movements_we_support_and_everyday_phrases_are_not_actions(text):
    assert unsupported_action(text) is None


# ---- the answer -------------------------------------------------------------------------------------------------------------

def person_plan(intent="exists"):
    return plan(intent=intent, noun="person", embed="a photo of a person", attrs=(), place=None, action="any",
                phrase="between 10:10 and 10:15", cams=("cam_01",))


def test_with_people_there_it_lists_them_and_never_says_yes():
    evs = [evidence("ev_1", peak=t(10, 11, 2)), evidence("ev_2", peak=t(10, 12, 40))]
    out = compose_checked(person_plan(), evs, cameras=[], action="putting something down")
    assert out.verdict == "partial" and not out.text.startswith("Yes")
    assert out.text.startswith("I can't tell whether anyone was putting something down: " + ACTION_NOTE)
    assert "These are the people seen at the Main gate camera between 10:10 and 10:15:" in out.text
    assert "1 more sighting." in out.text
    assert [s.evidence for s in out.sentences if s.kind == "fact"] == [("ev_1",), ("ev_2",)], "every fact cites evidence"


def test_one_person_reads_in_the_singular():
    out = compose_checked(person_plan("list"), [evidence("ev_1")], cameras=[], action="talking")
    assert "This is the person seen" in out.text and out.verdict == "partial"


def test_with_nobody_there_the_negative_stands_with_the_note():
    out = compose_checked(person_plan(), [], cameras=[], action="putting something down")
    assert out.verdict == "no", "nobody there means nobody there did it"
    assert out.sentences[0].kind == "note" and "No person" in out.text


def test_a_count_stays_a_count_of_who_was_seen_but_is_partial():
    out = compose_checked(person_plan("count"), [evidence("ev_1"), evidence("ev_2")], cameras=[], count=2,
                          action="running")
    assert out.verdict == "partial" and out.count == 2 and "Counted 2" in out.text and out.text.startswith("I can't tell")


def test_a_path_question_is_left_alone():
    p = person_plan("path")
    plain = compose_checked(p, [], cameras=[])
    assert compose_checked(p, [], cameras=[], action="running").text == plain.text


def test_without_an_action_nothing_changes():
    evs = [evidence("ev_1")]
    assert compose_checked(person_plan(), evs, cameras=[]).text.startswith("Yes.")


# ---- through the router -----------------------------------------------------------------------------------------------------

CAR_PLAN = QueryPlan(
    intent="exists", targets=[Target(noun="car", cls=["car"], attributes=["red"], embed_text="a photo of a red car")],
    place=Referent(text="main gate", role="place"), action="any", time=None,
    unresolved=[Referent(text="main gate", role="place")],
)


@pytest.mark.asyncio
async def test_the_router_answers_an_action_question_honestly(ws):  # noqa: F811
    red_car_crossing(ws)
    router = make_router(ws, gateway=FakeGateway(plan=CAR_PLAN))
    events = await collect(router.answer("did a red car turn left at the main gate", "s1"))
    ans = of(events, "answer")[0]
    assert ans["verdict"] == "partial" and ans["text"].startswith("I can't tell whether anyone was turning left")
    assert ans["evidence"], "who was there is still shown, with its evidence"


@pytest.mark.asyncio
async def test_the_switch_restores_the_old_answer(ws):  # noqa: F811
    from evora.query.router import RouterConfig

    red_car_crossing(ws)
    router = make_router(ws, gateway=FakeGateway(plan=CAR_PLAN))
    router.cfg = RouterConfig(accept=0.4, honest_actions=False)
    events = await collect(router.answer("did a red car turn left at the main gate", "s1"))
    assert of(events, "answer")[0]["verdict"] == "yes"

