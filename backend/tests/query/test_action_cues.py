"""Estimates for actions without a recogniser, from how bags, people and vehicles moved (synthetic tracks)."""
import pytest
from contracts.models import QueryPlan, Target

from evora.query.action_cues import CueConfig, find_cues
from evora.query.router import RouterConfig
from tests.query.test_router import FakeGateway, collect, make_router, of
from tests.query.ws_helpers import E, Workspace


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.camera("cam_01", "Gate")
    yield w
    w.close()


def move(ws, tid, cls, t0, t1, start, end, size=(0.06, 0.2), step=0.25, cam="cam_01", crops=(E[2],)):
    """A track whose box centre goes in a straight line from `start` to `end` between t0 and t1."""
    ws.track(tid, cam, cls=cls, t0=t0, t1=t1, crops=crops)
    pts, t = [], t0
    while t <= t1 + 1e-9:
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        cx, cy = start[0] + f * (end[0] - start[0]), start[1] + f * (end[1] - start[1])
        pts.append((t, cx - size[0] / 2, cy - size[1] / 2, cx + size[0] / 2, cy + size[1] / 2))
        t += step
    _add_points(ws, tid, pts)


def path(ws, tid, cls, waypoints, size=(0.06, 0.2), step=0.25, cam="cam_01"):
    """A track through (t, cx, cy) waypoints, linear in between."""
    ws.track(tid, cam, cls=cls, t0=waypoints[0][0], t1=waypoints[-1][0], crops=(E[2],))
    pts = []
    for (ta, xa, ya), (tb, xb, yb) in zip(waypoints, waypoints[1:], strict=False):
        t = ta
        while t < tb - 1e-9:
            f = (t - ta) / (tb - ta)
            cx, cy = xa + f * (xb - xa), ya + f * (yb - ya)
            pts.append((t, cx - size[0] / 2, cy - size[1] / 2, cx + size[0] / 2, cy + size[1] / 2))
            t += step
    t, cx, cy = waypoints[-1]
    pts.append((t, cx - size[0] / 2, cy - size[1] / 2, cx + size[0] / 2, cy + size[1] / 2))
    _add_points(ws, tid, pts)


def _add_points(ws, tid, pts):
    with ws.db.write() as c:
        c.executemany("INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,?,?,?,?,1)",
                      [(tid, *p) for p in pts])


def cues(ws, key, *tids):
    return find_cues(ws.db, key, [(t, "cam_01") for t in tids])


# ---- put down / drop -------------------------------------------------------------------------------------------------

def test_a_bag_left_where_a_person_was_points_at_that_person_and_moment(ws):
    path(ws, "p1", "person", [(1000, 0.2, 0.5), (1004, 0.32, 0.5), (1010, 0.7, 0.5)])
    path(ws, "b1", "backpack", [(1000, 0.2, 0.55), (1004, 0.32, 0.55), (1012, 0.32, 0.55)], size=(0.03, 0.04))
    got = cues(ws, "put_down", "p1")
    assert set(got) == {"p1"} and got["p1"].t == pytest.approx(1004.0, abs=0.75)
    assert "left a backpack" in got["p1"].why
    assert set(cues(ws, "drop", "p1")) == {"p1"}, "dropping uses the same cue"


def test_a_bag_carried_the_whole_time_is_not_put_down(ws):
    move(ws, "p1", "person", 1000, 1010, (0.2, 0.5), (0.7, 0.5))
    move(ws, "b1", "backpack", 1000, 1010, (0.2, 0.55), (0.7, 0.55), size=(0.03, 0.04))
    assert cues(ws, "put_down", "p1") == {}


def test_a_person_who_stays_by_the_bag_is_not_an_estimate(ws):
    path(ws, "p1", "person", [(1000, 0.3, 0.5), (1010, 0.31, 0.5)])
    path(ws, "b1", "suitcase", [(1000, 0.3, 0.55), (1010, 0.3, 0.55)], size=(0.03, 0.04))
    assert cues(ws, "put_down", "p1") == {}


def test_only_the_person_at_the_bag_is_cued(ws):
    path(ws, "p1", "person", [(1000, 0.2, 0.5), (1004, 0.32, 0.5), (1010, 0.7, 0.5)])
    move(ws, "p2", "person", 1000, 1010, (0.8, 0.5), (0.9, 0.5))
    path(ws, "b1", "handbag", [(1000, 0.2, 0.55), (1004, 0.32, 0.55), (1012, 0.32, 0.55)], size=(0.03, 0.04))
    assert set(cues(ws, "put_down", "p1", "p2")) == {"p1"}


# ---- pick up ---------------------------------------------------------------------------------------------------------

def test_a_still_bag_that_leaves_with_a_person_was_picked_up(ws):
    path(ws, "b1", "backpack", [(1000, 0.5, 0.55), (1006, 0.5, 0.55), (1010, 0.8, 0.55)], size=(0.03, 0.04))
    path(ws, "p1", "person", [(1002, 0.2, 0.5), (1006, 0.5, 0.5), (1010, 0.8, 0.5)])
    got = cues(ws, "pick_up", "p1")
    assert set(got) == {"p1"} and got["p1"].t == pytest.approx(1006.0, abs=0.75) and "picked up a backpack" in got["p1"].why


def test_a_still_bag_that_vanishes_while_a_person_is_on_it_is_a_weaker_estimate(ws):
    path(ws, "b1", "suitcase", [(1000, 0.5, 0.55), (1006, 0.5, 0.55)], size=(0.03, 0.04))
    path(ws, "p1", "person", [(1002, 0.2, 0.5), (1006, 0.5, 0.5), (1010, 0.8, 0.5)])
    got = cues(ws, "pick_up", "p1")
    assert got["p1"].score < 0.7 and "was gone" in got["p1"].why


def test_walking_past_a_bag_is_not_picking_it_up(ws):
    path(ws, "b1", "backpack", [(1000, 0.5, 0.55), (1012, 0.5, 0.55)], size=(0.03, 0.04))
    move(ws, "p1", "person", 1000, 1010, (0.2, 0.5), (0.8, 0.5))
    assert cues(ws, "pick_up", "p1") == {}


# ---- into / out of a vehicle -----------------------------------------------------------------------------------------

def stopped_car(ws, tid="c1", x=(0.4, 0.6), y=(0.4, 0.6), t=(1000, 1030)):
    move(ws, tid, "car", t[0], t[1], ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2), ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2),
         size=(x[1] - x[0], y[1] - y[0]))


def test_a_person_who_appears_next_to_a_stopped_car_got_out_of_it(ws):
    stopped_car(ws)
    move(ws, "p1", "person", 1005, 1015, (0.64, 0.5), (0.9, 0.5))
    got = cues(ws, "vehicle_out", "p1")
    assert got["p1"].t == pytest.approx(1005.0) and "appeared right next to a stopped car" in got["p1"].why
    assert cues(ws, "vehicle_in", "p1") == {}, "they walked away from it, they did not disappear into it"


def test_a_person_who_disappears_next_to_a_stopped_car_got_into_it(ws):
    stopped_car(ws)
    move(ws, "p1", "person", 1005, 1015, (0.9, 0.5), (0.64, 0.5))
    got = cues(ws, "vehicle_in", "p1")
    assert got["p1"].t == pytest.approx(1015.0) and "disappeared right next to" in got["p1"].why


def test_a_moving_car_is_not_got_out_of(ws):
    move(ws, "c1", "car", 1000, 1030, (0.1, 0.5), (0.9, 0.5), size=(0.2, 0.2))
    move(ws, "p1", "person", 1005, 1015, (0.4, 0.5), (0.5, 0.8))
    assert cues(ws, "vehicle_out", "p1") == {}


def test_coming_in_at_the_frame_edge_is_not_getting_out_of_a_car(ws):
    stopped_car(ws, x=(0.8, 0.99))
    move(ws, "p1", "person", 1005, 1015, (0.98, 0.5), (0.6, 0.5))
    assert cues(ws, "vehicle_out", "p1") == {}


def test_far_from_any_car_is_not_an_estimate(ws):
    stopped_car(ws)
    move(ws, "p1", "person", 1005, 1015, (0.1, 0.2), (0.2, 0.2))
    assert cues(ws, "vehicle_out", "p1") == {}


# ---- turns -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key,second_leg,expected", [
    ("turn_right", (0.6, 0.8), True),   # heading right, then down the screen: clockwise, a right turn on screen
    ("turn_left", (0.6, 0.2), True),    # heading right, then up the screen
    ("turn_left", (0.6, 0.8), False),
    ("turn_right", (0.6, 0.2), False),
])
def test_left_and_right_on_screen(ws, key, second_leg, expected):
    path(ws, "c1", "car", [(1000, 0.2, 0.5), (1005, 0.6, 0.5), (1010, *second_leg)])
    got = cues(ws, key, "c1")
    assert (set(got) == {"c1"}) is expected
    if expected:
        assert "on screen" in got["c1"].why and 1004 <= got["c1"].t <= 1006


def test_turning_around(ws):
    path(ws, "c1", "car", [(1000, 0.2, 0.5), (1005, 0.6, 0.5), (1006, 0.62, 0.52), (1011, 0.2, 0.55)])
    assert set(cues(ws, "u_turn", "c1")) == {"c1"}
    assert cues(ws, "turn_left", "c1") == {} and cues(ws, "turn_right", "c1") == {}, "a U-turn is not a left or right turn"


def test_a_gentle_curve_or_a_short_path_is_not_a_turn(ws):
    path(ws, "c1", "car", [(1000, 0.2, 0.5), (1005, 0.6, 0.5), (1010, 0.9, 0.58)])  # about 15 degrees
    path(ws, "c2", "car", [(1000, 0.5, 0.5), (1005, 0.52, 0.5), (1010, 0.52, 0.52)])  # barely moves
    assert cues(ws, "turn_right", "c1", "c2") == {}


# ---- running ---------------------------------------------------------------------------------------------------------

def test_fast_movement_for_a_body_size_is_running(ws):
    move(ws, "p1", "person", 1000, 1002, (0.1, 0.5), (0.9, 0.5))  # 0.4 a second at 0.2 high: 2 body heights a second
    move(ws, "p2", "person", 1000, 1010, (0.1, 0.5), (0.6, 0.5))  # 0.05 a second: a walk
    got = cues(ws, "run", "p1", "p2")
    assert set(got) == {"p1"} and "faster than walking" in got["p1"].why


def test_the_running_bar_is_a_setting(ws):
    move(ws, "p1", "person", 1000, 1002, (0.1, 0.5), (0.9, 0.5))
    assert find_cues(ws.db, "run", [("p1", "cam_01")], CueConfig(run_speed=10.0)) == {}


def test_an_action_without_a_cue_gives_nothing(ws):
    move(ws, "p1", "person", 1000, 1010, (0.1, 0.5), (0.9, 0.5))
    assert cues(ws, "talk", "p1") == {} and cues(ws, "put_down") == {}


# ---- in the answer ---------------------------------------------------------------------------------------------------

PERSON_PLAN = QueryPlan(
    intent="exists", targets=[Target(noun="person", cls=["person"], embed_text="a photo of a person")],
    action="any", camera_ids=["cam_01"],
)


def scene_with_a_left_bag(ws):
    move(ws, "p2", "person", 1000, 1010, (0.8, 0.3), (0.9, 0.3))  # someone else, nowhere near the bag
    path(ws, "p1", "person", [(1000, 0.2, 0.5), (1004, 0.32, 0.5), (1010, 0.7, 0.5)])
    path(ws, "b1", "backpack", [(1000, 0.2, 0.55), (1004, 0.32, 0.55), (1012, 0.32, 0.55)], size=(0.03, 0.04))


@pytest.mark.asyncio
async def test_the_answer_puts_the_estimated_person_first_at_the_estimated_moment(ws):
    scene_with_a_left_bag(ws)
    router = make_router(ws, gateway=FakeGateway(plan=PERSON_PLAN))
    router.cfg = RouterConfig(accept=0.4, show_unverified_actions=True)   # the estimates are an option now
    events = await collect(router.answer("did a person put something down", "s1"))
    ans = of(events, "answer")[0]
    first = ans["evidence"][0]
    assert first["track_id"] == "p1" and first["t_peak"] == pytest.approx(1004.0, abs=0.75)
    assert ans["verdict"] == "partial"
    assert "Most likely:" in ans["text"] and "left a backpack" in ans["text"] and "not a recognised action" in ans["text"]
    assert "1 other person" in ans["text"]
    assert any(w.startswith("action estimate: ") for w in first["why"])


@pytest.mark.asyncio
async def test_with_the_switch_off_the_plain_honest_answer_comes_back(ws):
    scene_with_a_left_bag(ws)
    router = make_router(ws, gateway=FakeGateway(plan=PERSON_PLAN))
    router.cfg = RouterConfig(accept=0.4, action_cues=False, show_unverified_actions=True)
    ans = of(await collect(router.answer("did a person put something down", "s1")), "answer")[0]
    assert ans["verdict"] == "partial" and "Most likely" not in ans["text"] and "These are the people" in ans["text"]


@pytest.mark.asyncio
async def test_no_movement_fits_so_the_plain_honest_answer_stays(ws):
    move(ws, "p1", "person", 1000, 1010, (0.1, 0.5), (0.3, 0.5))
    router = make_router(ws, gateway=FakeGateway(plan=PERSON_PLAN))
    router.cfg = RouterConfig(accept=0.4, show_unverified_actions=True)
    ans = of(await collect(router.answer("did a person put something down", "s1")), "answer")[0]
    assert "Most likely" not in ans["text"] and ans["text"].startswith("I can't tell whether anyone was putting")


# ---- perception's stored action events come first -----------------------------------------------------------------------

def stored_event(ws, tid, kind, t, **payload):
    ws.event(f"{tid}:{kind}:-:0", "cam_01", tid, kind, t, zone=None, **payload)


def test_a_stored_vehicle_event_is_used_instead_of_recomputing(ws):
    move(ws, "p1", "person", 1005, 1015, (0.1, 0.2), (0.2, 0.2))  # far from any car: geometry alone would say nothing
    stored_event(ws, "p1", "person_exits_vehicle", 1005.5, with_track="c9")
    got = cues(ws, "vehicle_out", "p1")
    assert got["p1"].t == pytest.approx(1005.5) and "appeared right next to a stopped vehicle" in got["p1"].why


def test_stored_turns_use_the_drivers_left_and_right(ws):
    path(ws, "c1", "car", [(1000, 0.2, 0.5), (1005, 0.6, 0.5), (1010, 0.6, 0.8)])  # a right turn on screen ...
    stored_event(ws, "c1", "vehicle_turn_left", 1005.0, turn_deg=-85)            # ... which perception read as left
    assert set(cues(ws, "turn_left", "c1")) == {"c1"} and cues(ws, "turn_right", "c1") == {}
    assert "about 85 degrees" in cues(ws, "turn_left", "c1")["c1"].why


def test_people_standing_together_back_up_talking_for_both_people(ws):
    move(ws, "p1", "person", 1000, 1010, (0.4, 0.5), (0.41, 0.5))
    move(ws, "p2", "person", 1000, 1010, (0.45, 0.5), (0.46, 0.5))
    stored_event(ws, "p1", "people_close", 1002.0, with_track="p2", seconds=6.0)
    got = cues(ws, "talk", "p1", "p2")
    assert set(got) == {"p1", "p2"} and "for 6.0 seconds" in got["p1"].why and got["p1"].score < 0.7
    assert set(cues(ws, "hand_over", "p1", "p2")) == {"p1", "p2"}


def test_an_index_with_action_events_but_none_of_this_kind_finds_nothing(ws):
    stopped_car(ws)
    move(ws, "p1", "person", 1005, 1015, (0.64, 0.5), (0.9, 0.5))  # geometry would call this getting out
    stored_event(ws, "c1", "vehicle_stop", 1000.0, seconds=30.0)  # perception ran its action pass on this camera
    assert cues(ws, "vehicle_out", "p1") == {}, "perception looked and found no exit: trust it"


def test_talking_without_stored_events_has_no_estimate(ws):
    move(ws, "p1", "person", 1000, 1010, (0.4, 0.5), (0.41, 0.5))
    move(ws, "p2", "person", 1000, 1010, (0.45, 0.5), (0.46, 0.5))
    assert cues(ws, "talk", "p1", "p2") == {}
