"""Action events from trajectories: built from synthetic paths so every cue has a known answer."""
import numpy as np

from evora.perception.actions import (
    Traj,
    action_rows,
    people_close_events,
    person_vehicle_events,
    vehicle_events,
)
from evora.perception.settings import IngestSettings

ST = IngestSettings()


def make(tid, cls, pts, w=0.1, h=0.06, dt=0.25):
    """Trajectory from foot points; box width w and height h."""
    pts = np.asarray(pts, dtype=float)
    t = np.arange(len(pts)) * dt
    box = np.stack([pts[:, 0] - w / 2, pts[:, 1] - h, pts[:, 0] + w / 2, pts[:, 1]], axis=1)
    return Traj(tid, cls, t, box)


def line(p0, p1, n):
    return [(p0[0] + (p1[0] - p0[0]) * i / (n - 1), p0[1] + (p1[1] - p0[1]) * i / (n - 1)) for i in range(n)]


def kinds(events):
    return [e[0] for e in events]


def test_a_car_that_waits_then_drives_off_has_a_start_and_one_that_arrives_and_parks_has_a_stop():
    wait_then_go = make("c1", "car", [(0.3, 0.6)] * 12 + line((0.3, 0.6), (0.7, 0.6), 24))
    assert "vehicle_start" in kinds(vehicle_events(wait_then_go, ST))
    arrive_and_park = make("c2", "car", line((0.1, 0.6), (0.5, 0.6), 24) + [(0.5, 0.6)] * 16)
    ev = vehicle_events(arrive_and_park, ST)
    assert "vehicle_stop" in kinds(ev) and "vehicle_start" not in kinds(ev)


def test_a_parked_car_with_a_little_jitter_has_no_events():
    rng = np.random.default_rng(0)
    pts = [(0.5 + rng.normal(0, 0.001), 0.6 + rng.normal(0, 0.001)) for _ in range(80)]
    assert vehicle_events(make("c3", "car", pts), ST) == []


def test_left_and_right_turns_as_the_driver_sees_them():
    # driving up the picture (away from the camera), then to the left side of the picture: a left turn
    left = make("l", "car", line((0.6, 0.9), (0.6, 0.5), 16) + line((0.6, 0.5), (0.2, 0.5), 16))
    assert "vehicle_turn_left" in kinds(vehicle_events(left, ST))
    right = make("r", "car", line((0.6, 0.9), (0.6, 0.5), 16) + line((0.6, 0.5), (1.0, 0.5), 16))
    assert "vehicle_turn_right" in kinds(vehicle_events(right, ST))
    # driving towards the camera and turning to the picture's right is the driver's right-hand... left turn
    toward = make("t", "car", line((0.5, 0.3), (0.5, 0.7), 16) + line((0.5, 0.7), (0.9, 0.7), 16))
    assert "vehicle_turn_left" in kinds(vehicle_events(toward, ST))


def test_a_u_turn_and_a_straight_drive():
    arc = [(0.5 + 0.15 * np.sin(a), 0.5 - 0.15 * np.cos(a)) for a in np.linspace(0, np.pi, 24)]
    u = make("u", "car", line((0.5, 0.9), (0.5, 0.65), 12) + arc + line((0.5, 0.5), (0.5, 0.9), 12))
    assert "vehicle_u_turn" in kinds(vehicle_events(u, ST))
    straight = make("s", "car", line((0.1, 0.6), (0.9, 0.6), 40))
    assert not {"vehicle_u_turn", "vehicle_turn_left", "vehicle_turn_right"} & set(kinds(vehicle_events(straight, ST)))


def test_reversing_back_along_the_same_line_after_a_stop():
    pts = line((0.2, 0.6), (0.6, 0.6), 20) + [(0.6, 0.6)] * 12 + line((0.6, 0.6), (0.3, 0.6), 16)
    assert "vehicle_reverse" in kinds(vehicle_events(make("rev", "car", pts), ST))
    forward = line((0.2, 0.6), (0.6, 0.6), 20) + [(0.6, 0.6)] * 12 + line((0.6, 0.6), (0.9, 0.6), 16)
    assert "vehicle_reverse" not in kinds(vehicle_events(make("fwd", "car", forward), ST))


def test_a_person_who_appears_beside_a_parked_car_got_out_of_it_and_one_from_the_edge_did_not():
    car = make("car", "car", [(0.5, 0.6)] * 60, w=0.2, h=0.12)
    out = make("p1", "person", line((0.62, 0.6), (0.8, 0.7), 12), w=0.03, h=0.1)
    ev = person_vehicle_events([out], [car], ST)
    assert [(e[0], e[1], e[3]["with_track"]) for e in ev] == [("person_exits_vehicle", "p1", "car")]
    from_edge = make("p2", "person", line((0.01, 0.6), (0.15, 0.6), 12), w=0.03, h=0.1)
    assert person_vehicle_events([from_edge], [car], ST) == []


def test_a_person_walking_up_to_a_parked_car_and_vanishing_got_in():
    car = make("car", "car", [(0.5, 0.6)] * 60, w=0.2, h=0.12)
    walker = make("p3", "person", line((0.9, 0.7), (0.62, 0.6), 12), w=0.03, h=0.1)
    ev = person_vehicle_events([walker], [car], ST)
    assert [e[0] for e in ev] == ["person_enters_vehicle"]
    moving_car = make("mc", "car", line((0.1, 0.6), (0.9, 0.6), 60), w=0.2, h=0.12)
    assert person_vehicle_events([walker], [moving_car], ST) == []        # the car is not standing still


def test_two_people_standing_close_for_a_while_are_talking_and_passers_by_are_not():
    a = make("a", "person", [(0.5, 0.6)] * 40, w=0.04, h=0.12)
    b = make("b", "person", [(0.53, 0.6)] * 40, w=0.04, h=0.12)
    ev = people_close_events([a, b], ST)
    assert len(ev) == 1 and ev[0][2]["with_track"] == "b"
    far = make("c", "person", [(0.9, 0.6)] * 40, w=0.04, h=0.12)
    assert people_close_events([a, far], ST) == []
    walking = make("d", "person", line((0.2, 0.6), (0.8, 0.6), 40), w=0.04, h=0.12)
    walking2 = make("e", "person", line((0.22, 0.6), (0.82, 0.6), 40), w=0.04, h=0.12)
    assert people_close_events([walking, walking2], ST) == []             # side by side but walking


def test_rows_have_the_events_table_shape_and_stable_ids():
    car = make("cam_01:t1", "car", [(0.3, 0.6)] * 12 + line((0.3, 0.6), (0.7, 0.6), 24))
    rows = action_rows("cam_01", [car], ST)
    assert rows and all(len(r) == 7 and r[1] == "cam_01" and r[4] is None for r in rows)
    assert rows[0][0] == "cam_01:t1:vehicle_start:-:0"
    assert action_rows("cam_01", [car], ST) == rows
