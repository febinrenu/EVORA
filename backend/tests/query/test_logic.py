import json
import sqlite3
from datetime import datetime

from contracts.models import TimeWindow, Zone

from evora.query.logic import (
    Candidate,
    EventRec,
    TrackRec,
    apply_action,
    best_per_track,
    concurrency,
    count_distinct,
    instant_in_window,
    order_matches,
    span_in_window,
    span_touches_tod,
    tod_contains,
)
from evora.query.timeparse import parse_tz

IST = parse_tz("+05:30")


def ts(h, m=0, day=9):
    return datetime(2026, 10, day, h, m, tzinfo=IST).timestamp()


def track(tid, cam="cam_01", t0=None, t1=None, best=None, gid=None, cls="car"):
    return TrackRec(tid, cam, cls, ts(9) if t0 is None else t0, ts(9, 1) if t1 is None else t1, best, gid)


def cand(t, score=0.5):
    return Candidate(t, score, ("siglip 0.31",))


def ev(eid, tid, kind, t, zone="z1", cam="cam_01", **payload):
    return EventRec(eid, cam, tid, kind, t, zone, payload)


LINE = Zone(id="z1", camera_id="cam_01", kind="line", points=[(0.1, 0.7), (0.9, 0.7)])
POLY = Zone(id="z1", camera_id="cam_01", kind="polygon", points=[(0, 0), (1, 0), (1, 1)])
FRAME = Zone(id="z1", camera_id="cam_01", kind="frame")


# --------------------------------------------------------------- time of day
def test_tod_contains_simple_and_wrapping_ranges():
    assert tod_contains(ts(20, 30), "20:00", None, IST)
    assert not tod_contains(ts(19, 59), "20:00", None, IST)
    assert tod_contains(ts(9, 30), "09:00", "10:00", IST)
    assert not tod_contains(ts(10, 1), "09:00", "10:00", IST)
    assert tod_contains(ts(23), "22:00", "06:00", IST) and tod_contains(ts(5), "22:00", "06:00", IST)
    assert not tod_contains(ts(12), "22:00", "06:00", IST)
    assert tod_contains(ts(3), None, None, IST)


def test_tod_depends_on_the_zone_not_utc():
    # 20:30 IST is 15:00 UTC
    assert tod_contains(ts(20, 30), "20:00", None, IST)
    assert not tod_contains(ts(20, 30), "20:00", None, parse_tz("UTC"))


def test_span_touches_tod():
    assert span_touches_tod(ts(19, 50), ts(20, 10), "20:00", None, IST)  # ends inside
    assert not span_touches_tod(ts(18), ts(19), "20:00", "21:00", IST)
    assert span_touches_tod(ts(23, 50), ts(0, 10, day=10), "00:00", "01:00", IST)  # across midnight
    assert span_touches_tod(ts(23), ts(2, day=10), "22:00", "06:00", IST)
    assert span_touches_tod(ts(9), ts(9) + 20 * 86400, "03:00", "04:00", IST)  # multi-week span: always true


def test_windows():
    w = TimeWindow(start=ts(9), end=ts(10))
    assert instant_in_window(ts(9, 30), w, IST) and not instant_in_window(ts(10, 1), w, IST)
    assert instant_in_window(5.0, None, IST)
    assert span_in_window(ts(8, 59), ts(9, 1), w, IST)  # overlap counts
    assert not span_in_window(ts(10, 1), ts(10, 5), w, IST)
    tod = TimeWindow(tod_after="20:00")
    assert not span_in_window(ts(9), ts(10), tod, IST) and span_in_window(ts(9), ts(21), tod, IST)


# -------------------------------------------------------------- line zones
def test_pass_through_a_line_uses_the_crossing_time_not_the_track():
    t = track("t1", t0=ts(9, 10), t1=ts(9, 20), best=ts(9, 11))
    out = apply_action([cand(t)], [ev("e1", "t1", "cross_line", ts(9, 14, ) + 3)], "pass_through", {"cam_01": LINE},
                       zone_required=True)
    assert len(out) == 1
    m = out[0]
    assert m.t_peak == ts(9, 14) + 3 and m.event_id == "e1"
    assert (m.t_start, m.t_end) == (t.t_start, t.t_end)
    assert any(w.startswith("cross_line") for w in m.why) and "siglip 0.31" in m.why


def test_track_without_a_crossing_does_not_pass_through():
    out = apply_action([cand(track("t1"))], [], "pass_through", {"cam_01": LINE}, zone_required=True)
    assert out == []


def test_crossing_of_a_different_zone_is_ignored():
    out = apply_action([cand(track("t1"))], [ev("e1", "t1", "cross_line", ts(9), zone="other")], "pass_through",
                       {"cam_01": LINE}, zone_required=True)
    assert out == []


def test_each_crossing_is_its_own_match_and_best_per_track_collapses_them():
    evs = [ev("e1", "t1", "cross_line", ts(9, 1)), ev("e2", "t1", "cross_line", ts(9, 5))]
    out = apply_action([cand(track("t1"))], evs, "pass_through", {"cam_01": LINE}, zone_required=True)
    assert [m.event_id for m in out] == ["e1", "e2"]
    assert len(best_per_track(out)) == 1 and best_per_track(out)[0].event_id == "e1"  # ties: earliest


def test_enter_and_exit_on_a_line_follow_direction():
    evs = [ev("in", "t1", "cross_line", ts(9, 1), direction="a_to_b"),
           ev("out", "t1", "cross_line", ts(9, 5), direction="b_to_a")]
    c = [cand(track("t1"))]
    assert [m.event_id for m in apply_action(c, evs, "enter", {"cam_01": LINE}, zone_required=True)] == ["in"]
    assert [m.event_id for m in apply_action(c, evs, "exit", {"cam_01": LINE}, zone_required=True)] == ["out"]
    assert len(apply_action(c, evs, "pass_through", {"cam_01": LINE}, zone_required=True)) == 2


def test_zone_direction_restricts_pass_through_and_conflicts_with_the_action():
    one_way = Zone(id="z1", camera_id="cam_01", kind="line", points=[(0, 0.5), (1, 0.5)], direction="a_to_b")
    evs = [ev("in", "t1", "cross_line", ts(9, 1), direction="a_to_b"),
           ev("out", "t1", "cross_line", ts(9, 5), direction="b_to_a")]
    c = [cand(track("t1"))]
    assert [m.event_id for m in apply_action(c, evs, "pass_through", {"cam_01": one_way}, zone_required=True)] == ["in"]
    assert apply_action(c, evs, "exit", {"cam_01": one_way}, zone_required=True) == []  # asks b_to_a, zone is a_to_b


def test_crossing_with_unknown_direction_is_accepted():
    evs = [ev("e", "t1", "cross_line", ts(9, 1))]
    assert len(apply_action([cand(track("t1"))], evs, "enter", {"cam_01": LINE}, zone_required=True)) == 1


# ----------------------------------------------------------- polygon zones
def test_polygon_pass_through_needs_both_enter_and_exit():
    both = [ev("a", "t1", "enter_zone", ts(9, 1)), ev("b", "t1", "exit_zone", ts(9, 2))]
    only_in = [ev("a", "t1", "enter_zone", ts(9, 1))]
    c = [cand(track("t1"))]
    out = apply_action(c, both, "pass_through", {"cam_01": POLY}, zone_required=True)
    assert [m.event_id for m in out] == ["a"]
    assert apply_action(c, only_in, "pass_through", {"cam_01": POLY}, zone_required=True) == []


def test_polygon_enter_exit_dwell():
    evs = [ev("a", "t1", "enter_zone", ts(9, 1)), ev("b", "t1", "exit_zone", ts(9, 2)),
           ev("d", "t1", "dwell", ts(9, 1, ) + 30, seconds=45)]
    c = [cand(track("t1"))]
    z = {"cam_01": POLY}
    assert [m.event_id for m in apply_action(c, evs, "enter", z, zone_required=True)] == ["a"]
    assert [m.event_id for m in apply_action(c, evs, "exit", z, zone_required=True)] == ["b"]
    assert [m.event_id for m in apply_action(c, evs, "dwell", z, zone_required=True)] == ["d"]


def test_dwell_needs_a_zone_that_can_hold_it():
    evs = [ev("d", "t1", "dwell", ts(9, 1))]
    assert apply_action([cand(track("t1"))], evs, "dwell", {"cam_01": LINE}, zone_required=True) == []


# -------------------------------------------------------------- frame zones
def test_frame_zone_means_presence():
    out = apply_action([cand(track("t1", best=ts(9, 0) + 20))], [], "pass_through", {"cam_01": FRAME},
                       zone_required=True)
    assert len(out) == 1 and out[0].t_peak == ts(9, 0) + 20


def test_frame_zone_enter_uses_appear_event_or_track_start():
    c = [cand(track("t1", t0=ts(9, 3)))]
    with_event = apply_action(c, [ev("ap", "t1", "appear", ts(9, 3) + 1, zone=None)], "enter", {"cam_01": FRAME},
                              zone_required=True)
    assert with_event[0].event_id == "ap" and with_event[0].t_peak == ts(9, 3) + 1
    without = apply_action(c, [], "enter", {"cam_01": FRAME}, zone_required=True)
    assert without[0].t_peak == ts(9, 3)


# ----------------------------------------------------- cameras and no place
def test_place_required_excludes_cameras_without_a_zone():
    c = [cand(track("t1", cam="cam_01")), cand(track("t2", cam="cam_02"))]
    evs = [ev("e1", "t1", "cross_line", ts(9)), ev("e2", "t2", "cross_line", ts(9), cam="cam_02")]
    out = apply_action(c, evs, "pass_through", {"cam_01": LINE}, zone_required=True)
    assert [m.track_id for m in out] == ["t1"]


def test_no_place_means_any_event_of_the_right_kind_counts():
    c = [cand(track("t1")), cand(track("t2", cam="cam_02"))]
    evs = [ev("e1", "t1", "cross_line", ts(9)), ev("e2", "t2", "enter_zone", ts(9), cam="cam_02")]
    out = apply_action(c, evs, "pass_through", None, zone_required=False)
    assert sorted(m.track_id for m in out) == ["t1", "t2"]
    assert apply_action([cand(track("t3"))], [], "pass_through", None) == []


def test_action_any_only_applies_the_window():
    w = TimeWindow(start=ts(9, 30), end=ts(10))
    early = track("t1", t0=ts(9), t1=ts(9, 10))
    late = track("t2", t0=ts(9, 40), t1=ts(9, 50), best=ts(9, 45))
    out = apply_action([cand(early), cand(late)], [], "any", None, window=w, tz=IST)
    assert [m.track_id for m in out] == ["t2"] and out[0].t_peak == ts(9, 45)


def test_event_time_must_be_in_the_window_not_just_the_track():
    t = track("t1", t0=ts(9), t1=ts(10))
    evs = [ev("early", "t1", "cross_line", ts(9, 5)), ev("late", "t1", "cross_line", ts(9, 50))]
    w = TimeWindow(start=ts(9, 30), end=ts(10, 30))
    out = apply_action([cand(t)], evs, "pass_through", {"cam_01": LINE}, window=w, tz=IST, zone_required=True)
    assert [m.event_id for m in out] == ["late"]


def test_time_of_day_filter_applies_to_the_event():
    t = track("t1", t0=ts(19), t1=ts(21))
    evs = [ev("e1", "t1", "cross_line", ts(19, 30)), ev("e2", "t1", "cross_line", ts(20, 30))]
    w = TimeWindow(tod_after="20:00")
    out = apply_action([cand(t)], evs, "pass_through", {"cam_01": LINE}, window=w, tz=IST, zone_required=True)
    assert [m.event_id for m in out] == ["e2"]


def test_unknown_action_matches_nothing():
    assert apply_action([cand(track("t1"))], [ev("e", "t1", "cross_line", ts(9))], "teleport", {"cam_01": LINE}) == []


# ------------------------------------------------------ ordering and counts
def make_matches():
    evs = [ev("e1", "t1", "cross_line", ts(9, 10)), ev("e2", "t2", "cross_line", ts(9, 5)),
           ev("e3", "t3", "cross_line", ts(9, 20))]
    cands = [cand(track("t1", gid="g1"), 0.4), cand(track("t2", gid="g1"), 0.9), cand(track("t3"), 0.6)]
    return apply_action(cands, evs, "pass_through", {"cam_01": LINE}, zone_required=True)


def test_first_last_and_ranked_order():
    ms = make_matches()
    assert [m.event_id for m in order_matches(ms, "first")] == ["e2", "e1", "e3"]
    assert [m.event_id for m in order_matches(ms, "last")] == ["e3", "e1", "e2"]
    assert [m.event_id for m in order_matches(ms, "list")] == ["e2", "e3", "e1"]  # by score


def test_count_uses_global_ids_where_linked():
    ms = make_matches()
    assert len(ms) == 3 and count_distinct(ms) == 2  # t1 and t2 are the same person
    assert count_distinct([]) == 0


def test_event_from_sqlite_row_and_json_payload():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE events(id, camera_id, track_id, kind, zone_id, t, payload)")
    conn.execute("INSERT INTO events VALUES('e1','cam_01','t1','cross_line','z1',12.5,?)",
                 (json.dumps({"direction": "a_to_b"}),))
    conn.execute("INSERT INTO events VALUES('e2','cam_01','t1','appear',NULL,13.0,'not json')")
    rows = conn.execute("SELECT * FROM events ORDER BY t").fetchall()
    first, second = (EventRec.from_row(r) for r in rows)
    assert first.payload == {"direction": "a_to_b"} and first.zone_id == "z1" and first.t == 12.5
    assert second.payload == {} and second.zone_id is None


def test_a_time_of_day_range_ends_at_the_stated_clock_time_not_at_the_end_of_that_minute():
    def at(h, m, sec):
        return datetime(2026, 10, 9, h, m, sec, tzinfo=IST).timestamp()

    # "between 10:12 and 10:13" is one minute: 10:12:00 to 10:13:00
    assert tod_contains(at(10, 12, 0), "10:12", "10:13", IST) and tod_contains(at(10, 12, 59), "10:12", "10:13", IST)
    assert tod_contains(at(10, 13, 0), "10:12", "10:13", IST)  # the end instant itself is included
    assert not tod_contains(at(10, 13, 1), "10:12", "10:13", IST)  # but nothing after it
    assert not tod_contains(at(10, 11, 59), "10:12", "10:13", IST)
    assert tod_contains(at(5, 59, 59), "22:00", "06:00", IST) and not tod_contains(at(6, 0, 30), "22:00", "06:00", IST)
    assert tod_contains(at(23, 59, 59), "20:00", None, IST) and not tod_contains(at(19, 59, 59), "20:00", None, IST)
    assert tod_contains(at(0, 0, 0), None, "06:00", IST) and not tod_contains(at(6, 0, 1), None, "06:00", IST)


def test_a_time_of_day_with_seconds_does_not_crash():
    from evora.query.logic import tod_contains
    assert tod_contains(13 * 3600 + 53 * 60 + 10, "13:53:00", "13:54:00")
    assert not tod_contains(15 * 3600, "13:53:00", "13:54:00")


def test_concurrency_counts_who_is_in_view_at_once_not_how_many_tracks_exist():
    # four people stay for ten seconds; each one's track breaks in two with a one-second gap (two fragments each)
    people = {}
    for i in range(4):
        people[f"p{i}a"] = [k * 0.25 for k in range(0, 21)]            # 0-5 s
        people[f"p{i}b"] = [6 + k * 0.25 for k in range(0, 17)]        # 6-10 s
    out = concurrency(people)
    assert len(people) == 8                                               # eight tracks ...
    assert out["typical"] == 4 and out["peak"] == 4                        # ... four people, never more in view at once


def test_concurrency_bridges_short_hiding_and_not_long_absences():
    one = {"a": [0.0, 0.25, 0.5, 2.0, 2.25]}                 # hidden for 1.5 s: still there
    assert concurrency(one) == {"typical": 1, "peak": 1, "seconds": 3}
    long_gap = {"a": [0.0, 0.25, 10.0, 10.25]}               # gone for 10 s: two separate visits, never together
    assert concurrency(long_gap) == {"typical": 1, "peak": 1, "seconds": 2}
    assert concurrency({}) is None and concurrency({"a": []}) is None


def test_concurrency_median_is_robust_to_a_brief_crowd():
    times = {f"c{i}": [0.0, 0.25] for i in range(5)}         # five people together for one second only
    times["steady"] = [k * 0.25 for k in range(0, 41)]       # one person for ten seconds
    out = concurrency(times)
    assert out["peak"] == 6 and out["typical"] == 1


def test_a_count_uses_the_person_group_before_the_identity():
    from evora.query.logic import Match, count_distinct

    def m(tid, gid, group):
        return Match(tid, "cam_01", 0.0, 0.0, 1.0, 0.8, gid, None, (), group)

    matches = [m("a", "g1", "p1"), m("b", "g2", "p1"), m("c", "g3", None), m("d", None, None)]
    assert count_distinct(matches) == 3          # a and b are one person, c by identity, d by track
