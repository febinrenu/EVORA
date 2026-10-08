import sqlite3

import pytest
from eval import actions as act
from eval.queries import GroundTruthHit


def database(tmp_path, rows):
    path = tmp_path / "evora.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE events(id TEXT, camera_id TEXT, track_id TEXT, kind TEXT, zone_id TEXT, t REAL, payload TEXT)")
    conn.executemany("INSERT INTO events VALUES(?,?,?,?,NULL,?,'{}')",
                     [(f"e{i}", cam, f"t{i}", kind, t) for i, (cam, kind, t) in enumerate(rows)])
    conn.commit()
    conn.close()
    return path


def hit(cam, start, end):
    return GroundTruthHit(camera_id=cam, start=start, end=end)


def test_precision_and_recall_count_events_in_labelled_windows_and_windows_with_events(tmp_path):
    path = database(tmp_path, [("cam_01", "vehicle_stop", 105.0), ("cam_01", "vehicle_stop", 300.0),
                               ("cam_02", "vehicle_stop", 105.0), ("cam_01", "vehicle_start", 105.0)])
    labels = [hit("cam_01", 100.0, 110.0), hit("cam_01", 200.0, 210.0)]
    q = act.event_quality(path, ("vehicle_stop",), labels)
    assert q["events"] == 3 and q["labelled_windows"] == 2             # another kind of event is not counted
    assert q["precision"] == pytest.approx(1 / 3, abs=1e-3) and q["recall"] == pytest.approx(1 / 2)
    assert q["events_on_cameras_with_no_label"] == 1                    # the cam_02 event has nothing to be compared with


def test_slack_widens_a_window_and_the_gap_is_distance_to_the_nearest_one(tmp_path):
    path = database(tmp_path, [("cam_01", "vehicle_stop", 111.5), ("cam_01", "vehicle_stop", 140.0)])
    q = act.event_quality(path, ("vehicle_stop",), [hit("cam_01", 100.0, 110.0)])
    assert q["precision"] == 0.5                                        # 111.5 is within 2 s of the window, 140 is not
    assert q["median_seconds_from_a_labelled_window"] == 30.0           # upper median of 1.5 and 30
    none = act.event_quality(path, ("vehicle_reverse",), [hit("cam_01", 100.0, 110.0)])
    assert none["events"] == 0 and none["precision"] is None and none["recall"] == 0.0
    assert act.event_quality(path, ("vehicle_stop",), [])["recall"] is None


def test_the_three_settings_differ_only_in_how_actions_are_answered():
    assert set(act.SETTINGS) == {"plain", "estimates", "detected"}
    assert act.SETTINGS["plain"]["honest_actions"] is False
    assert act.SETTINGS["estimates"]["show_unverified_actions"] is True and act.SETTINGS["estimates"]["detected_actions"] is False
    assert act.SETTINGS["detected"] == {"honest_actions": True, "detected_actions": True, "show_unverified_actions": False}
    from dataclasses import fields

    from evora.query.router import RouterConfig
    names = {f.name for f in fields(RouterConfig)}
    assert all(set(s) <= names for s in act.SETTINGS.values())          # every switch the tool flips really exists


def test_the_table_has_a_row_per_setting_and_n_a_for_what_was_not_measured():
    metrics = {"hit@1": {"value": 0.5, "n": 4}, "abstain_rate": {"value": None, "n": 0}}
    results = {name: {"metrics": metrics, "queries": [{}] * 4} for name in act.SETTINGS}
    lines = act.table(results).splitlines()
    assert lines[0].startswith("| Setting | n | Hit@1") and len(lines) == 2 + 3
    assert lines[2].startswith("| plain | 4 | 0.50 |") and "n/a" in lines[2]
