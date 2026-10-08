import asyncio
import json
from datetime import UTC, datetime

import pytest
from contracts.models import Zone

from evora.alerts import store
from evora.alerts.notify import Notifier
from evora.alerts.store import StandingRule
from evora.core import zones
from tests.alerts.conftest import T0


def watch(env, **kw) -> str:
    rule = StandingRule(
        targets=["person"], place="main gate", camera_ids=["cam_01"], zone_id="z_gate", events=["cross_line"],
        direction="a_to_b", cooldown_s=30.0, summary="Alert when a person enters main gate.", **kw,
    )
    sq = store.create_standing(env.db, "watch the gate", rule)
    env.engine.invalidate()
    return sq.id


@pytest.fixture()
def gate(env):
    env.line()
    env.remember_gate()
    env.track()
    return env


def test_a_matching_crossing_alerts_once_with_registered_evidence(gate):
    sq = watch(gate)
    alerts = gate.engine.evaluate(gate.event())
    assert len(alerts) == 1 and alerts[0].standing_query_id == sq
    ev = alerts[0].evidence
    assert (ev.camera_id, ev.track_id, ev.t_peak, ev.offset_s) == ("cam_01", "cam_01:t1", T0 + 10, 10.0)
    assert ev.bbox == (0.3, 0.4, 0.4, 0.8) and ev.thumb_url.endswith(f"{ev.id}.jpg")
    assert any("Alert when" in w for w in ev.why)
    from evora.evidence import store as evidence_store

    assert evidence_store.get(gate.db, ev.id).camera_id == "cam_01", "the media routes can render it"
    assert gate.engine.evaluate(gate.event()) == [], "the same event never alerts the same watch twice"
    assert len(store.list_alerts(gate.db)) == 1


@pytest.mark.parametrize(
    "event,why",
    [
        ({"direction": "b_to_a"}, "wrong direction"),
        ({"kind": "enter_zone"}, "wrong kind"),
        ({"zone": "z_other"}, "other zone"),
        ({"camera": "cam_02", "track": "cam_02:t9"}, "other camera"),
        ({"track": "cam_01:car"}, "wrong class"),
        ({"t": T0 + 3 * 3600 + 10}, "outside the hours"),
    ],
)
def test_events_that_do_not_fit_the_watch_are_ignored(gate, event, why):
    watch(gate, tod_after="09:00", tod_before="09:30")
    gate.track("cam_01:car", cls="car")
    gate.track("cam_02:t9", camera="cam_02")
    assert gate.engine.evaluate(gate.event(**event)) == [], why


def test_an_inactive_watch_is_silent_and_reactivation_works(gate):
    sq = watch(gate)
    store.set_active(gate.db, sq, False)
    gate.engine.invalidate()
    assert gate.engine.evaluate(gate.event()) == []
    store.set_active(gate.db, sq, True)
    gate.engine.invalidate()
    assert len(gate.engine.evaluate(gate.event())) == 1


def test_hours_that_wrap_midnight(gate):
    watch(gate, tod_after="22:00", tod_before="06:00")
    assert gate.engine.evaluate(gate.event("noon", t=T0 + 3 * 3600)) == []  # 12:00 is outside
    assert len(gate.engine.evaluate(gate.event("late", t=T0 + 14 * 3600))) == 1  # 23:00 is inside
    assert len(gate.engine.evaluate(gate.event("after_midnight", t=T0 + 15 * 3600))) == 1  # 00:00 next day is inside


def test_the_workspace_time_zone_decides_what_after_8pm_means(gate):
    gate.db.set_meta("tz", "Asia/Kolkata")
    watch(gate, tod_after="20:00")
    ist = datetime(2026, 10, 8, 20, 30, tzinfo=__import__("zoneinfo").ZoneInfo("Asia/Kolkata")).timestamp()
    assert len(gate.engine.evaluate(gate.event("evening", t=ist))) == 1
    assert gate.engine.evaluate(gate.event("morning", t=T0 + 10)) == []  # 14:30 in Kolkata


def test_cooldown_is_per_identity_and_follows_event_time(gate):
    watch(gate)
    gate.track("cam_01:t2")
    assert len(gate.engine.evaluate(gate.event("a", t=T0 + 10))) == 1
    assert gate.engine.evaluate(gate.event("b", t=T0 + 20)) == [], "same person again inside the cooldown"
    assert len(gate.engine.evaluate(gate.event("c", t=T0 + 20, track="cam_01:t2"))) == 1, "someone else is not suppressed"
    assert len(gate.engine.evaluate(gate.event("d", t=T0 + 60))) == 1, "after the cooldown"


def test_one_person_on_two_cameras_is_one_identity(gate):
    watch(gate)
    gate.track("cam_01:t5", global_id="g_1")
    gate.track("cam_01:t6", global_id="g_1")
    assert len(gate.engine.evaluate(gate.event("a", track="cam_01:t5", t=T0 + 10))) == 1
    assert gate.engine.evaluate(gate.event("b", track="cam_01:t6", t=T0 + 15)) == []


def test_correcting_the_place_moves_existing_watches(gate):
    watch(gate)
    zones.save(gate.db, Zone(id="z_new", camera_id="cam_02", kind="line", points=[(0.1, 0.5), (0.9, 0.5)]))
    gate.track("cam_02:t3", camera="cam_02")
    fact = gate.kb.find_exact("place", "main gate")[0]
    gate.kb.supersede(fact.id, {"camera_id": "cam_02", "zone_id": "z_new"})
    assert gate.engine.evaluate(gate.event("old")) == [], "the old gate no longer counts"
    assert len(gate.engine.evaluate(gate.event("new", zone="z_new", camera="cam_02", track="cam_02:t3"))) == 1


def test_a_zone_with_its_own_direction_is_respected(gate):
    zones.save(gate.db, Zone(id="z_gate", camera_id="cam_01", kind="line", points=[(0.1, 0.7), (0.9, 0.7)], direction="b_to_a"))
    watch(gate)
    assert gate.engine.evaluate(gate.event()) == [], "the zone only counts b_to_a crossings"


def test_camera_level_events_match_on_the_camera(gate):
    rule = StandingRule(targets=["person"], camera_ids=["cam_01"], events=["appear"], summary="Alert when a person appears.")
    store.create_standing(gate.db, "watch cam 1", rule)
    gate.engine.invalidate()
    assert len(gate.engine.evaluate(gate.event("ap", kind="appear", zone=None, direction=None))) == 1
    assert gate.engine.evaluate(gate.event("ap2", kind="appear", zone=None, direction=None, camera="cam_02")) == []


def test_alerts_are_announced_on_the_bus(gate):
    async def scenario():
        sub = gate.bus.subscribe()
        watch(gate)
        await asyncio.to_thread(gate.engine.evaluate, gate.event())
        return json.loads((await asyncio.wait_for(sub.queue.get(), 2))["data"])

    note = asyncio.run(scenario())
    assert note["kind"] == "alert" and note["historical"] is False and note["alert"]["camera_id"] == "cam_01"


# ---- recorded footage -----------------------------------------------------------------------------------------

def test_backfill_raises_alerts_in_time_order_and_is_repeatable(gate):
    watch(gate)
    gate.track("cam_01:t2")
    gate.event("late", t=T0 + 100, track="cam_01:t2")
    gate.event("early", t=T0 + 10)
    assert gate.engine.backfill("cam_01") == 2
    assert [a.t for a in reversed(store.list_alerts(gate.db))] == [T0 + 10, T0 + 100]
    assert gate.engine.backfill("cam_01") == 0, "a second pass adds nothing"
    assert all("found in earlier footage" in a.evidence.why for a in store.list_alerts(gate.db))


def test_a_new_watch_reports_what_already_happened_once(gate):
    gate.event("before")
    sq = watch(gate)
    assert gate.engine.backfill(only_rule=sq) == 1
    assert gate.engine.backfill(only_rule=sq) == 0


def test_acknowledging(gate):
    watch(gate)
    alert = gate.engine.evaluate(gate.event())[0]
    assert [a.id for a in store.list_alerts(gate.db, acknowledged=False)] == [alert.id]
    assert store.acknowledge(gate.db, alert.id).acknowledged is True
    assert store.list_alerts(gate.db, acknowledged=False) == []
    with pytest.raises(store.AlertNotFound):
        store.acknowledge(gate.db, "al_nope")


def test_alerts_survive_a_restart(tmp_path):
    from evora.core.db import close_all
    from tests.alerts.conftest import Env

    first = Env(tmp_path / "ws")
    first.line()
    first.remember_gate()
    first.track()
    watch(first)
    first.engine.evaluate(first.event())
    close_all()
    second = Env.__new__(Env)  # reopen the same workspace without recreating cameras
    from evora.core import workspace as wsmod
    from evora.core.db import open_db

    second.db = open_db(wsmod.get("alerts", tmp_path / "ws").db_path)
    assert len(store.list_alerts(second.db)) == 1 and len(store.list_standing(second.db)) == 1
    close_all()


# ---- phone push ---------------------------------------------------------------------------------------------------

class FakeGateway:
    def __init__(self):
        self.sent = []

    async def notify(self, topic, title, message):
        self.sent.append((topic, title, message))


def notifier(gateway, topic="secret-topic", onprem=False):
    sent = []
    return Notifier(gateway, lambda: onprem, submit=lambda coro: sent.append(asyncio.run(coro)), topic=lambda: topic), sent


def test_push_sends_text_only_through_the_gateway():
    gw = FakeGateway()
    n, _ = notifier(gw)
    assert n.push("evora alert", "Alert when a person enters main gate. (Gate, 20:14)") is True
    assert gw.sent == [("secret-topic", "evora alert", "Alert when a person enters main gate. (Gate, 20:14)")]


@pytest.mark.parametrize("kw", [{"topic": None}, {"onprem": True}])
def test_push_is_skipped_without_a_topic_and_in_onprem_mode(kw):
    gw = FakeGateway()
    n, _ = notifier(gw, **kw)
    assert n.push("t", "m") is False and gw.sent == []


def test_push_without_a_gateway_method_is_skipped_with_one_hint(caplog):
    import logging

    n, _ = notifier(object())
    with caplog.at_level(logging.INFO, logger="evora.alerts.notify"):
        assert n.push("t", "m") is False and n.push("t", "m") is False
    hint = "ntfy push skipped: the gateway has no notify method yet (requested from M3)"
    assert [r.message for r in caplog.records].count(hint) == 1
    assert "secret-topic" not in caplog.text


def test_engine_pushes_live_alerts_but_not_historical_ones(gate):
    gw = FakeGateway()
    n, _ = notifier(gw)
    gate.engine.notifier = n
    watch(gate)
    gate.track("cam_01:t2")
    gate.engine.evaluate(gate.event("live"), historical=False)
    gate.engine.evaluate(gate.event("old", track="cam_01:t2"), historical=True)
    assert len(gw.sent) == 1 and "Gate" in gw.sent[0][2] and "secret" not in gw.sent[0][2]
    assert datetime.fromtimestamp(T0 + 10, UTC).strftime("%H:%M") in gw.sent[0][2]
