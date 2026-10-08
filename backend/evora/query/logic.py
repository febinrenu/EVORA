"""Temporal and spatial logic over candidate tracks and stored events.

Pure functions: no database, no models. Retrieval hands in scored candidates and the
events recorded for them; this module decides which ones satisfy the plan's action,
zone and time constraints and where in time each match happened (the line-crossing
frame, not the whole track), then orders and counts them.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

from contracts.models import TimeWindow, Zone

# action -> event kinds that satisfy it, per zone kind. "frame" means the whole camera view.
ACTION_EVENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "pass_through": {"line": ("cross_line",), "polygon": ("enter_zone", "exit_zone"), "frame": ()},
    "enter": {"line": ("cross_line",), "polygon": ("enter_zone",), "frame": ("appear",)},
    "exit": {"line": ("cross_line",), "polygon": ("exit_zone",), "frame": ("disappear",)},
    "dwell": {"line": (), "polygon": ("dwell",), "frame": ("dwell",)},
    "appear": {"line": (), "polygon": ("enter_zone",), "frame": ("appear",)},
}
# detected actions (keys of query/actions.py) -> the event kinds perception writes for them. A "yes" about one of these
# comes only from a matching event; an action that is not here is not recognised and cannot be answered with "yes".
ACTION_EVENT_KINDS: dict[str, tuple[str, ...]] = {
    "vehicle_out": ("person_exits_vehicle",),
    "vehicle_in": ("person_enters_vehicle",),
    "u_turn": ("vehicle_u_turn",),
    "turn_left": ("vehicle_turn_left",),
    "turn_right": ("vehicle_turn_right",),
    "reverse": ("vehicle_reverse",),
    "start": ("vehicle_start",),
    "stop": ("vehicle_stop",),
    "talk": ("people_close",),
}
# direction a line crossing must have for "enter" / "exit" (a_to_b is "inwards" by convention)
LINE_DIRECTION = {"enter": "a_to_b", "exit": "b_to_a"}
MAX_TOD_SCAN_DAYS = 8


@dataclass(frozen=True)
class TrackRec:
    id: str
    camera_id: str
    cls: str
    t_start: float
    t_end: float
    best_t: float | None = None
    global_id: str | None = None
    group_id: str | None = None      # fragments of one person within a camera, for counting (attrs.person_group)


@dataclass(frozen=True)
class EventRec:
    id: str
    camera_id: str
    track_id: str
    kind: str
    t: float
    zone_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> EventRec:
        row = dict(row)
        payload = row.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload or "{}")
            except json.JSONDecodeError:
                payload = {}
        return cls(row["id"], row["camera_id"], row["track_id"], row["kind"], row["t"], row.get("zone_id"), payload)


@dataclass(frozen=True)
class Candidate:
    track: TrackRec
    score: float
    why: tuple[str, ...] = ()


@dataclass(frozen=True)
class Match:
    track_id: str
    camera_id: str
    t_start: float
    t_peak: float  # when it happened: the crossing, the entry, the dwell start, else the best frame
    t_end: float
    score: float
    global_id: str | None = None
    event_id: str | None = None
    why: tuple[str, ...] = ()
    group_id: str | None = None


# ------------------------------------------------------------- time filters
def _seconds_of_day(ts: float, tz: tzinfo) -> float:
    dt = datetime.fromtimestamp(ts, tz)
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


def _parse_hhmm(value: str) -> int:
    hour, minute = value.split(":")[:2]  # a model may add seconds: 00:13:53
    return int(hour) * 60 + int(minute)


def tod_contains(ts: float, after: str | None, before: str | None, tz: tzinfo = UTC) -> bool:
    """Is the local time of day of `ts` inside [after, before]? Ranges may wrap midnight.

    The bounds are clock times to the minute and the comparison is to the second: "between 10:12 and 10:13" is
    10:12:00 to 10:13:00, not the whole of the 10:13 minute.
    """
    if not after and not before:
        return True
    sec = _seconds_of_day(ts, tz)
    lo = _parse_hhmm(after) * 60 if after else 0
    hi = _parse_hhmm(before) * 60 if before else 24 * 3600
    if lo <= hi:
        return lo <= sec <= hi
    return sec >= lo or sec <= hi  # "after 22:00 before 06:00"


def span_touches_tod(t0: float, t1: float, after: str | None, before: str | None, tz: tzinfo = UTC) -> bool:
    """Does any moment of [t0, t1] fall inside the time-of-day range?"""
    if not after and not before:
        return True
    if t1 < t0:
        t0, t1 = t1, t0
    first = datetime.fromtimestamp(t0, tz).date()
    last = datetime.fromtimestamp(t1, tz).date()
    if (last - first).days >= MAX_TOD_SCAN_DAYS:
        return True
    lo = _parse_hhmm(after) if after else 0
    hi = _parse_hhmm(before) if before else 24 * 60
    day = first
    while day <= last:
        midnight = datetime.combine(day, datetime.min.time(), tzinfo=tz)
        if lo <= hi:
            windows = [(lo, hi)]
        else:
            windows = [(0, hi), (lo, 24 * 60)]
        for a, b in windows:
            w0 = (midnight + timedelta(minutes=a)).timestamp()
            w1 = (midnight + timedelta(minutes=b)).timestamp()
            if t0 <= w1 and t1 >= w0:
                return True
        day += timedelta(days=1)
    return False


def instant_in_window(t: float, window: TimeWindow | None, tz: tzinfo = UTC) -> bool:
    if window is None:
        return True
    if window.start is not None and t < window.start:
        return False
    if window.end is not None and t > window.end:
        return False
    return tod_contains(t, window.tod_after, window.tod_before, tz)


def span_in_window(t0: float, t1: float, window: TimeWindow | None, tz: tzinfo = UTC) -> bool:
    if window is None:
        return True
    if window.start is not None and t1 < window.start:
        return False
    if window.end is not None and t0 > window.end:
        return False
    return span_touches_tod(max(t0, window.start or t0), min(t1, window.end or t1), window.tod_after,
                            window.tod_before, tz)


# ------------------------------------------------------------ action logic
def _direction_ok(event: EventRec, zone: Zone, action: str) -> bool:
    """A crossing must agree with both the zone's own direction and the action's (enter/exit)."""
    got = event.payload.get("direction")
    need = [d for d in (zone.direction if zone.direction != "any" else None, LINE_DIRECTION.get(action)) if d]
    return got is None or all(got == d for d in need)


def _default_match(cand: Candidate, t: float | None = None, event: EventRec | None = None,
                   extra: Iterable[str] = ()) -> Match:
    tr = cand.track
    if t is None:
        t = tr.best_t if tr.best_t is not None else (tr.t_start + tr.t_end) / 2
    return Match(tr.id, tr.camera_id, tr.t_start, t, tr.t_end, cand.score, tr.global_id,
                 event.id if event else None, (*cand.why, *extra), tr.group_id)


def apply_action(
    candidates: Sequence[Candidate],
    events: Iterable[EventRec],
    action: str,
    zones_by_camera: Mapping[str, Zone] | None = None,
    window: TimeWindow | None = None,
    tz: tzinfo = UTC,
    zone_required: bool = False,
) -> list[Match]:
    """Keep the candidates that satisfy `action`, with the moment it happened.

    `zones_by_camera` holds the zone each camera's place resolved to. With
    `zone_required` (the plan names a place) cameras without a zone cannot match.
    Action "any" only applies the time window to the track span.
    """
    zones = zones_by_camera or {}
    by_track: dict[str, list[EventRec]] = defaultdict(list)
    for ev in events:
        by_track[ev.track_id].append(ev)

    matches: list[Match] = []
    for cand in candidates:
        tr = cand.track
        zone = zones.get(tr.camera_id)
        if zone_required and zone is None:
            continue

        if action == "any":
            if span_in_window(tr.t_start, tr.t_end, window, tz):
                matches.append(_default_match(cand))
            continue

        kinds = ACTION_EVENTS.get(action)
        if kinds is None:
            continue
        evs = sorted(by_track.get(tr.id, ()), key=lambda e: e.t)
        if zone is None:
            # no place named: any event of the right kind on the camera counts
            wanted = {k for ks in kinds.values() for k in ks}
            relevant = [e for e in evs if e.kind in wanted]
        else:
            wanted = set(kinds[zone.kind])
            relevant = [e for e in evs if e.kind in wanted and (e.zone_id in (zone.id, None) if zone.kind == "frame"
                                                                  else e.zone_id == zone.id)]
            if zone.kind == "line":
                relevant = [e for e in relevant if _direction_ok(e, zone, action)]
            if action == "pass_through" and zone.kind == "polygon":
                kinds_seen = {e.kind for e in relevant}
                if not {"enter_zone", "exit_zone"} <= kinds_seen:
                    relevant = []  # went in but never out (or the reverse): not a pass-through
                else:
                    relevant = [e for e in relevant if e.kind == "enter_zone"][:1]
            if action == "pass_through" and zone.kind == "frame":
                if span_in_window(tr.t_start, tr.t_end, window, tz):
                    matches.append(_default_match(cand, extra=(f"present in {zone.id}",)))
                continue
            if action in ("enter", "exit", "dwell", "appear") and zone.kind == "frame" and not relevant:
                if action in ("enter", "appear") and span_in_window(tr.t_start, tr.t_end, window, tz):
                    matches.append(_default_match(cand, t=tr.t_start, extra=("appeared in frame",)))
                continue
        for ev in relevant:
            if instant_in_window(ev.t, window, tz):
                matches.append(_default_match(cand, t=ev.t, event=ev, extra=(f"{ev.kind} {ev.t:.0f}",)))
    return matches


# ---------------------------------------------------- ordering and counting
def best_per_track(matches: Iterable[Match]) -> list[Match]:
    """One match per track: the one with the highest score, then the earliest."""
    best: dict[str, Match] = {}
    for m in matches:
        cur = best.get(m.track_id)
        if cur is None or (m.score, -m.t_peak) > (cur.score, -cur.t_peak):
            best[m.track_id] = m
    return list(best.values())


def order_matches(matches: Iterable[Match], intent: str) -> list[Match]:
    """first -> earliest, last -> latest, everything else -> best score first."""
    items = list(matches)
    if intent == "first":
        return sorted(items, key=lambda m: (m.t_peak, -m.score))
    if intent == "last":
        return sorted(items, key=lambda m: (-m.t_peak, -m.score))
    return sorted(items, key=lambda m: (-m.score, m.t_peak))


def concurrency(
    times_by_track: Mapping[str, Sequence[float]], bin_s: float = 1.0, bridge_s: float = 2.0
) -> dict[str, int] | None:
    """How many tracks are in view at the same moment: {"typical", "peak", "seconds"}, or None with no points.

    Each track is present in every bin between its stored points, bridging gaps of up to `bridge_s` (a person hidden
    for a moment is still there). `typical` is the median over the bins where anything is in view: for a room it is
    the number of people in it, which separate-appearance counts overstate whenever one person's track is broken.
    """
    present: dict[int, int] = {}
    for times in times_by_track.values():
        ordered = sorted(times)
        if not ordered:
            continue
        bins: set[int] = set()
        previous = None
        for t in ordered:
            b = int(t // bin_s)
            bins.add(b)
            if previous is not None and 0 < t - previous <= bridge_s:
                bins.update(range(int(previous // bin_s), b + 1))
            previous = t
        for b in bins:
            present[b] = present.get(b, 0) + 1
    if not present:
        return None
    counts = sorted(present.values())
    mid = len(counts) // 2
    typical = counts[mid] if len(counts) % 2 else round((counts[mid - 1] + counts[mid]) / 2)
    return {"typical": int(typical), "peak": int(counts[-1]), "seconds": len(counts)}


def count_distinct(matches: Iterable[Match]) -> int:
    """Distinct identities: global ids where linked, else individual tracks."""
    return len({m.group_id or m.global_id or m.track_id for m in matches})
