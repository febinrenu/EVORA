"""Actions read from trajectories: vehicles starting, stopping, turning, reversing; people getting in and out of
vehicles; people standing close together. Everything comes from `track_points` (about 4 per second), so it can be
recomputed without decoding video, like the zone events.

Event kinds written to `events` (zone_id NULL, payload JSON):

* vehicle_start / vehicle_stop            a stationary vehicle begins to move / a moving one comes to rest
* vehicle_turn_left / vehicle_turn_right  heading changes by 40 to 140 degrees (left and right as the driver sees it)
* vehicle_u_turn                          heading changes by more than 140 degrees while moving
* vehicle_reverse                         after a stop it moves back along the way it came (a straight reversal)
* person_exits_vehicle                    a person track starts, away from the frame edge, next to a stationary vehicle
* person_enters_vehicle                   a person track ends, away from the frame edge, next to a stationary vehicle
* people_close                            two people stay within a body height of each other for a few seconds

These are cues, not recognition: a pedestrian who appears behind a parked car looks like someone getting out of it.
The payload says what the cue was (`with_track`, `seconds`, `turn_deg`) so an answer can show its evidence.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from evora.core.db import Database
from evora.perception.settings import IngestSettings

log = logging.getLogger("evora.perception.actions")

VEHICLES = {"car", "truck", "bus", "motorcycle"}
ACTION_KINDS = (
    "vehicle_start", "vehicle_stop", "vehicle_turn_left", "vehicle_turn_right", "vehicle_u_turn", "vehicle_reverse",
    "person_exits_vehicle", "person_enters_vehicle", "people_close",
)
EDGE = 0.03          # a track that starts or ends this close to the frame edge came from or went out of the picture


@dataclass
class Traj:
    id: str
    cls: str
    t: np.ndarray        # seconds, increasing
    box: np.ndarray      # (n, 4) x1, y1, x2, y2 normalised

    @property
    def foot(self) -> np.ndarray:
        return np.stack([(self.box[:, 0] + self.box[:, 2]) / 2.0, self.box[:, 3]], axis=1)

    @property
    def height(self) -> np.ndarray:
        return self.box[:, 3] - self.box[:, 1]


def speeds(tr: Traj, window_s: float = 1.0) -> np.ndarray:
    """Foot-point speed (frame widths per second) over about `window_s`, per sample."""
    f, t = tr.foot, tr.t
    out = np.zeros(len(t))
    for i in range(len(t)):
        lo = int(np.searchsorted(t, t[i] - window_s / 2.0, "left"))
        hi = min(len(t) - 1, int(np.searchsorted(t, t[i] + window_s / 2.0, "right")) - 1)
        if hi > lo and t[hi] > t[lo]:
            out[i] = float(np.linalg.norm(f[hi] - f[lo]) / (t[hi] - t[lo]))
    return out


def segments(tr: Traj, st: IngestSettings) -> list[tuple[str, int, int]]:
    """Alternating ('still'|'moving', first sample, last sample) with hysteresis and minimum durations."""
    v = speeds(tr)
    state = "moving" if v[0] > st.action_move_speed else "still"
    out: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(v)):
        new = state
        if state == "still" and v[i] > st.action_move_speed:
            new = "moving"
        elif state == "moving" and v[i] < st.action_still_speed:
            new = "still"
        if new != state:
            out.append((state, start, i - 1))
            state, start = new, i
    out.append((state, start, len(v) - 1))
    merged: list[tuple[str, int, int]] = []
    for seg in out:      # a pause or a stutter shorter than the minimum belongs to its neighbours
        dur = tr.t[seg[2]] - tr.t[seg[1]]
        if merged and dur < st.action_min_segment_s and seg[0] != merged[-1][0] and len(out) > 2:
            merged[-1] = (merged[-1][0], merged[-1][1], seg[2])
        elif merged and seg[0] == merged[-1][0]:
            merged[-1] = (seg[0], merged[-1][1], seg[2])
        else:
            merged.append(seg)
    return merged


def heading_deg(tr: Traj, a: int, b: int) -> float | None:
    """Direction of travel between two samples, in degrees (image coordinates, y down)."""
    d = tr.foot[b] - tr.foot[a]
    if float(np.linalg.norm(d)) < 1e-4:
        return None
    return math.degrees(math.atan2(d[1], d[0]))


def _diff(a: float, b: float) -> float:
    """Signed turn from heading a to heading b in (-180, 180]; negative is a left turn as the driver sees it."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return d


def vehicle_events(tr: Traj, st: IngestSettings) -> list[tuple[str, float, dict]]:
    if len(tr.t) < 6:
        return []
    segs = segments(tr, st)
    out: list[tuple[str, float, dict]] = []
    last_move_heading: float | None = None
    last_still_end: float | None = None
    for k, (state, a, b) in enumerate(segs):
        dur = float(tr.t[b] - tr.t[a])
        if state == "still":
            last_still_end = float(tr.t[b])
            if k > 0 and dur >= st.action_min_still_s:
                out.append(("vehicle_stop", float(tr.t[a]), {"seconds": round(dur, 1)}))
            continue
        if k > 0 and segs[k - 1][0] == "still" and tr.t[segs[k - 1][2]] - tr.t[segs[k - 1][1]] >= st.action_min_still_s:
            out.append(("vehicle_start", float(tr.t[a]), {}))
        path = float(np.sum(np.linalg.norm(np.diff(tr.foot[a:b + 1], axis=0), axis=1))) if b > a else 0.0
        if path < st.action_min_path or dur < st.action_min_segment_s:
            continue
        n = b - a + 1
        q = max(2, n // 4)
        h0, h1 = heading_deg(tr, a, a + q), heading_deg(tr, b - q, b)
        if h0 is not None and h1 is not None:
            turn = _diff(h0, h1)
            mid = float(tr.t[a + n // 2])
            if abs(turn) >= st.action_u_turn_deg:
                out.append(("vehicle_u_turn", mid, {"turn_deg": round(turn)}))
            elif abs(turn) >= st.action_turn_deg:
                out.append(("vehicle_turn_left" if turn < 0 else "vehicle_turn_right", mid, {"turn_deg": round(turn)}))
        h_all = heading_deg(tr, a, b)
        if (h_all is not None and last_move_heading is not None and last_still_end is not None
                and tr.t[a] - last_still_end < st.action_reverse_gap_s and abs(_diff(last_move_heading, h_all)) > 150.0
                and h0 is not None and h1 is not None and abs(_diff(h0, h1)) < st.action_turn_deg):
            out.append(("vehicle_reverse", float(tr.t[a]), {}))
        if h_all is not None:
            last_move_heading = h_all
    return out


def _box_at(tr: Traj, t: float) -> np.ndarray:
    return tr.box[int(np.argmin(np.abs(tr.t - t)))]


def _near_vehicle(person_foot: np.ndarray, vbox: np.ndarray, pad: float) -> bool:
    w, h = vbox[2] - vbox[0], vbox[3] - vbox[1]
    inside_x = vbox[0] - pad * w <= person_foot[0] <= vbox[2] + pad * w
    inside_y = vbox[1] - pad * h <= person_foot[1] <= vbox[3] + pad * h
    return inside_x and inside_y


def _still_at(tr: Traj, t: float, st: IngestSettings) -> bool:
    if t < tr.t[0] - 1.0 or t > tr.t[-1] + 1.0:
        return False
    v = speeds(tr)
    lo, hi = np.searchsorted(tr.t, t - 1.0, "left"), np.searchsorted(tr.t, t + 1.0, "right")
    window = v[lo:hi] if hi > lo else v[[int(np.argmin(np.abs(tr.t - t)))]]
    return bool(np.max(window) < st.action_move_speed)


def person_vehicle_events(people: list[Traj], vehicles: list[Traj], st: IngestSettings) -> list[tuple[str, str, float, dict]]:
    """(kind, person track id, time, payload) for people who start or end next to a stationary vehicle."""
    out: list[tuple[str, str, float, dict]] = []
    for p in people:
        if len(p.t) < 3:
            continue
        for kind, idx in (("person_exits_vehicle", 0), ("person_enters_vehicle", -1)):
            foot, t = p.foot[idx], float(p.t[idx])
            if not (EDGE < foot[0] < 1 - EDGE and EDGE < p.box[idx, 1] and foot[1] < 1 - EDGE):
                continue          # entered or left through the edge of the picture
            for v in vehicles:
                if v.t[0] - 1.0 > t or v.t[-1] + 1.0 < t or not _still_at(v, t, st):
                    continue
                if _near_vehicle(foot, _box_at(v, t), st.action_vehicle_pad):
                    out.append((kind, p.id, t, {"with_track": v.id}))
                    break
    return out


def people_close_events(people: list[Traj], st: IngestSettings) -> list[tuple[str, float, dict]]:
    """(person track id, time, payload) for two people standing close for at least `action_close_s` seconds."""
    out: list[tuple[str, float, dict]] = []
    sp = {p.id: speeds(p) for p in people}
    for i, a in enumerate(people):
        for b in people[i + 1:]:
            lo, hi = max(a.t[0], b.t[0]), min(a.t[-1], b.t[-1])
            if hi - lo < st.action_close_s:
                continue
            grid = np.arange(lo, hi, 0.25)
            ia = np.clip(np.searchsorted(a.t, grid), 0, len(a.t) - 1)
            ib = np.clip(np.searchsorted(b.t, grid), 0, len(b.t) - 1)
            dist = np.linalg.norm(a.foot[ia] - b.foot[ib], axis=1)
            tall = np.maximum(a.height[ia], b.height[ib])
            slow = (sp[a.id][ia] < st.action_move_speed) & (sp[b.id][ib] < st.action_move_speed)
            ok = (dist < st.action_close_height * tall) & slow
            run = 0
            for k, flag in enumerate(ok):
                run = run + 1 if flag else 0
                if run * 0.25 >= st.action_close_s:
                    out.append((a.id, float(grid[k - run + 1]), {"with_track": b.id, "seconds": round(run * 0.25, 1)}))
                    break
    return out


def load_trajectories(db: Database, camera_id: str) -> list[Traj]:
    with db.read() as c:
        cls = {r["id"]: r["cls"] for r in c.execute("SELECT id, cls FROM tracks WHERE camera_id=?", (camera_id,))}
        rows = c.execute(
            "SELECT p.track_id, p.t, p.x1, p.y1, p.x2, p.y2 FROM track_points p JOIN tracks k ON k.id = p.track_id "
            "WHERE k.camera_id=? ORDER BY p.track_id, p.t", (camera_id,)).fetchall()
    grouped: dict[str, list] = defaultdict(list)
    for r in rows:
        grouped[r["track_id"]].append((r["t"], r["x1"], r["y1"], r["x2"], r["y2"]))
    out = []
    for tid, pts in grouped.items():
        arr = np.asarray(pts, dtype=np.float64)
        out.append(Traj(tid, cls.get(tid, ""), arr[:, 0], arr[:, 1:5]))
    return out


def action_rows(camera_id: str, trajectories: list[Traj], st: IngestSettings) -> list[tuple]:
    """Rows for the `events` table: (id, camera_id, track_id, kind, zone_id, t, payload)."""
    vehicles = [t for t in trajectories if t.cls in VEHICLES and len(t.t) >= 2]
    people = [t for t in trajectories if t.cls == "person" and len(t.t) >= 2]
    found: list[tuple[str, str, float, dict]] = []
    for v in vehicles:
        found += [(kind, v.id, t, payload) for kind, t, payload in vehicle_events(v, st)]
    found += person_vehicle_events(people, vehicles, st)
    found += [("people_close", tid, t, payload) for tid, t, payload in people_close_events(people, st)]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    rows = []
    for kind, tid, t, payload in sorted(found, key=lambda f: f[2]):
        n = counts[(tid, kind)]
        counts[(tid, kind)] += 1
        rows.append((f"{tid}:{kind}:-:{n}", camera_id, tid, kind, None, t, json.dumps(payload)))
    return rows
