"""Estimates for actions we cannot recognise, from how people, bags and vehicles moved.

There is no action recogniser. For a few actions the stored tracks still point at a likely person and moment:
a bag that becomes still where a person was (put down, dropped), a still bag that leaves with a person (picked up),
a person who appears or disappears right next to a stopped vehicle (got out of or into it), a path that bends
(turned left, right, around) and a speed well above walking (ran). Each estimate is a `Cue`; the answer presents it
as the most likely moment, labelled as an estimate, never as a recognised action.

All positions are normalised image coordinates (0..1, y down), so "left" and "right" are on screen: a camera's
perspective can flip them, and the answer says so.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from evora.core.db import Database

BAGS = ("backpack", "handbag", "suitcase", "umbrella")
VEHICLES = ("car", "truck", "bus", "motorcycle", "bicycle")
CUE_KEYS = {"put_down", "drop", "pick_up", "vehicle_out", "vehicle_in", "turn_left", "turn_right", "u_turn", "run",
            "talk", "greet", "hand_over"}
# actions perception already stores as events (perception/actions.py): read them first, they are computed at indexing
# time with the driver's left and right; older indexes without them fall back to the geometry below
EVENT_KINDS = {
    "vehicle_out": ("person_exits_vehicle",), "vehicle_in": ("person_enters_vehicle",),
    "turn_left": ("vehicle_turn_left",), "turn_right": ("vehicle_turn_right",), "u_turn": ("vehicle_u_turn",),
    "talk": ("people_close",), "greet": ("people_close",), "hand_over": ("people_close",),
}
STORED_ACTIONS = ("vehicle_start", "vehicle_stop", "vehicle_turn_left", "vehicle_turn_right", "vehicle_u_turn",
                  "vehicle_reverse", "person_exits_vehicle", "person_enters_vehicle", "people_close")


class Box(NamedTuple):
    t: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Cue:
    track_id: str
    t: float          # the moment the clip and thumbnail should show
    score: float      # 0..1, how clearly the movement fits; used only to order estimates
    why: str          # a predicate for "this person ...", e.g. "left a backpack that then stayed still"


@dataclass(frozen=True)
class CueConfig:
    still_move: float = 0.02        # a bag "stays still" when its centre moves less than this ...
    still_s: float = 3.0            # ... for at least this long
    grow: float = 0.15              # a bag is "at" a person when inside the person's box grown by this fraction
    separate_widths: float = 1.0    # the person then moves away more than this many of their own widths
    separate_within_s: float = 10.0
    vehicle_still_move: float = 0.01  # a vehicle is "stopped" when its centre moves less than this over +-2 s
    vehicle_reach: float = 0.5      # a person appears or disappears within this many vehicle widths of it
    edge: float = 0.02              # a track that starts or ends this close to the frame edge came from or left the view
    turn_deg: float = 45.0          # a heading change of at least this is a turn ...
    u_turn_deg: float = 135.0       # ... and at least this, turning around
    min_path: float = 0.08          # a path shorter than this (normalised) is too short to have a direction
    run_speed: float = 1.5          # body heights per second over `run_window_s`; walking is about 0.6-0.8
    run_window_s: float = 2.0


def find_cues(db: Database, key: str, tracks: Iterable[tuple[str, str]], cfg: CueConfig | None = None) -> dict[str, Cue]:
    """Cues for the candidate tracks `(track_id, camera_id)`, keyed by track id. Empty when the action has no cue."""
    cfg = cfg or CueConfig()
    tracks = list(tracks)
    if key not in CUE_KEYS or not tracks:
        return {}
    stored = _from_events(db, key, tracks)
    if stored is not None:
        return stored
    if key in ("talk", "greet", "hand_over"):
        return {}  # only perception's people_close events can point at these
    points = _points(db, [tid for tid, _ in tracks])
    if key in ("put_down", "drop", "pick_up"):
        return _bag_cues(db, key, tracks, points, cfg)
    if key in ("vehicle_out", "vehicle_in"):
        return _vehicle_cues(db, key, tracks, points, cfg)
    if key in ("turn_left", "turn_right", "u_turn"):
        return {tid: c for tid, _ in tracks if (c := _turn(tid, points.get(tid, []), key, cfg)) is not None}
    return {tid: c for tid, _ in tracks if (c := _run(tid, points.get(tid, []), cfg)) is not None}


# ------------------------------------------------------------------ stored action events
def _from_events(db: Database, key: str, tracks: list[tuple[str, str]]) -> dict[str, Cue] | None:
    """Cues from perception's action events, or None when these cameras were indexed before such events existed."""
    kinds = EVENT_KINDS.get(key)
    if not kinds:
        return None
    ids, cams = {tid for tid, _ in tracks}, sorted({cam for _, cam in tracks})
    cam_marks = ",".join("?" * len(cams))
    with db.read() as c:
        indexed = c.execute(f"SELECT 1 FROM events WHERE camera_id IN ({cam_marks}) AND kind IN "  # noqa: S608
                            f"({','.join('?' * len(STORED_ACTIONS))}) LIMIT 1", [*cams, *STORED_ACTIONS]).fetchone()
        if indexed is None:
            return None
        rows = c.execute(f"SELECT track_id, kind, t, payload FROM events WHERE camera_id IN ({cam_marks}) AND kind IN "  # noqa: S608
                         f"({','.join('?' * len(kinds))}) ORDER BY t", [*cams, *kinds]).fetchall()
    out: dict[str, Cue] = {}
    for r in rows:
        try:
            payload = json.loads(r["payload"] or "{}")
        except ValueError:
            payload = {}
        cue_why, score = _event_words(r["kind"], payload)
        for tid in (r["track_id"], payload.get("with_track")):
            if tid in ids and tid not in out:
                out[tid] = Cue(tid, float(r["t"]), score, cue_why)
    return out


def _event_words(kind: str, payload: dict) -> tuple[str, float]:
    deg = payload.get("turn_deg")
    about = f" (about {abs(deg):.0f} degrees)" if isinstance(deg, int | float) else ""
    words = {
        "person_exits_vehicle": ("appeared right next to a stopped vehicle", 0.8),
        "person_enters_vehicle": ("disappeared right next to a stopped vehicle", 0.8),
        "vehicle_turn_left": (f"turned left{about}", 0.8),
        "vehicle_turn_right": (f"turned right{about}", 0.8),
        "vehicle_u_turn": (f"turned around{about}", 0.8),
        "people_close": (f"stood within a body height of another person for {payload.get('seconds', 'a few')} seconds", 0.6),
    }
    return words.get(kind, ("moved in a way that fits", 0.5))


# ------------------------------------------------------------------ data
def _points(db: Database, ids: list[str]) -> dict[str, list[Box]]:
    out: dict[str, list[Box]] = defaultdict(list)
    with db.read() as c:
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            for r in c.execute(f"SELECT track_id, t, x1, y1, x2, y2 FROM track_points WHERE track_id IN "  # noqa: S608
                               f"({','.join('?' * len(chunk))}) ORDER BY track_id, t", chunk):
                out[r["track_id"]].append(Box(r["t"], r["x1"], r["y1"], r["x2"], r["y2"]))
    return out


def _others(db: Database, camera_ids: set[str], classes: tuple[str, ...]) -> dict[str, tuple[str, list[Box]]]:
    """Every track of these classes on these cameras, with its points: {track_id: (class, points)}."""
    with db.read() as c:
        cams, kinds = ",".join("?" * len(camera_ids)), ",".join("?" * len(classes))
        rows = c.execute(f"SELECT id, cls FROM tracks WHERE camera_id IN ({cams}) AND cls IN ({kinds})",  # noqa: S608
                         [*camera_ids, *classes]).fetchall()
    cls = {r["id"]: r["cls"] for r in rows}
    pts = _points(db, list(cls))
    return {tid: (cls[tid], pts[tid]) for tid in cls if pts.get(tid)}


def _at(points: list[Box], t: float, tol: float = 0.5) -> Box | None:
    if not points:
        return None
    best = min(points, key=lambda p: abs(p.t - t))
    return best if abs(best.t - t) <= tol else None


def _inside(bag: Box, person: Box, grow: float) -> bool:
    gw, gh = grow * person.w, grow * person.h
    return person.x1 - gw <= bag.cx <= person.x2 + gw and person.y1 - gh <= bag.cy <= person.y2 + gh


def _still_from(points: list[Box], cfg: CueConfig) -> float | None:
    """The first moment from which the box stays still for `still_s`, or None."""
    for i, start in enumerate(points):
        for j in range(i + 1, len(points)):
            if math.hypot(points[j].cx - start.cx, points[j].cy - start.cy) >= cfg.still_move:
                break
            if points[j].t - start.t >= cfg.still_s:
                return start.t
    return None


def _still_until(points: list[Box], cfg: CueConfig) -> float | None:
    """The last moment of a stretch of at least `still_s` during which the box stayed still, or None."""
    reversed_points = [Box(-p.t, p.x1, p.y1, p.x2, p.y2) for p in reversed(points)]
    t = _still_from(reversed_points, cfg)
    return -t if t is not None else None


# ------------------------------------------------------------------ bags
def _bag_cues(db: Database, key: str, tracks: list[tuple[str, str]], points: dict[str, list[Box]],
              cfg: CueConfig) -> dict[str, Cue]:
    by_cam: dict[str, list[str]] = defaultdict(list)
    for tid, cam in tracks:
        by_cam[cam].append(tid)
    bags = _others(db, set(by_cam), BAGS)
    out: dict[str, Cue] = {}
    for bag_id, (bag_cls, bag_pts) in bags.items():
        cam = _camera_of(db, bag_id)
        for person_id in by_cam.get(cam, []):
            person = points.get(person_id, [])
            cue = (_left_behind if key in ("put_down", "drop") else _picked_up)(person_id, person, bag_cls, bag_pts, cfg)
            if cue is not None and (person_id not in out or cue.score > out[person_id].score):
                out[person_id] = cue
    return out


def _camera_of(db: Database, track_id: str) -> str:
    with db.read() as c:
        row = c.execute("SELECT camera_id FROM tracks WHERE id=?", (track_id,)).fetchone()
    return row["camera_id"] if row else ""


def _left_behind(person_id: str, person: list[Box], bag_cls: str, bag: list[Box], cfg: CueConfig) -> Cue | None:
    """The bag becomes still where this person is, then the person moves away from it."""
    t_still = _still_from(bag, cfg)
    if t_still is None:
        return None
    b, p = _at(bag, t_still), _at(person, t_still)
    if b is None or p is None or not _inside(b, p, cfg.grow):
        return None
    for later in person:
        if t_still < later.t <= t_still + cfg.separate_within_s:
            if math.hypot(later.cx - b.cx, later.cy - b.cy) > cfg.separate_widths * max(p.w, 1e-6) + p.w / 2:
                return Cue(person_id, t_still, 0.8, f"left a {bag_cls} that then stayed where they had been")
    return None


def _picked_up(person_id: str, person: list[Box], bag_cls: str, bag: list[Box], cfg: CueConfig) -> Cue | None:
    """A bag that stood still is at this person when it stops standing still, and then goes with them."""
    t_end = _still_until(bag, cfg)
    if t_end is None:
        return None
    b, p = _at(bag, t_end), _at(person, t_end)
    if b is None or p is None or not _inside(b, p, cfg.grow):
        return None
    after = [x for x in bag if x.t > t_end + 0.2]
    if not after:  # the bag disappears while this person is on it: it most likely went with them (weaker)
        return Cue(person_id, t_end, 0.6, f"was at a {bag_cls} that had been standing still, which then was gone")
    moved_with = sum(1 for x in after if (q := _at(person, x.t)) is not None and _inside(x, q, cfg.grow))
    if moved_with >= 2:
        return Cue(person_id, t_end, 0.8, f"picked up a {bag_cls} that had been standing still and carried it away")
    return None


# ------------------------------------------------------------------ vehicles
def _vehicle_cues(db: Database, key: str, tracks: list[tuple[str, str]], points: dict[str, list[Box]],
                  cfg: CueConfig) -> dict[str, Cue]:
    cams = {cam for _, cam in tracks}
    vehicles = _others(db, cams, VEHICLES)
    cam_of = {vid: _camera_of(db, vid) for vid in vehicles}
    out: dict[str, Cue] = {}
    for tid, cam in tracks:
        person = points.get(tid, [])
        if len(person) < 2:
            continue
        edge = person[0] if key == "vehicle_out" else person[-1]
        if min(edge.x1, edge.y1, 1 - edge.x2, 1 - edge.y2) <= cfg.edge:
            continue  # came into or left the view at the frame edge: not from or into a vehicle
        for vid, (v_cls, v_pts) in vehicles.items():
            if cam_of[vid] != cam:
                continue
            v = _at(v_pts, edge.t)
            if v is None or not _stopped(v_pts, edge.t, cfg):
                continue
            gap = _rect_distance(edge, v)
            if gap <= cfg.vehicle_reach * max(v.w, 1e-6):
                verb = "appeared right next to" if key == "vehicle_out" else "disappeared right next to"
                cue = Cue(tid, edge.t, 0.75 - min(gap, 0.2), f"{verb} a stopped {v_cls}")
                if tid not in out or cue.score > out[tid].score:
                    out[tid] = cue
    return out


def _stopped(points: list[Box], t: float, cfg: CueConfig) -> bool:
    near = [p for p in points if abs(p.t - t) <= 2.0]
    if len(near) < 2:
        return False
    return max(math.hypot(a.cx - b.cx, a.cy - b.cy) for a in near for b in near) < cfg.vehicle_still_move


def _rect_distance(a: Box, b: Box) -> float:
    dx = max(b.x1 - a.x2, a.x1 - b.x2, 0.0)
    dy = max(b.y1 - a.y2, a.y1 - b.y2, 0.0)
    return math.hypot(dx, dy)


# ------------------------------------------------------------------ paths
def _turn(tid: str, pts: list[Box], key: str, cfg: CueConfig) -> Cue | None:
    if len(pts) < 4:
        return None
    q = max(1, len(pts) // 4)
    h1 = (pts[q].cx - pts[0].cx, pts[q].cy - pts[0].cy)
    h2 = (pts[-1].cx - pts[-1 - q].cx, pts[-1].cy - pts[-1 - q].cy)
    if math.hypot(*h1) < cfg.min_path / 4 or math.hypot(*h2) < cfg.min_path / 4:
        return None
    path = sum(math.hypot(b.cx - a.cx, b.cy - a.cy) for a, b in zip(pts, pts[1:], strict=False))
    if path < cfg.min_path:
        return None
    cross, dot = h1[0] * h2[1] - h1[1] * h2[0], h1[0] * h2[0] + h1[1] * h2[1]
    angle = math.degrees(math.atan2(cross, dot))   # y points down: a positive angle is clockwise on screen, a right turn
    size = abs(angle)
    if key == "u_turn":
        ok, words = size >= cfg.u_turn_deg, "turned around"
    elif key == "turn_right":
        ok, words = cfg.turn_deg <= angle < cfg.u_turn_deg, "turned right on screen"
    else:
        ok, words = -cfg.u_turn_deg < angle <= -cfg.turn_deg, "turned left on screen"
    if not ok:
        return None
    mid = pts[len(pts) // 2].t
    return Cue(tid, mid, min(0.9, 0.5 + size / 360), f"{words} (about {size:.0f} degrees)")


def _run(tid: str, pts: list[Box], cfg: CueConfig) -> Cue | None:
    if len(pts) < 3:
        return None
    heights = sorted(p.h for p in pts)
    body = max(heights[len(heights) // 2], 1e-6)
    best, best_t = 0.0, None
    j = 0
    for i, start in enumerate(pts):
        j = max(j, i + 1)
        while j < len(pts) and pts[j].t - start.t < cfg.run_window_s:
            j += 1
        if j >= len(pts):
            break
        end = pts[j]
        speed = math.hypot(end.cx - start.cx, end.cy - start.cy) / body / (end.t - start.t)
        if speed > best:
            best, best_t = speed, (start.t + end.t) / 2
    if best_t is None or best < cfg.run_speed:
        return None
    return Cue(tid, best_t, min(0.9, 0.4 + best / 10), f"moved about {best:.1f} body heights a second, faster than walking")


__all__ = ["CUE_KEYS", "Cue", "CueConfig", "find_cues"]
