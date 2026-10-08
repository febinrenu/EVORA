"""Events from stored trajectories: line crossings, zone enter/exit, dwell, appear/disappear.

Everything is computed from `track_points` alone, so drawing a zone after ingestion produces its events
in well under a second without touching the video (retroactive zones).

Geometry conventions (normalised image coordinates, y grows downwards):
  * the tracked point is the foot point, the bottom centre of the box
  * for a line from a to b, "left" is the side where dx*(py-ay) - dy*(px-ax) > 0 with (dx, dy) = b - a
  * crossing from left to right is `a_to_b`, right to left is `b_to_a` (this is what Zone.direction means)
Event kinds and `payload` follow contracts/schema.sql: cross_line (direction), enter_zone, exit_zone,
dwell (seconds, and duration_s as an alias), appear and disappear (x, y; zone_id NULL).
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from contracts.models import Zone

from evora.core.db import Database, open_db
from evora.perception.settings import IngestSettings

log = logging.getLogger("evora.perception.events")

Point = tuple[float, float]


@dataclass(frozen=True)
class Sample:
    t: float
    x: float   # foot point
    y: float


def foot(x1: float, y1: float, x2: float, y2: float) -> Point:
    return (x1 + x2) / 2.0, y2


def side(p: Point, a: Point, b: Point) -> float:
    """Signed distance of p from the infinite line a-b, in coordinate units (left > 0)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0
    return (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / length


def _along(p: Point, a: Point, b: Point) -> float:
    """Where p projects on a-b: 0 at a, 1 at b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    return 0.0 if denom == 0 else ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom


def line_crossings(samples: Sequence[Sample], a: Point, b: Point, hysteresis: float = 0.01) -> list[tuple[float, str]]:
    """(time, direction) of every crossing of the segment a-b.

    A crossing counts once the path is at least `hysteresis` beyond the line on the new side, which
    suppresses jitter along the line. Time is interpolated to where the path met the line, and the
    meeting point has to lie on the drawn segment, not on its extension.
    """
    out: list[tuple[float, str]] = []
    firm: tuple[Sample, float] | None = None   # last sample clearly on one side, and its signed distance
    for s in samples:
        d = side((s.x, s.y), a, b)
        if abs(d) < hysteresis:
            continue
        if firm is not None and (d > 0) != (firm[1] > 0):
            prev, d0 = firm
            frac = d0 / (d0 - d)
            meet = (prev.x + (s.x - prev.x) * frac, prev.y + (s.y - prev.y) * frac)
            if -0.02 <= _along(meet, a, b) <= 1.02:
                out.append((prev.t + (s.t - prev.t) * frac, "a_to_b" if d0 > 0 else "b_to_a"))
        firm = (s, d)
    return out


def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > p[1]) != (y2 > p[1]) and p[0] < (x2 - x1) * (p[1] - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


@dataclass(frozen=True)
class Span:
    t_in: float
    t_out: float | None   # None when the track ended inside


def inside_spans(samples: Sequence[Sample], poly: Sequence[Point], debounce: int = 2) -> list[Span]:
    """Maximal stretches inside `poly`. A change of state must persist for `debounce` samples to count."""
    spans: list[Span] = []
    state = False
    run = 0
    start_idx = 0
    t_in = 0.0
    for i, s in enumerate(samples):
        now = point_in_polygon((s.x, s.y), poly)
        if now == state:
            run = 0
            continue
        if run == 0:
            start_idx = i
        run += 1
        if run >= debounce:
            t_change = samples[start_idx].t
            if now:
                t_in = t_change
            else:
                spans.append(Span(t_in, t_change))
            state, run = now, 0
    if state:
        spans.append(Span(t_in, None))
    return spans


def track_events(track_id: str, camera_id: str, samples: Sequence[Sample], zones: Sequence[Zone], *,
                 t_end: float, dwell_s: float, hysteresis: float, debounce: int) -> list[tuple]:
    """Event rows (id, camera_id, track_id, kind, zone_id, t, payload_json) for one track and the given zones."""
    rows: list[tuple] = []

    def add(kind: str, zone_id: str | None, t: float, payload: dict) -> None:
        n = sum(1 for r in rows if r[3] == kind and r[4] == zone_id)
        rows.append((f"{track_id}:{kind}:{zone_id or '-'}:{n}", camera_id, track_id, kind, zone_id, t, json.dumps(payload)))

    for z in zones:
        if z.kind == "line" and len(z.points) >= 2:
            for t, direction in line_crossings(samples, z.points[0], z.points[1], hysteresis):
                add("cross_line", z.id, t, {"direction": direction})
        elif z.kind == "polygon" and len(z.points) >= 3:
            for span in inside_spans(samples, z.points, debounce):
                add("enter_zone", z.id, span.t_in, {})
                if span.t_out is not None:
                    add("exit_zone", z.id, span.t_out, {})
                last = span.t_out if span.t_out is not None else t_end
                if last - span.t_in >= dwell_s:
                    seconds = round(last - span.t_in, 2)
                    add("dwell", z.id, span.t_in + dwell_s, {"seconds": seconds, "duration_s": seconds})
        elif z.kind == "frame" and samples and samples[-1].t - samples[0].t >= dwell_s:
            seconds = round(samples[-1].t - samples[0].t, 2)
            add("dwell", z.id, samples[0].t + dwell_s, {"seconds": seconds, "duration_s": seconds})
    return rows


def load_samples(db: Database, camera_id: str) -> tuple[dict[str, list[Sample]], dict[str, tuple[float, float]]]:
    """Foot-point trajectories per track of one camera, and each track's (t_start, t_end)."""
    with db.read() as c:
        spans = {r["id"]: (r["t_start"], r["t_end"]) for r in c.execute(
            "SELECT id, t_start, t_end FROM tracks WHERE camera_id=?", (camera_id,))}
        rows = c.execute(
            "SELECT p.track_id, p.t, p.x1, p.y1, p.x2, p.y2 FROM track_points p JOIN tracks t ON t.id = p.track_id "
            "WHERE t.camera_id=? ORDER BY p.track_id, p.t", (camera_id,)).fetchall()
    by_track: dict[str, list[Sample]] = defaultdict(list)
    for r in rows:
        x, y = foot(r["x1"], r["y1"], r["x2"], r["y2"])
        by_track[r["track_id"]].append(Sample(r["t"], x, y))
    return by_track, spans


def recompute_events(camera_id: str, zones: list[Zone], *, db: Database | None = None,
                     settings: IngestSettings | None = None) -> int:
    """Recompute events for `zones` on one camera from stored trajectories and return the rows written.

    Existing events of those zones are replaced. appear/disappear (no zone) are refreshed every call.
    """
    if db is None:
        from evora.perception.pipeline import resolve_workspace

        db = open_db(resolve_workspace().db_path)
    st = settings or IngestSettings()
    by_track, spans = load_samples(db, camera_id)
    zone_ids = [z.id for z in zones]
    rows: list[tuple] = []
    for tid, (t_start, t_end) in spans.items():
        samples = by_track.get(tid, [])
        first = (samples[0].x, samples[0].y) if samples else (0.0, 0.0)
        last = (samples[-1].x, samples[-1].y) if samples else (0.0, 0.0)
        for kind, t, xy in (("appear", t_start, first), ("disappear", t_end, last)):
            payload = json.dumps({"x": round(xy[0], 4), "y": round(xy[1], 4)})
            rows.append((f"{tid}:{kind}:-:0", camera_id, tid, kind, None, t, payload))
        rows += track_events(tid, camera_id, samples, zones, t_end=t_end, dwell_s=st.dwell_s,
                             hysteresis=st.line_hysteresis, debounce=st.zone_debounce)
    with db.write() as c:
        if zone_ids:
            marks = ",".join("?" * len(zone_ids))
            c.execute(f"DELETE FROM events WHERE camera_id=? AND zone_id IN ({marks})", (camera_id, *zone_ids))  # noqa: S608 - placeholders only
        c.execute("DELETE FROM events WHERE camera_id=? AND zone_id IS NULL AND kind IN ('appear','disappear')", (camera_id,))
        c.executemany(
            "INSERT OR REPLACE INTO events(id,camera_id,track_id,kind,zone_id,t,payload) VALUES(?,?,?,?,?,?,?)", rows)
    log.info("%s: %d events for %d zones", camera_id, len(rows), len(zones))
    return len(rows)


def zones_of(db: Database, camera_id: str) -> list[Zone]:
    """The zones currently stored for a camera (used when L2 reruns)."""
    with db.read() as c:
        rows = c.execute("SELECT id, camera_id, kind, points, direction FROM zones WHERE camera_id=?", (camera_id,)).fetchall()
    return [Zone(id=r["id"], camera_id=r["camera_id"], kind=r["kind"], points=[tuple(p) for p in json.loads(r["points"] or "[]")],
                 direction=r["direction"]) for r in rows]
