"""Live sessions: when a recorded file was being replayed as a live stream, so wall-clock times map back to the file.

During replay-as-live the analysed tracks carry wall-clock times, but the footage to render lives in the file. Each (re)start
of the restream begins a session; inside one, file position = ((t - started_at) x speed) mod duration.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from evora.core.db import Database

MAX_SESSIONS = 200  # keep the meta value small on a long-running demo


@dataclass
class Session:
    started_at: float
    speed: float = 1.0
    ended_at: float | None = None


def _key(camera_id: str) -> str:
    return f"live_sessions.{camera_id}"


def load(db: Database, camera_id: str) -> list[Session]:
    raw = db.get_meta(_key(camera_id))
    if not raw:
        return []
    try:
        return [Session(**item) for item in json.loads(raw)]
    except (ValueError, TypeError):
        return []


def _save(db: Database, camera_id: str, sessions: list[Session]) -> None:
    db.set_meta(_key(camera_id), json.dumps([asdict(s) for s in sessions[-MAX_SESSIONS:]]))


def begin(db: Database, camera_id: str, started_at: float, speed: float = 1.0) -> None:
    """Start a session; one that was still open ends at this moment (the stream restarted)."""
    sessions = load(db, camera_id)
    if sessions and sessions[-1].ended_at is None:
        sessions[-1].ended_at = started_at
    sessions.append(Session(started_at, speed))
    _save(db, camera_id, sessions)


def end(db: Database, camera_id: str, t: float) -> None:
    sessions = load(db, camera_id)
    if sessions and sessions[-1].ended_at is None:
        sessions[-1].ended_at = t
        _save(db, camera_id, sessions)


def file_offset(db: Database, camera_id: str, duration_s: float | None, t: float) -> float | None:
    """Seconds into the recorded file for wall-clock time `t`, or None when `t` is not inside a replay session."""
    if not duration_s or duration_s <= 0:
        return None
    hit = None
    for s in load(db, camera_id):
        if s.started_at <= t and (s.ended_at is None or t <= s.ended_at):
            hit = s  # later sessions win
    if hit is None:
        return None
    return ((t - hit.started_at) * hit.speed) % duration_s
