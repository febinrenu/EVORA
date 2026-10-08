"""Standing queries and alerts in SQLite."""
from __future__ import annotations

import json
import time
import uuid

from contracts.models import Alert, Evidence, StandingQuery
from pydantic import BaseModel, Field

from evora.core.db import Database


class StandingRule(BaseModel):
    """What a watch looks for. Stored as JSON in `standing_queries.rule`."""

    targets: list[str] = Field(default_factory=list)       # detector classes, e.g. ["person"]
    attributes: list[str] = Field(default_factory=list)    # shown to the user; matched best-effort later
    place: str | None = None                              # the phrase, re-resolved each time so corrections apply
    camera_ids: list[str] = Field(default_factory=list)
    zone_id: str | None = None
    events: list[str] = Field(default_factory=list)       # event kinds that count
    direction: str | None = None                          # a_to_b | b_to_a for line crossings
    tod_after: str | None = None
    tod_before: str | None = None
    cooldown_s: float = 30.0
    summary: str = ""


class AlertNotFound(KeyError):
    pass


class StandingNotFound(KeyError):
    pass


def _to_standing(row) -> StandingQuery:
    return StandingQuery(
        id=row["id"], text=row["text"], rule=json.loads(row["rule"]), active=bool(row["active"]),
        created_at=row["created_at"],
    )


def create_standing(db: Database, text: str, rule: StandingRule) -> StandingQuery:
    sq_id = f"sq_{uuid.uuid4().hex[:8]}"
    with db.write() as c:
        c.execute(
            "INSERT INTO standing_queries(id,text,rule,active,created_at) VALUES(?,?,?,1,?)",
            (sq_id, text, rule.model_dump_json(), time.time()),
        )
    return get_standing(db, sq_id)


def get_standing(db: Database, sq_id: str) -> StandingQuery:
    with db.read() as c:
        row = c.execute("SELECT * FROM standing_queries WHERE id=?", (sq_id,)).fetchone()
    if row is None:
        raise StandingNotFound(sq_id)
    return _to_standing(row)


def list_standing(db: Database, active_only: bool = False) -> list[StandingQuery]:
    sql = "SELECT * FROM standing_queries" + (" WHERE active=1" if active_only else "") + " ORDER BY created_at, id"
    with db.read() as c:
        return [_to_standing(r) for r in c.execute(sql)]


def set_active(db: Database, sq_id: str, active: bool) -> StandingQuery:
    get_standing(db, sq_id)
    with db.write() as c:
        c.execute("UPDATE standing_queries SET active=? WHERE id=?", (1 if active else 0, sq_id))
    return get_standing(db, sq_id)


def _to_alert(row) -> Alert:
    return Alert(
        id=row["id"], standing_query_id=row["sq_id"], t=row["t"], camera_id=row["camera_id"],
        evidence=Evidence.model_validate_json(row["evidence"]), acknowledged=bool(row["acked"]),
    )


def insert_alert(db: Database, alert: Alert, track_id: str | None) -> bool:
    """Store an alert. Returns False when this rule already alerted for this event (same deterministic id)."""
    with db.write() as c:
        cur = c.execute(
            "INSERT OR IGNORE INTO alerts(id,sq_id,t,camera_id,track_id,evidence,acked) VALUES(?,?,?,?,?,?,0)",
            (alert.id, alert.standing_query_id, alert.t, alert.camera_id, track_id, alert.evidence.model_dump_json()),
        )
        return cur.rowcount == 1


def list_alerts(db: Database, acknowledged: bool | None = None, limit: int = 200) -> list[Alert]:
    sql, args = "SELECT * FROM alerts", []
    if acknowledged is not None:
        sql, args = sql + " WHERE acked=?", [1 if acknowledged else 0]
    with db.read() as c:
        return [_to_alert(r) for r in c.execute(sql + " ORDER BY t DESC, id LIMIT ?", [*args, limit])]


def acknowledge(db: Database, alert_id: str) -> Alert:
    with db.write() as c:
        if c.execute("UPDATE alerts SET acked=1 WHERE id=?", (alert_id,)).rowcount == 0:
            raise AlertNotFound(alert_id)
    with db.read() as c:
        return _to_alert(c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone())


def pinned_evidence_ids(db: Database) -> set[str]:
    """Evidence that belongs to an alert. Its cached clip and thumbnail are kept when the media cache is trimmed."""
    out: set[str] = set()
    with db.read() as c:
        rows = c.execute("SELECT evidence FROM alerts").fetchall()
    for row in rows:
        try:
            out.add(str(json.loads(row["evidence"])["id"]))
        except (ValueError, KeyError, TypeError):
            continue
    return out
