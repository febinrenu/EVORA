"""Append-only audit log (queries, unblur, exports)."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from evora.core.db import Database


def record(db: Database, action: str, detail: dict[str, Any], actor: str = "local") -> str:
    entry_id = f"aud_{uuid.uuid4().hex[:12]}"
    with db.write() as c:
        c.execute(
            "INSERT INTO audit_log(id,actor,action,detail,t) VALUES(?,?,?,?,?)",
            (entry_id, actor, action, json.dumps(detail), time.time()),
        )
    return entry_id


def entries(
    db: Database, action: str | None = None, limit: int | None = None, newest_first: bool = False,
) -> list[dict[str, Any]]:
    sql, args = "SELECT * FROM audit_log", []
    if action:
        sql, args = sql + " WHERE action=?", [action]
    sql += " ORDER BY t DESC, rowid DESC" if newest_first else " ORDER BY t"
    if limit is not None:
        sql, args = sql + " LIMIT ?", [*args, limit]
    with db.read() as c:
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in c.execute(sql, args)]
