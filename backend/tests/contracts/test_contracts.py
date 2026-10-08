import json
import sqlite3
from pathlib import Path

import pytest
from contracts import models as m
from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / "contracts" / "fixtures"
SCHEMA = ROOT / "contracts" / "schema.sql"

CASES = {
    "cameras": list[m.CameraInfo],
    "plan": m.QueryPlan,
    "answer": m.Answer,
    "clarify_request": m.ClarifyRequest,
    "memory_facts": list[m.MemoryFact],
    "path": list[m.PathHop],
    "alert": m.Alert,
    "standing_query": m.StandingQuery,
    "zone": m.Zone,
    "ingest_jobs": list[m.IngestJob],
    "stream_query": list[m.StreamEvent],
    "stream_clarify": list[m.StreamEvent],
}


@pytest.mark.parametrize("name", CASES)
def test_fixture_matches_model(name):
    TypeAdapter(CASES[name]).validate_python(json.loads((FIX / f"{name}.json").read_text()))


def test_every_fixture_is_covered():
    on_disk = {p.stem for p in FIX.glob("*.json")}
    assert on_disk - set(CASES) == {"health"}


def test_schema_applies_and_is_idempotent():
    db = sqlite3.connect(":memory:")
    ddl = SCHEMA.read_text()
    db.executescript(ddl)
    db.executescript(ddl)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    assert {"cameras", "tracks", "track_points", "zones", "events", "memory_facts", "pending_queries",
            "alerts", "audit_log", "ingest_jobs", "query_log"} <= tables


def test_foreign_keys_enforced():
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA.read_text())
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("insert into tracks(id,camera_id,cls,t_start,t_end,n_obs) values('x','nope','person',0,1,1)")


def test_answer_has_dual_timestamps_on_every_evidence():
    ans = m.Answer.model_validate(json.loads((FIX / "answer.json").read_text()))
    assert len(ans.evidence) == 3 and all(e.offset_s >= 0 and e.t_peak for e in ans.evidence)
