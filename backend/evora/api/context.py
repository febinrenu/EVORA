"""Per-process application context: active workspace, database, event bus and job runner."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evora.core import workspace as wsmod
from evora.core.bus import Bus
from evora.core.db import Database, open_db
from evora.core.jobs import IngestFn, JobRunner
from evora.core.workspace import Workspace


@dataclass
class AppContext:
    cfg: dict[str, Any]
    ws: Workspace
    db: Database
    bus: Bus
    runner: JobRunner

    @classmethod
    def build(
        cls, cfg: dict[str, Any], workspaces_root: Path | None = None, ingest_fn: IngestFn | None = None,
    ) -> AppContext:
        name = os.environ.get("evora_WORKSPACE") or cfg["workspace"]["default"]
        ws = wsmod.create(name, workspaces_root)
        db = open_db(ws.db_path)
        bus = Bus()
        jobs = cfg["jobs"]
        runner = JobRunner(
            db, bus, profile=os.environ.get("evora_PROFILE", "cpu"), workers=jobs["workers"],
            default_layers=jobs["default_layers"], ingest_fn=ingest_fn, stub_tick_s=jobs["stub_tick_s"],
        )
        runner.recover()
        return cls(cfg, ws, db, bus, runner)
