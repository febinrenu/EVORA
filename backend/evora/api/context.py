"""Per-process application context: workspace, database, bus, jobs, media, memory, gateway and query router."""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from evora.alerts.compiler import StandingCompiler
from evora.alerts.engine import AlertEngine
from evora.alerts.notify import Notifier
from evora.core import perception_adapter
from evora.core import settings as app_settings
from evora.core import workspace as wsmod
from evora.core.bus import Bus
from evora.core.config import load_env_file
from evora.core.db import Database, open_db
from evora.core.jobs import IngestFn, JobRunner
from evora.core.media_service import BlurFn, MediaService, UnblurTokens
from evora.core.privacy_guard import guard
from evora.core.workspace import Workspace
from evora.core.zone_service import ZoneService
from evora.evidence.prerender import Prerenderer
from evora.llm.gateway import Gateway
from evora.llm.keypool import KeyPool
from evora.llm.schemas import GatewayConfig
from evora.memory.embedder import TextEmbedder
from evora.memory.resolve import Equivalence
from evora.memory.service import MemoryService, build_memory, gateway_equivalence
from evora.query.planner import Planner, SqlitePlanCache


@dataclass
class AppContext:
    cfg: dict[str, Any]
    ws: Workspace
    db: Database
    bus: Bus
    runner: JobRunner
    media: MediaService
    unblur: UnblurTokens
    prerender: Prerenderer
    memory: MemoryService
    settings: dict[str, Any]
    mock: bool = False
    gateway: Any = None
    http: httpx.AsyncClient | None = None
    clarifier: Any = None
    router: Any = None
    zones: Any = None
    alerts: Any = None
    notifier: Any = None
    planner: Any = None
    compiler: Any = None

    @classmethod
    def build(
        cls, cfg: dict[str, Any], workspaces_root: Path | None = None, ingest_fn: IngestFn | None = None,
        blur_provider: Callable[[], BlurFn | None] | None = None, embedder: TextEmbedder | None = None,
        equivalence: Equivalence | None = None, gateway: Any = None, mock: bool | None = None,
    ) -> AppContext:
        name = os.environ.get("evora_WORKSPACE") or cfg["workspace"]["default"]
        ws = wsmod.create(name, workspaces_root)
        db = open_db(ws.db_path)
        bus = Bus()
        jobs = cfg["jobs"]
        runner = JobRunner(
            db, bus, profile=os.environ.get("evora_PROFILE", "cpu"), workers=jobs["workers"],
            default_layers=jobs["default_layers"], ingest_fn=ingest_fn, stub_tick_s=jobs["stub_tick_s"], ws=ws,
        )
        runner.recover()
        defaults = {"onprem": bool(cfg["llm"]["onprem"]), "blur_faces": bool(cfg["media"]["blur_faces"]), "reference_now": None}
        settings = app_settings.load(db, defaults, force_onprem=os.environ.get("evora_ONPREM") == "1")
        guard.install(lambda: bool(settings["onprem"]), cfg.get("privacy", {}).get("allow_hosts", []))
        media = MediaService(ws, cfg, blur_provider or perception_adapter.get_blur_faces)
        prerender = Prerenderer(db, media, lambda: bool(settings["blur_faces"]), top=int(cfg["media"]["prerender_top"]))
        mock = bool(os.environ.get("evora_MOCK") == "1") if mock is None else mock
        http: httpx.AsyncClient | None = None
        if gateway is None and not mock:
            load_env_file()
            http = httpx.AsyncClient()
            gateway = Gateway(
                GatewayConfig.from_env(os.environ), KeyPool.from_env(os.environ), http, onprem=lambda: bool(settings["onprem"])
            )
        if equivalence is None and gateway is not None:
            equivalence = gateway_equivalence(gateway)
        memory = build_memory(db, ws, cfg, embedder=embedder, equivalence=equivalence)
        ctx = cls(cfg, ws, db, bus, runner, media, UnblurTokens(), prerender, memory, settings, mock, gateway, http)
        ctx.zones = ZoneService(db, bus, memory.kb)
        memory.clarifier.on_zone = ctx.zones.recompute_zone
        ctx.notifier = Notifier(gateway, lambda: bool(settings["onprem"]))
        ctx.alerts = AlertEngine(db, bus, memory.kb, ctx.notifier)

        def after_ingest(camera_id: str) -> None:
            ctx.zones.recompute_camera(camera_id)
            ctx.alerts.backfill(camera_id)

        runner.on_done = after_ingest
        ctx.zones.on_recomputed = lambda camera_id: ctx.alerts.backfill(camera_id)
        if not mock:
            from evora.api.query_wiring import ClarifierAdapter, build_router

            ctx.clarifier = ClarifierAdapter(memory)
            ctx.planner = Planner(gateway, SqlitePlanCache(db))
            ctx.compiler = StandingCompiler(db, ctx.planner, memory, float(cfg["alerts"]["default_cooldown_s"]))
            ctx.router = build_router(ctx, gateway, ctx.clarifier, ctx.planner)
        return ctx
