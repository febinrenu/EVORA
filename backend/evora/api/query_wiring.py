"""Adapters that plug M1's memory and media into M3's query router."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from contracts.models import ClarifyResponse, Evidence, QueryPlan

from evora.core import cameras as cams
from evora.core import perception_adapter
from evora.core.media_service import MediaError
from evora.core.vectors import open_store
from evora.memory.clarify import ClarifyOutcome
from evora.query.planner import Planner
from evora.query.retrieve import RetrievalConfig, RetrievalResult, Retriever, SearchScope
from evora.query.router import Router, RouterConfig
from evora.query.verify import Verifier

if TYPE_CHECKING:
    from evora.api.context import AppContext

log = logging.getLogger("evora.query_wiring")


class ClarifierAdapter:
    """M3's `Clarifier` protocol over MemoryService.

    The HTTP route applies the user's answer first (so a bad answer is a 422, not a broken stream) and stashes the
    outcome here; `resume` hands it to the router, which re-runs the question with the new fact in memory.
    """

    def __init__(self, memory) -> None:  # noqa: ANN001 - MemoryService, kept loose to avoid an import cycle in tests
        self.memory = memory
        self._resumed: dict[str, tuple[str, QueryPlan]] = {}

    def ask(self, query_id, text, plan, ref, resolution, options):  # noqa: ANN001, ANN201 - mirrors the router protocol
        return self.memory.ask(query_id, text, plan, ref, resolution)  # `options` are built by the memory clarifier

    def stash(self, outcome: ClarifyOutcome) -> None:
        pending = outcome.pending
        self._resumed[pending.query_id] = (pending.text, QueryPlan.model_validate(pending.plan))

    def resume(self, resp: ClarifyResponse) -> tuple[str, QueryPlan] | None:
        return self._resumed.pop(resp.query_id, None)


class NullRetriever:
    """Stands in until M2's query embedder is installed: planning, memory and clarify are real, retrieval is empty."""

    async def search(self, plan: QueryPlan, scope: SearchScope | None = None) -> RetrievalResult:
        result = RetrievalResult()
        result.notes.append("Footage is still being indexed, so there is nothing to search yet.")
        return result


class CropSource:
    """Images the verifier sends to a vision model. Faces are blurred first; with no blur available, nothing is sent."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def crop_for(self, ev: Evidence) -> bytes | None:
        data = self._best_crop(ev) or self._frame(ev)
        if data is None:
            return None
        if not self.ctx.settings["blur_faces"]:
            return data
        blur = self.ctx.media.blur_function()
        if blur is None:
            log.warning("verification skipped: face blur is not installed and nothing unblurred may leave the machine")
            return None
        try:
            return blur(data)
        except Exception as exc:  # noqa: BLE001 - a blur model that cannot run means nothing may be sent
            log.warning("verification skipped: face blur failed (%s)", exc.__class__.__name__)
            return None

    def _best_crop(self, ev: Evidence) -> bytes | None:
        if not ev.track_id:
            return None
        with self.ctx.db.read() as c:
            row = c.execute("SELECT best_crop FROM tracks WHERE id=?", (ev.track_id,)).fetchone()
        if row is None or not row["best_crop"]:
            return None
        media_dir = self.ctx.ws.media_dir.resolve()
        path = (media_dir / row["best_crop"]).resolve()
        if media_dir not in path.parents or not path.is_file():  # M2 stores paths relative to media/; never leave it
            return None
        return path.read_bytes()

    def _frame(self, ev: Evidence) -> bytes | None:
        try:
            cam = cams.get_camera(self.ctx.db, ev.camera_id)
            return self.ctx.media.frame(cam, ev.t_peak, want_blur=False, bbox=None)[0]
        except (cams.CameraNotFound, MediaError):
            return None


def build_router(ctx: AppContext, gateway: Any, clarifier: ClarifierAdapter, planner: Planner) -> Router:
    embedder = perception_adapter.get_query_embedder()
    if embedder is not None:
        retriever: Any = Retriever(
            ctx.db, open_store(ctx.ws.vectors_dir), embedder, RetrievalConfig.from_cfg(ctx.cfg), gateway
        )
    else:
        log.warning("no query embedder installed: retrieval is disabled until perception.embed is available")
        retriever = NullRetriever()
    return Router(
        ctx.db, planner, retriever, ctx.memory.resolver, clarifier,
        Verifier(gateway, CropSource(ctx)), RouterConfig(), reference_override=lambda: ctx.settings["reference_now"],
    )

