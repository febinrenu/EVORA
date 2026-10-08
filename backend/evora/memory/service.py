"""MemoryService: the single object the router and the API talk to."""
from __future__ import annotations

from typing import Any

from contracts.models import CameraInfo, ClarifyRequest, ClarifyResponse, MemoryFact, QueryPlan, Referent
from pydantic import BaseModel

from evora.core.cameras import list_cameras
from evora.core.db import Database
from evora.core.vectors import open_store
from evora.core.workspace import Workspace
from evora.memory.clarify import Clarifier, ClarifyOutcome
from evora.memory.embedder import TextEmbedder, default_embedder
from evora.memory.kb import KnowledgeBase, normalize
from evora.memory.resolve import Bound, Equivalence, Resolution, Resolver
from evora.memory.tod import parse_tod_range

_SYSTEM = (
    "You decide whether two short phrases refer to the same thing in one surveillance site "
    "(a place, an object or a time of day). Be strict: answer true only if a person at the site "
    'would use them interchangeably. Reply only with JSON: {"same": true|false}.'
)


class EquivalenceResult(BaseModel):
    same: bool


def gateway_equivalence(gateway: Any) -> Equivalence:
    """Wrap `Gateway.chat_json` as the grey-band equivalence check."""

    async def check(new_phrase: str, known_phrase: str, kind: str) -> bool:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Kind: {kind}\nPhrase A: {known_phrase}\nPhrase B: {new_phrase}"},
        ]
        return bool((await gateway.chat_json("equivalence", messages, EquivalenceResult)).same)

    return check


class MemoryService:
    def __init__(self, db: Database, kb: KnowledgeBase, resolver: Resolver, clarifier: Clarifier):
        self.db, self.kb, self.resolver, self.clarifier = db, kb, resolver, clarifier

    async def resolve(self, ref: Referent) -> Resolution:
        return await self.resolver.resolve(ref)

    def ask(
        self, query_id: str, text: str, plan: QueryPlan | dict[str, Any], ref: Referent, resolution: Resolution,
    ) -> ClarifyRequest:
        return self.clarifier.ask(query_id, text, plan, ref, resolution)

    def apply(self, resp: ClarifyResponse) -> ClarifyOutcome:
        return self.clarifier.apply(resp)

    def supersede(self, fact_id: str, new_binding: dict[str, Any], source: str = "correction") -> MemoryFact:
        return self.kb.supersede(fact_id, new_binding, source)

    async def resolve_time(self, phrase: str) -> tuple[str, str] | None:
        """A remembered time phrase ("after hours") as (tod_after, tod_before), or None if it is not known.

        The planner calls this before it would ask, so a time word is only ever explained once.
        """
        resolution = await self.resolver.resolve(Referent(text=phrase, role="time"))
        if isinstance(resolution, Bound):
            b = resolution.fact.binding
            if b.get("tod_after") and b.get("tod_before"):
                return b["tod_after"], b["tod_before"]
        return None

    def define_time(self, phrase: str, hours: str, source: str = "statement") -> MemoryFact:
        """'after hours means 8pm to 6am': store (or correct) a time alias from a typed statement."""
        after, before = parse_tod_range(hours)
        binding = {"tod_after": after, "tod_before": before}
        existing = self.kb.find_exact("time", phrase)
        if existing:
            return self.kb.supersede(existing[0].id, binding, "correction")
        return self.kb.create("time", normalize(phrase), binding, source)

    def cameras(self) -> list[CameraInfo]:
        return list_cameras(self.db)


def build_memory(
    db: Database, ws: Workspace, cfg: dict[str, Any], *, embedder: TextEmbedder | None = None,
    equivalence: Equivalence | None = None, cache_dir=None,
) -> MemoryService:
    mem = cfg["memory"]
    kb = KnowledgeBase(db, open_store(ws.vectors_dir), embedder or default_embedder(cache_dir))
    resolver = Resolver(
        kb, lambda: list_cameras(db), equivalence, tau_hi=mem["tau_hi"], tau_lo=mem["tau_lo"],
        margin=mem["margin"], alias_embed=mem["alias_embed"],
    )
    return MemoryService(db, kb, resolver, Clarifier(db, kb))

