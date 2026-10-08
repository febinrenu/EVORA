"""Referent resolution: exact alias, alias embeddings, LLM equivalence for the grey band, camera names."""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from contracts.models import CameraInfo, MemoryFact, Referent

from evora.memory.kb import KnowledgeBase, normalize

log = logging.getLogger("evora.memory.resolve")

# (new phrase, known phrase, kind) -> are they the same thing at this site?
Equivalence = Callable[[str, str, str], Awaitable[bool]]


@dataclass(frozen=True)
class Bound:
    fact: MemoryFact
    via: str  # exact | embedding | equivalence | camera
    persisted: bool = True


@dataclass(frozen=True)
class Ambiguous:
    facts: list[MemoryFact] = field(default_factory=list)


@dataclass(frozen=True)
class Unknown:
    pass


Resolution = Bound | Ambiguous | Unknown


class Resolver:
    def __init__(
        self, kb: KnowledgeBase, cameras: Callable[[], list[CameraInfo]], equivalence: Equivalence | None = None, *,
        tau_hi: float = 0.85, tau_lo: float = 0.60, margin: float = 0.03, alias_embed: bool = True,
    ):
        self.kb, self._cameras, self._equivalence = kb, cameras, equivalence
        self.tau_hi, self.tau_lo, self.margin, self.alias_embed = tau_hi, tau_lo, margin, alias_embed

    def _bound(self, fact: MemoryFact, via: str, phrase: str | None = None) -> Bound:
        if phrase is not None:
            self.kb.add_alias(fact.id, phrase)  # the next paraphrase of this is an exact hit
        self.kb.touch(fact.id)
        return Bound(self.kb.get(fact.id), via)

    async def resolve(self, ref: Referent) -> Resolution:
        phrase = normalize(ref.text)
        if not phrase:
            return Unknown()

        exact = self.kb.find_exact(ref.role, phrase)
        if len(exact) == 1:
            return self._bound(exact[0], "exact")
        if len(exact) > 1:
            return Ambiguous(exact)

        if self.alias_embed:
            hits = self.kb.search(phrase, ref.role)
            if hits and hits[0].similarity >= self.tau_hi:
                close = [h for h in hits if h.similarity >= self.tau_hi and hits[0].similarity - h.similarity <= self.margin]
                if len(close) > 1:
                    return Ambiguous([h.fact for h in close])
                return self._bound(hits[0].fact, "embedding", ref.text)
            if hits and hits[0].similarity >= self.tau_lo and self._equivalence is not None:
                best = hits[0]
                try:
                    same = await self._equivalence(phrase, best.phrase, ref.role)
                except Exception as exc:  # noqa: BLE001 - any gateway failure must degrade to "ask the user once"
                    log.warning("equivalence check failed for %r: %s", phrase, exc)
                    same = False
                if same:
                    return self._bound(best.fact, "equivalence", ref.text)

        if ref.role == "place":
            for cam in self._cameras():
                if normalize(cam.name) == phrase:
                    ephemeral = MemoryFact(
                        id=f"cam:{cam.id}", kind="place", canonical=cam.name, aliases=[],
                        binding={"camera_id": cam.id}, source="statement", created_at=cam.t0,
                    )
                    return Bound(ephemeral, "camera", persisted=False)
        return Unknown()
