"""Known places ledger: list, add, edit, correct and forget memory facts."""
from __future__ import annotations

from typing import Any, Literal

from contracts.models import MemoryFact
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from evora.api.context import AppContext
from evora.evidence import audit
from evora.memory.kb import FactNotFound, KBError


class FactIn(BaseModel):
    kind: Literal["place", "object", "time"]
    canonical: str = Field(min_length=1, max_length=120)
    aliases: list[str] = []
    binding: dict[str, Any] = {}
    source: Literal["clarification", "statement", "correction", "import"] = "statement"


class FactPatch(BaseModel):
    canonical: str | None = Field(default=None, min_length=1, max_length=120)
    aliases: list[str] | None = None
    binding: dict[str, Any] | None = None


def make_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/memory")
    kb = ctx.memory.kb

    def _missing(fid: str) -> HTTPException:
        return HTTPException(404, f"unknown fact: {fid}")

    @router.get("")
    def list_facts(kind: str | None = None, include_superseded: bool = False) -> list[MemoryFact]:
        return kb.list(kind, include_superseded)

    @router.post("")
    def add_fact(body: FactIn) -> MemoryFact:
        try:
            fact = kb.create(body.kind, body.canonical, body.binding, body.source, body.aliases)
        except KBError as exc:
            raise HTTPException(422, str(exc)) from None
        audit.record(ctx.db, "memory_add", {"fact_id": fact.id, "canonical": fact.canonical})
        return fact

    @router.patch("/{fid}")
    def patch_fact(fid: str, body: FactPatch) -> MemoryFact:
        try:
            fact = kb.get(fid)
            if fact.superseded_by:
                raise HTTPException(409, "this fact was already replaced by a newer one")
            if body.canonical is not None or body.aliases is not None:
                fact = kb.update(fid, body.canonical, body.aliases)
            if body.binding is not None and body.binding != fact.binding:  # a new binding is a correction
                fact = kb.supersede(fid, body.binding, "correction")
                audit.record(ctx.db, "memory_correct", {"old": fid, "new": fact.id})
            return fact
        except FactNotFound:
            raise _missing(fid) from None
        except KBError as exc:
            raise HTTPException(422, str(exc)) from None

    @router.delete("/{fid}")
    def delete_fact(fid: str) -> list[MemoryFact]:
        try:
            kb.delete(fid)
        except FactNotFound:
            raise _missing(fid) from None
        audit.record(ctx.db, "memory_delete", {"fact_id": fid})
        return kb.list()

    return router
