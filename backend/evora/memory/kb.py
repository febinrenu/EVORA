"""Knowledge base: memory facts in SQLite plus their alias vectors in LanceDB."""
from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Any

import lancedb
from contracts.models import MemoryFact

from evora.core.db import Database
from evora.core.vectors import ensure_tables
from evora.memory.embedder import DIM, TextEmbedder

KINDS = ("place", "object", "time")
# possessives (my, our) are kept on purpose: "my car" is a specific car, "a car" is not
_LEADING = re.compile(r"^(?:(?:at|in|near|by|on)(?:\s+|$))?(?:(?:the|a|an|this|that)(?:\s+|$))?")
_FACT_ID = re.compile(r"^mf_[0-9a-f]{8}$")


class KBError(ValueError):
    pass


class FactNotFound(KeyError):
    pass


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and leading preposition/determiner: 'At the Main Gate!' -> 'main gate'."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^\w\s]", " ", t).replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return _LEADING.sub("", t).strip()


# words that name a kind of place rather than the place itself: "the loading bay area" is "the loading bay"
_PLACE_SUFFIX = re.compile(r"\s+(?:area|zone|region|section|spot|space|side|part|place|location|bit)$")


def place_core(phrase: str) -> str:
    """A normalized place name without a trailing generic word, never emptied: 'loading bay area' -> 'loading bay'."""
    t = normalize(phrase)
    while True:
        shorter = _PLACE_SUFFIX.sub("", t).strip()
        if not shorter or shorter == t:
            return t
        t = shorter


@dataclass(frozen=True)
class AliasHit:
    fact: MemoryFact
    phrase: str
    similarity: float


class KnowledgeBase:
    def __init__(self, db: Database, store: lancedb.DBConnection, embedder: TextEmbedder):
        self.db, self.store, self.embedder = db, store, embedder
        ensure_tables(store, {"embed_dim_text": DIM}, only={"aliases"})
        self._sync_index()

    # --- vector index ---
    def _table(self):
        return self.store.open_table("aliases")

    def _phrases(self, fact: MemoryFact) -> list[str]:
        seen: dict[str, str] = {}
        for p in [fact.canonical, *fact.aliases]:
            seen.setdefault(normalize(p), normalize(p))
        return [p for p in seen if p]

    def _index(self, fact: MemoryFact, phrases: list[str]) -> None:
        if not phrases:
            return
        vecs = self.embedder.embed(phrases)
        self._table().add([{"vector": v.tolist(), "fact_id": fact.id, "alias": p} for v, p in zip(vecs, phrases, strict=True)])

    def _unindex(self, fact_id: str) -> None:
        if not _FACT_ID.match(fact_id):
            raise KBError("invalid fact id")
        self._table().delete(f"fact_id = '{fact_id}'")

    def _sync_index(self) -> None:
        """Rebuild the alias vectors when the embedding model changed: vector spaces must never be mixed."""
        if self.db.get_meta("embed_model_text") == self.embedder.name:
            return
        self.store.drop_table("aliases")
        ensure_tables(self.store, {"embed_dim_text": DIM}, only={"aliases"})
        for fact in self.list():
            self._index(fact, self._phrases(fact))
        self.db.set_meta("embed_model_text", self.embedder.name)

    def search(self, phrase: str, kind: str, k: int = 5) -> list[AliasHit]:
        """Nearest live facts of this kind, best alias per fact, by cosine similarity."""
        table = self._table()
        if table.count_rows() == 0:
            return []
        vec = self.embedder.embed([phrase])[0].tolist()
        rows = table.search(vec).metric("cosine").limit(k * 3).to_list()
        best: dict[str, AliasHit] = {}
        for r in rows:
            try:
                fact = self.get(r["fact_id"])
            except FactNotFound:
                continue
            if fact.kind != kind or fact.superseded_by:
                continue
            sim = 1.0 - float(r["_distance"])
            if fact.id not in best or sim > best[fact.id].similarity:
                best[fact.id] = AliasHit(fact, r["alias"], sim)
        return sorted(best.values(), key=lambda h: -h.similarity)[:k]

    # --- rows ---
    @staticmethod
    def _to_fact(row: sqlite3.Row, inferred: list[str]) -> MemoryFact:
        return MemoryFact(
            id=row["id"], kind=row["kind"], canonical=row["canonical"], aliases=json.loads(row["aliases"]),
            inferred_aliases=inferred, binding=json.loads(row["binding"]), source=row["source"],
            created_at=row["created_at"], last_used_at=row["last_used_at"], use_count=row["use_count"] or 0,
            superseded_by=row["superseded_by"],
        )

    @staticmethod
    def _inferred(c: sqlite3.Connection) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in c.execute("SELECT fact_id, alias FROM memory_inferred ORDER BY created_at, alias"):
            out.setdefault(r["fact_id"], []).append(r["alias"])
        return out

    def get(self, fact_id: str) -> MemoryFact:
        with self.db.read() as c:
            row = c.execute("SELECT * FROM memory_facts WHERE id=?", (fact_id,)).fetchone()
            inferred = self._inferred(c).get(fact_id, [])
        if row is None:
            raise FactNotFound(fact_id)
        return self._to_fact(row, inferred)

    def list(self, kind: str | None = None, include_superseded: bool = False) -> list[MemoryFact]:
        sql, args = "SELECT * FROM memory_facts WHERE 1=1", []
        if kind:
            sql, args = sql + " AND kind=?", [kind]
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        with self.db.read() as c:
            inferred = self._inferred(c)
            return [self._to_fact(r, inferred.get(r["id"], [])) for r in c.execute(sql + " ORDER BY created_at, id", args)]

    def find_exact(self, kind: str, phrase: str) -> list[MemoryFact]:
        target = normalize(phrase)
        return [f for f in self.list(kind) if target and target in self._phrases(f)]

    def find_place_variant(self, phrase: str) -> list[MemoryFact]:
        """Places whose name matches once a generic word ('area', 'zone', 'side' ...) is dropped from either side."""
        core = place_core(phrase)
        if not core:
            return []
        return [f for f in self.list("place") if any(place_core(p) == core for p in self._phrases(f))]

    def create(
        self, kind: str, canonical: str, binding: dict[str, Any], source: str = "statement",
        aliases: list[str] | None = None,
    ) -> MemoryFact:
        if kind not in KINDS:
            raise KBError(f"kind must be one of {KINDS}")
        if not normalize(canonical):
            raise KBError("canonical name cannot be empty")
        fact_id = f"mf_{uuid.uuid4().hex[:8]}"
        with self.db.write() as c:
            c.execute(
                "INSERT INTO memory_facts(id,kind,canonical,aliases,binding,source,created_at) VALUES(?,?,?,?,?,?,?)",
                (fact_id, kind, canonical.strip(), json.dumps(aliases or []), json.dumps(binding), source, time.time()),
            )
        fact = self.get(fact_id)
        self._index(fact, self._phrases(fact))
        return fact

    def add_alias(self, fact_id: str, phrase: str, inferred: bool = False) -> bool:
        """Remember another way of saying the same thing. `inferred` marks a silent guess by the resolver.

        Returns False when the phrase was already known.
        """
        fact = self.get(fact_id)
        if not normalize(phrase) or normalize(phrase) in self._phrases(fact):
            return False
        with self.db.write() as c:
            c.execute("UPDATE memory_facts SET aliases=? WHERE id=?", (json.dumps([*fact.aliases, phrase.strip()]), fact_id))
            if inferred:
                c.execute(
                    "INSERT OR IGNORE INTO memory_inferred(fact_id, alias, created_at) VALUES(?,?,?)",
                    (fact_id, phrase.strip(), time.time()),
                )
        self._index(fact, [normalize(phrase)])
        return True

    def confirm_alias(self, fact_id: str, phrase: str) -> bool:
        """The user accepts a guessed alias as correct. Returns False if it was not a guess."""
        target = normalize(phrase)
        fact = self.get(fact_id)
        guesses = [a for a in fact.inferred_aliases if normalize(a) == target]
        with self.db.write() as c:
            for a in guesses:
                c.execute("DELETE FROM memory_inferred WHERE fact_id=? AND alias=?", (fact_id, a))
        return bool(guesses)

    def update(
        self, fact_id: str, canonical: str | None = None, aliases: list[str] | None = None,
        binding: dict[str, Any] | None = None,
    ) -> MemoryFact:
        fact = self.get(fact_id)
        if canonical is not None and not normalize(canonical):
            raise KBError("canonical name cannot be empty")
        with self.db.write() as c:
            c.execute(
                "UPDATE memory_facts SET canonical=?, aliases=?, binding=? WHERE id=?",
                (canonical.strip() if canonical is not None else fact.canonical,
                 json.dumps(aliases if aliases is not None else fact.aliases),
                 json.dumps(binding if binding is not None else fact.binding), fact_id),
            )
        if aliases is not None:
            keep = {normalize(a) for a in aliases}
            with self.db.write() as c:
                for a in fact.inferred_aliases:
                    if normalize(a) not in keep:
                        c.execute("DELETE FROM memory_inferred WHERE fact_id=? AND alias=?", (fact_id, a))
        updated = self.get(fact_id)
        self._unindex(fact_id)
        self._index(updated, self._phrases(updated))
        return updated

    def delete(self, fact_id: str) -> None:
        self.get(fact_id)
        self._unindex(fact_id)
        with self.db.write() as c:
            c.execute("UPDATE zones SET fact_id=NULL WHERE fact_id=?", (fact_id,))
            c.execute("DELETE FROM memory_facts WHERE id=?", (fact_id,))

    def touch(self, fact_id: str) -> None:
        with self.db.write() as c:
            c.execute(
                "UPDATE memory_facts SET use_count=COALESCE(use_count,0)+1, last_used_at=? WHERE id=?", (time.time(), fact_id)
            )

    def supersede(self, fact_id: str, new_binding: dict[str, Any], source: str = "correction") -> MemoryFact:
        """Replace a fact's binding. The new fact keeps the name and aliases; the old one stops matching."""
        old = self.get(fact_id)
        if old.superseded_by:
            raise KBError("fact was already superseded")
        self._unindex(fact_id)
        guessed = {normalize(a) for a in old.inferred_aliases}  # learned under the binding that was just corrected
        kept = [a for a in old.aliases if normalize(a) not in guessed]
        new = self.create(old.kind, old.canonical, new_binding, source, kept)
        with self.db.write() as c:
            c.execute("UPDATE memory_facts SET superseded_by=? WHERE id=?", (new.id, fact_id))
        return self.get(new.id)
