"""Ground-truth query files (PLAN section 9.3): `eval/queries/<set>.yaml`."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

QUERIES_DIR = Path(__file__).resolve().parent / "queries"
Split = Literal["dev", "test", "judge_sim"]


class GroundTruthHit(BaseModel):
    camera_id: str
    start: float
    end: float

    @field_validator("start", "end", mode="before")
    @classmethod
    def _to_epoch(cls, value: Any) -> float:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("ground-truth times need a UTC offset, e.g. 2026-10-09T09:14:00+05:30")
            return value.timestamp()
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                raise ValueError(f"ground-truth time {value!r} needs a UTC offset")
            return parsed.timestamp()
        return float(value)


class Expected(BaseModel):
    verdict: Literal["yes", "no", "found", "not_found", "partial", "count"]
    hits: list[GroundTruthHit] = Field(default_factory=list)
    path: list[GroundTruthHit] = Field(default_factory=list)  # ordered hops for path queries
    count: int | None = None

    @field_validator("verdict", mode="before")
    @classmethod
    def _yaml_booleans(cls, value: Any) -> Any:
        return {True: "yes", False: "no"}.get(value, value) if isinstance(value, bool) else value  # bare yes/no


class QueryItem(BaseModel):
    id: str
    text: str
    workspace: str
    intent: Literal["exists", "list", "count", "first", "last", "path", "describe", "standing"]
    first_time_requires_clarify: list[str] = Field(default_factory=list)
    clarify_answer: dict[str, Any] | None = None
    expected: Expected
    tags: list[str] = Field(default_factory=list)
    split: Split = "dev"

    @property
    def is_negative(self) -> bool:
        return self.intent not in ("count", "path") and not self.expected.hits

    @property
    def is_retrieval(self) -> bool:
        return self.intent not in ("count", "path", "standing") and bool(self.expected.hits)


def load_queries(paths: list[Path] | None = None, split: Split | None = None) -> list[QueryItem]:
    """Load every `*.yaml` in eval/queries (files starting with `_` are examples and skipped)."""
    files = paths or sorted(p for p in QUERIES_DIR.glob("*.yaml") if not p.name.startswith("_"))
    items: list[QueryItem] = []
    seen: set[str] = set()
    for path in files:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise ValueError(f"{path}: expected a list of queries")
        for entry in raw:
            item = QueryItem.model_validate(entry)
            if item.id in seen:
                raise ValueError(f"{path}: duplicate query id {item.id}")
            seen.add(item.id)
            if split is None or item.split == split:
                items.append(item)
    return items
