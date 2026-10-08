"""Describe intent: "what happened at the gate yesterday evening?" answered from stored events.

The facts come from the events and tracks tables, never from a model. A language model may
phrase them (gpt-oss-120b), but only under three rules enforced here: every sentence cites
fact ids that exist, no sentence may contain a clock time (the code adds the timestamps from
the cited facts), and anything that breaks a rule is thrown away in favour of a deterministic
summary built from the same facts.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, tzinfo
from typing import Protocol

from contracts.models import TimeWindow
from pydantic import BaseModel

from evora.core.db import Database
from evora.llm.schemas import LLMError
from evora.query.compose import clock, pluralize
from evora.query.logic import instant_in_window, span_in_window

log = logging.getLogger("evora.query.describe")

MAX_FACTS = 24          # what the model is shown (the token budget is small)
NOTABLE = ("cross_line", "enter_zone", "exit_zone", "dwell")
VERBS = {"cross_line": "crossed", "enter_zone": "entered", "exit_zone": "left", "dwell": "stayed at",
         "appear": "appeared at", "disappear": "left the view of"}
MAX_SENTENCES = 4
MAX_SENTENCE_CHARS = 280
_CLOCK = re.compile(r"\b\d{1,2}[:.]\d{2}\b|\b\d{1,2}\s?(?:am|pm)\b", re.IGNORECASE)
_FACT_ID = re.compile(r"E\d+")


@dataclass(frozen=True)
class Fact:
    id: str
    camera_id: str
    track_id: str
    kind: str
    t: float
    text: str            # "a person in red crossed the main gate"
    track_start: float
    track_end: float
    global_id: str | None = None


@dataclass
class Overview:
    tracks_by_class: dict[str, int] = field(default_factory=dict)
    camera_ids: list[str] = field(default_factory=list)
    first_t: float | None = None
    last_t: float | None = None
    total_events: int = 0


@dataclass(frozen=True)
class DescribedSentence:
    text: str
    fact_ids: tuple[str, ...]


class Narration(BaseModel):
    class Line(BaseModel):
        text: str
        cites: list[str] = []

    sentences: list[Line] = []


class Gateway(Protocol):
    async def chat_json(self, task: str, messages: list[dict], schema: type[Narration]) -> Narration: ...


# ----------------------------------------------------------------------- facts
def _who(cls: str, attrs: Mapping[str, object]) -> str:
    colour = attrs.get("upper_color") or attrs.get("color")
    if attrs.get("is_ir"):
        colour = None
    if cls == "person":
        carrying = [c.replace("_", " ") for c in (attrs.get("carrying") or [])]  # type: ignore[union-attr]
        text = "a person" + (f" in {colour}" if colour else "")
        return text + (f" carrying a {carrying[0]}" if carrying else "")
    kind = attrs.get("vehicle_type") if attrs.get("vehicle_type") and attrs.get("vehicle_type") != cls else cls
    return f"a {colour + ' ' if colour else ''}{kind}"


def _zone_labels(db: Database) -> dict[str, str]:
    with db.read() as c:
        rows = c.execute("SELECT z.id AS zid, m.canonical AS name FROM zones z JOIN memory_facts m ON m.id = z.fact_id "
                         "WHERE m.superseded_by IS NULL").fetchall()
    return {r["zid"]: r["name"] for r in rows}


def gather_facts(db: Database, camera_ids: set[str], window: TimeWindow | None, tz: tzinfo = UTC,
                 limit: int = MAX_FACTS) -> tuple[list[Fact], Overview]:
    """Events in scope as readable, citable facts (time ordered), plus counts of what was seen."""
    cameras: dict[str, str] = {}
    with db.read() as c:
        cameras = {r["id"]: r["name"] for r in c.execute("SELECT id, name FROM cameras")}
        scope = [cid for cid in cameras if not camera_ids or cid in camera_ids]
        if not scope:
            return [], Overview()
        marks = ",".join("?" * len(scope))
        tracks = {r["id"]: dict(r) for r in c.execute(
            f"SELECT id, camera_id, cls, t_start, t_end, attrs, global_id FROM tracks WHERE camera_id IN ({marks})", scope)}
        events = [dict(r) for r in c.execute(
            f"SELECT id, camera_id, track_id, kind, zone_id, t FROM events WHERE camera_id IN ({marks}) ORDER BY t", scope)]
    tracks = {tid: t for tid, t in tracks.items() if span_in_window(t["t_start"], t["t_end"], window, tz)}
    overview = Overview(
        tracks_by_class=dict(Counter(t["cls"] for t in tracks.values())), camera_ids=scope,
        first_t=min((t["t_start"] for t in tracks.values()), default=None),
        last_t=max((t["t_end"] for t in tracks.values()), default=None))
    scoped = [e for e in events if e["track_id"] in tracks and instant_in_window(e["t"], window, tz)]
    overview.total_events = len(scoped)
    notable = [e for e in scoped if e["kind"] in NOTABLE]
    chosen = notable if notable else [e for e in scoped if e["kind"] == "appear"]
    if len(chosen) > limit:  # keep the first and last, spread the rest evenly
        step = (len(chosen) - 1) / (limit - 1)
        chosen = [chosen[round(i * step)] for i in range(limit)]
    labels = _zone_labels(db)
    facts: list[Fact] = []
    for n, e in enumerate(chosen, start=1):
        tr = tracks[e["track_id"]]
        where = labels.get(e["zone_id"]) or cameras[e["camera_id"]]
        who = _who(tr["cls"], json.loads(tr["attrs"] or "{}"))
        place = f"the {where}" if e["zone_id"] in labels else f"the {where} camera"
        facts.append(Fact(f"E{n}", e["camera_id"], e["track_id"], e["kind"], e["t"],
                          f"{who} {VERBS[e['kind']]} {place}", tr["t_start"], tr["t_end"], tr["global_id"]))
    return facts, overview


# ------------------------------------------------------------------ summaries
def _count_phrase(by_class: Mapping[str, int]) -> str:
    order = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]
    parts = []
    for cls in sorted(by_class, key=lambda c: (order.index(c) if c in order else 99, c)):
        n = by_class[cls]
        parts.append(f"{n} {cls if n == 1 else pluralize(cls)}")
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] if parts else "nothing"


def deterministic_sentences(facts: Sequence[Fact], overview: Overview, where: str, when: str, tz: tzinfo,
                            cam_names: Mapping[str, str], source_names: Mapping[str, str] | None = None,
                            with_date: bool = False) -> list[DescribedSentence]:
    """A plain summary from the same facts the model would see."""
    suffix = f" {when}" if when else ""
    scope = f" at {where}" if where else ""
    if not overview.tracks_by_class:
        return [DescribedSentence(f"No activity was recorded{scope}{suffix}.", ())]
    out: list[DescribedSentence] = []
    lead = f"{_count_phrase(overview.tracks_by_class)} were tracked{scope}{suffix}."
    if facts:
        out.append(DescribedSentence(lead, (facts[0].id, facts[-1].id) if len(facts) > 1 else (facts[0].id,)))
        for f in facts[:5]:
            out.append(DescribedSentence(
                f"{f.text[:1].upper() + f.text[1:]} at {clock(f.t, tz, with_date)}.", (f.id,)))
        if len(facts) > 5:
            out.append(DescribedSentence(f"{len(facts) - 5} more events are listed in the evidence.", (facts[5].id,)))
    else:
        out.append(DescribedSentence(lead + " None of them crossed or entered a marked area.", ()))
    return out


def valid_narration(lines: Sequence[Narration.Line], facts: Mapping[str, Fact]) -> list[DescribedSentence] | None:
    """Accept a model's wording only if every rule holds; otherwise None (use the deterministic summary)."""
    if not lines or len(lines) > MAX_SENTENCES:
        return None
    out: list[DescribedSentence] = []
    for line in lines:
        text = " ".join(line.text.split())
        cites = tuple(dict.fromkeys(c for c in line.cites if _FACT_ID.fullmatch(c)))
        if not text or len(text) > MAX_SENTENCE_CHARS or not cites or any(c not in facts for c in cites):
            return None
        if _CLOCK.search(text):
            return None  # times come from the facts, never from the model
        out.append(DescribedSentence(text if text.endswith((".", "!", "?")) else text + ".", cites))
    return out


def narration_messages(facts: Sequence[Fact], overview: Overview, where: str, when: str) -> list[dict]:
    listing = "\n".join(f"[{f.id}] {f.text}" for f in facts)
    return [
        {"role": "system", "content": (
            "You summarise what a security camera system saw, in 2 to 4 short factual sentences. "
            "Use only the numbered facts you are given. After each sentence put the fact ids it rests on in "
            "`cites`. Never write a clock time or a date (the system adds them). Never mention anything that is "
            'not in the facts. Reply only with JSON: {"sentences": [{"text": "...", "cites": ["E1"]}]}.')},
        {"role": "user", "content": (
            f"Scope: {where or 'all cameras'} {when}".strip() + f"\nTracked: {_count_phrase(overview.tracks_by_class)}\n"
            f"Facts:\n{listing}")},
    ]


async def narrate(gateway: Gateway | None, facts: Sequence[Fact], overview: Overview, where: str,
                  when: str) -> list[DescribedSentence] | None:
    """The model's wording if it passes every rule, else None."""
    if gateway is None or not facts:
        return None
    try:
        result = await gateway.chat_json("describe", narration_messages(facts, overview, where, when), Narration)
    except LLMError as exc:
        log.info("narration unavailable (%s); using the deterministic summary", exc)
        return None
    checked = valid_narration(result.sentences, {f.id: f for f in facts})
    if checked is None:
        log.info("narration rejected by the grounding rules; using the deterministic summary")
    return checked
