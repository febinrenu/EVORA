"""Deterministic answer text.

Every sentence is built from a template over evidence we actually hold; nothing is
free-form. Each factual sentence carries the ids of the evidence it rests on, and
`validate` rejects any that cite nothing or cite something we do not have. The text the
UI shows is just the sentences joined, with no markup.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, tzinfo
from typing import Literal, Protocol

from contracts.models import Evidence, PathHop, QueryPlan

IR_NOTE_THRESHOLD = 0.3
COLOUR_WORDS = {"black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"}


class CameraLike(Protocol):
    id: str
    name: str
    layers: Sequence[str]
    ir_fraction: float | None


@dataclass(frozen=True)
class Sentence:
    text: str
    evidence: tuple[str, ...] = ()
    kind: Literal["fact", "negative", "note"] = "fact"  # only "fact" must cite evidence


@dataclass
class Composed:
    verdict: Literal["yes", "no", "found", "not_found", "partial", "count"]
    sentences: list[Sentence]
    notes: list[str] = field(default_factory=list)
    count: int | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


class UngroundedAnswer(ValueError):
    """A factual sentence cites no evidence, or evidence that does not exist."""


# ------------------------------------------------------------- formatting
def clock(t: float, tz: tzinfo = UTC, with_date: bool = False) -> str:
    dt = datetime.fromtimestamp(t, tz)
    stamp = dt.strftime("%H:%M:%S")
    return f"{dt.day} {dt.strftime('%b')} {stamp}" if with_date else stamp


def offset_label(offset_s: float) -> str:
    total = max(0, int(offset_s))
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def pluralize(noun: str) -> str:
    irregular = {"person": "people", "man": "men", "woman": "women", "child": "children"}
    if noun in irregular:
        return irregular[noun]
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def stamp(ev: Evidence, tz: tzinfo, with_date: bool, source: str | None) -> str:
    """The two timestamps we always show: the camera's wall clock and the offset into its file."""
    return (f"{clock(ev.t_peak, tz, with_date)} on {ev.camera_name} "
            f"({offset_label(ev.offset_s)} into {source or ev.camera_name})")


# ------------------------------------------------------------- plan -> words
_PAST = {"pass_through": "passed through", "enter": "entered", "exit": "left", "dwell": "stayed near",
         "appear": "appeared at", "any": "was seen at"}
_GERUND = {"pass_through": "passing through", "enter": "entering", "exit": "leaving", "dwell": "staying near",
           "appear": "appearing at", "any": "seen at"}
_BARE_PAST = {"pass_through": "passed by", "enter": "entered", "exit": "left", "dwell": "stayed",
              "appear": "appeared", "any": "was seen"}  # when neither a place nor a camera is named
_BARE_GERUND = {"pass_through": "passing by", "enter": "entering", "exit": "leaving", "dwell": "staying",
                "appear": "appearing", "any": "seen"}


def subject(plan: QueryPlan) -> str:
    """'a red car' (with its article), from the planner's own visual description."""
    parts = [t.embed_text.removeprefix("a photo of ").strip() or t.noun for t in plan.targets]
    return " and ".join(parts) if parts else "activity"


def bare_subject(plan: QueryPlan) -> str:
    text = subject(plan)
    for article in ("a ", "an "):
        if text.startswith(article):
            return text[len(article):]
    return text


def _where(plan: QueryPlan, evidence: Sequence[Evidence], names: Mapping[str, str]) -> str:
    if plan.place is not None:
        text = plan.place.text
        return text if text.startswith(("the ", "a ")) else f"the {text}"
    if plan.camera_ids:
        label = [names.get(c) or next((e.camera_name for e in evidence if e.camera_id == c), c)
                 for c in plan.camera_ids]
        return " or ".join(f"the {n} camera" for n in label)
    return ""


def _clause(plan: QueryPlan, where: str, past: bool) -> str:
    table = (_PAST if past else _GERUND) if where else (_BARE_PAST if past else _BARE_GERUND)
    return f"{table[plan.action]} {where}".strip()


def _when(plan: QueryPlan) -> str:
    return f" {plan.time.phrase.strip()}" if plan.time and plan.time.phrase else ""


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:]


# ------------------------------------------------------------------ notes
def make_notes(plan: QueryPlan, cameras: Sequence[CameraLike], evidence: Sequence[Evidence]) -> list[str]:
    notes: list[str] = []
    relevant_ids = set(plan.camera_ids) | {e.camera_id for e in evidence}
    for cam in cameras:
        if relevant_ids and cam.id not in relevant_ids:
            continue
        layers = set(cam.layers)
        if "L1" not in layers:
            notes.append(f"{cam.name} is still being indexed, so results from it may be incomplete.")
    wants_colour = any(a in COLOUR_WORDS for t in plan.targets for a in t.attributes)
    if wants_colour:
        for cam in cameras:
            if (relevant_ids and cam.id not in relevant_ids) or cam.ir_fraction is None:
                continue
            if cam.ir_fraction >= IR_NOTE_THRESHOLD:
                notes.append(f"{cam.name} has night or infrared footage, where colours are unreliable.")
    return notes


# ---------------------------------------------------------------- compose
def compose(
    plan: QueryPlan,
    evidence: Sequence[Evidence],
    *,
    count: int | None = None,
    nearest_miss: Evidence | None = None,
    path: Sequence[PathHop] = (),
    cameras: Sequence[CameraLike] = (),
    source_names: Mapping[str, str] | None = None,
    tz: tzinfo = UTC,
    reference_now: float | None = None,
    partial: bool = False,
) -> Composed:
    """Build the verdict and the grounded text. `evidence` is already ordered best/first/last."""
    sources = dict(source_names or {})
    names = {c.id: c.name for c in cameras}
    shown = list(evidence) + ([nearest_miss] if nearest_miss else [])
    days = {datetime.fromtimestamp(e.t_peak, tz).date() for e in shown}
    ref_day = datetime.fromtimestamp(reference_now, tz).date() if reference_now is not None else None
    with_date = len(days) > 1 or (ref_day is not None and days != {ref_day} and bool(days))

    def at(ev: Evidence) -> str:
        return stamp(ev, tz, with_date, sources.get(ev.camera_id))

    where = _where(plan, evidence, names)
    notes = make_notes(plan, cameras, evidence)
    sentences: list[Sentence] = []
    ids = tuple(e.id for e in evidence)
    intent = plan.intent

    if intent == "count":
        n = count if count is not None else len(evidence)
        noun = plan.targets[0].noun if plan.targets else "item"
        label = f"{n} matching {noun if n == 1 else pluralize(noun)}"
        clause = _clause(plan, where, past=False)
        sentences.append(Sentence(f"Counted {label} {clause}{_when(plan)}.".replace("  ", " "), ids,
                                  "fact" if ids else "negative"))
        if evidence:
            first = min(evidence, key=lambda e: e.t_peak)
            sentences.append(Sentence(f"First at {at(first)}.", (first.id,)))
        return Composed("count", sentences, notes, n)

    if intent == "path" and path:
        hop_ids = tuple(h.evidence_id for h in path)
        route = " → ".join(f"{h.camera_name} {clock(h.t_in, tz, with_date)}" for h in path)
        sentences.append(Sentence(f"Path of {bare_subject(plan)}: {route}.", hop_ids))
        return Composed("partial" if partial else "found", sentences, notes)

    if not evidence:
        past = _clause(plan, where, past=True)
        verdict = "no" if intent == "exists" else "not_found"
        sentences.append(Sentence(f"No {bare_subject(plan)} {past}{_when(plan)}.", (), "negative"))
        if nearest_miss is not None:
            sentences.append(Sentence(
                f"Closest: {at(nearest_miss)}, match score {nearest_miss.score:.2f}.", (nearest_miss.id,)))
        return Composed(verdict, sentences, notes)

    best = evidence[0]
    if intent == "exists":
        sentences.append(Sentence(
            f"Yes. {_sentence_case(subject(plan))} {_clause(plan, where, past=True)}{_when(plan)}: {at(best)}.",
            (best.id,)))
        if len(evidence) > 1:
            sentences.append(Sentence(f"{len(evidence) - 1} more match{'es' if len(evidence) > 2 else ''} found.",
                                      ids[1:]))
        return Composed("partial" if partial else "yes", sentences, notes)

    if intent in ("first", "last"):
        word = "first" if intent == "first" else "last"
        sentences.append(Sentence(
            f"The {word} match for {bare_subject(plan)} {_clause(plan, where, past=False)}{_when(plan)} "
            f"was at {at(best)}.".replace("  ", " "), (best.id,)))
        return Composed("partial" if partial else "found", sentences, notes)

    # list, describe and anything else that returns evidence
    n = len(evidence)
    sentences.append(Sentence(
        f"Found {n} match{'es' if n != 1 else ''} for {bare_subject(plan)} "
        f"{_clause(plan, where, past=False)}{_when(plan)}.".replace("  ", " "), ids))
    sentences.append(Sentence(f"Best match: {at(best)}.", (best.id,)))
    return Composed("partial" if partial else "found", sentences, notes)


# -------------------------------------------------------------- validator
def validate(sentences: Sequence[Sentence], known_ids: set[str]) -> None:
    """Raise UngroundedAnswer unless every factual sentence cites evidence we hold."""
    for s in sentences:
        if s.kind != "fact":
            continue
        if not s.evidence:
            raise UngroundedAnswer(f"sentence cites no evidence: {s.text!r}")
        missing = [i for i in s.evidence if i not in known_ids]
        if missing:
            raise UngroundedAnswer(f"sentence cites unknown evidence {missing}: {s.text!r}")


def compose_checked(plan: QueryPlan, evidence: Sequence[Evidence], **kw) -> Composed:
    """`compose` plus the validator. Cited ids must be in `evidence` (or be the nearest miss)."""
    out = compose(plan, evidence, **kw)
    known = {e.id for e in evidence}
    if kw.get("nearest_miss") is not None:
        known.add(kw["nearest_miss"].id)
    validate(out.sentences, known)
    return out
