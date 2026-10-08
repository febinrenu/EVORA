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
from typing import Any, Literal, Protocol

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
UNCONFIRMED_NOTE = "Nothing stored or checked shows"


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
    unconfirmed: bool = False,
    concurrent: Mapping[str, Mapping[str, int]] | None = None,
    appearances: int | None = None,
    unreadable: int = 0,
    action: str | None = None,
) -> Composed:
    """Build the verdict and the grounded text. `evidence` is already ordered best/first/last.

    `unconfirmed`: attributes were asked for but nothing (stored attributes, caption, visual check) supports
    them on any candidate, so the answer must not state them as fact.
    `action`: the question asks about an action that cannot be recognised (see query/actions.py); the answer then
    says so and lists who was there instead of claiming the action.
    """
    if action and plan.intent != "path":
        return _about_action(plan, evidence, action, count=count, nearest_miss=nearest_miss, path=path, cameras=cameras,
                             source_names=source_names, tz=tz, reference_now=reference_now, partial=partial,
                             unconfirmed=unconfirmed, concurrent=concurrent, appearances=appearances,
                             unreadable=unreadable)
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

    if intent == "count" and unreadable and not count and plan.targets and plan.targets[0].attributes:
        # attributes were asked for and none could be read: say so instead of answering zero
        wanted = " ".join(plan.targets[0].attributes)
        noun = plan.targets[0].noun
        being = f"wearing {wanted}" if noun in ("person", "people") else wanted
        sentences.append(Sentence(
            f"I can't tell how many {pluralize(noun)} are {being}: the colour could not be read for "
            f"{unreadable} of them (too small, too dark or too far away).", (), "negative"))
        notes.append("Nothing is counted as zero here: the colour is unknown, not absent.")
        return Composed("partial", sentences, notes, None)

    if intent == "count" and concurrent:
        cam_names = {**names, **{e.camera_id: e.camera_name for e in evidence}}
        typical = max(v["typical"] for v in concurrent.values())
        peak = max(v["peak"] for v in concurrent.values())
        noun = plan.targets[0].noun if plan.targets else "item"
        label = noun if typical == 1 else pluralize(noun)
        spot = f" of {where}" if where else ""
        sentence = (f"About {typical} {label} {'was' if typical == 1 else 'were'} in view{spot} at the same time"
                    f"{_when(plan)}")
        sentence += f" (up to {peak} at once)." if peak > typical else "."
        sentences.append(Sentence(sentence.replace("  ", " "), tuple(e.id for e in evidence), "fact" if evidence else "negative"))
        if len(concurrent) > 1:
            per = ", ".join(f"{cam_names.get(c, c)} {v['typical']}" for c, v in sorted(concurrent.items()))
            sentences.append(Sentence(f"Per camera: {per}.", (), "note"))
            notes.append("These cameras may show the same place, so the largest single-camera number is given, not the sum.")
        if appearances is not None and appearances > typical:
            notes.append(f"{appearances} separate appearances were tracked, but people who leave the view or are hidden for a "
                         "while come back as new tracks. The number above is how many were in view at the same time, "
                         "which is the closest the footage gets to how many people there are.")
        return Composed("count", sentences, notes, typical)

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
        if n > 1:
            notes.append("This counts separate appearances. Someone who leaves the view or is hidden for a while and returns "
                         "can be counted more than once, and small or distant objects can be missed.")
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
    if unconfirmed:
        partial = True
        wanted = ", ".join(plan.targets[0].attributes)
        notes.append(f"{UNCONFIRMED_NOTE} {wanted} on these candidates; they are the closest "
                     f"{plan.targets[0].noun} matches.")
    if intent == "exists" and unconfirmed:
        sentences.append(Sentence(
            f"I can't confirm {subject(plan)}. A {plan.targets[0].noun} {_clause(plan, where, past=True)}{_when(plan)}, "
            f"but {wanted} could not be established: {at(best)}.", (best.id,)))
        if len(evidence) > 1:
            sentences.append(Sentence(f"{len(evidence) - 1} more candidate{'s' if len(evidence) > 2 else ''}.", ids[1:]))
        return Composed("partial", sentences, notes)
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


ACTION_NOTE = "I recognise people and vehicles, where they go and what they wear and carry, not actions like this."


def _about_action(plan: QueryPlan, evidence: Sequence[Evidence], action: str, **kw) -> Composed:
    """An answer that does not claim an action it cannot see: who was there, with the evidence, verdict partial.

    With nobody there at all the plain negative stands (nobody there could have done it). A count stays a count of
    who was seen, marked partial.
    """
    lead = Sentence(f"I can't tell whether anyone was {action}: {ACTION_NOTE}", (), "note")
    inner = compose(plan, evidence, **kw)
    if not evidence or plan.intent == "count":
        inner.sentences.insert(0, lead)
        if plan.intent == "count":
            inner.verdict = "partial"
        return inner
    tz, sources = kw.get("tz", UTC), dict(kw.get("source_names") or {})
    names = {c.id: c.name for c in kw.get("cameras") or ()}
    days = {datetime.fromtimestamp(e.t_peak, tz).date() for e in evidence}
    ref = kw.get("reference_now")
    with_date = len(days) > 1 or (ref is not None and days != {datetime.fromtimestamp(ref, tz).date()})
    noun = plan.targets[0].noun if plan.targets else "object"
    where = _where(plan, evidence, names)
    best, ids = evidence[0], tuple(e.id for e in evidence)
    who = f"This is the {noun}" if len(evidence) == 1 else f"These are the {pluralize(noun)}"
    sentences = [lead, Sentence(
        f"{who} {_clause(plan, where, past=False)}{_when(plan)}: {stamp(best, tz, with_date, sources.get(best.camera_id))}."
        .replace("  ", " "), (best.id,))]
    if len(evidence) > 1:
        sentences.append(Sentence(f"{len(evidence) - 1} more sighting{'s' if len(evidence) > 2 else ''}.", ids[1:]))
    return Composed("partial", sentences, inner.notes, inner.count)


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


OBJECT_NOTE = ("Found by looking for {what} in {frames} stored frames with an open-vocabulary detector. Objects hidden "
               "behind people or furniture, and very small ones, can be missed; counts are what is in view in one frame, "
               "not a total over time.")


def compose_objects(
    plan: QueryPlan,
    cameras_summary: Sequence[Mapping[str, Any]],
    evidence: Sequence[Evidence],
    *,
    noun: str,
    colour: str | None,
    tz: tzinfo = UTC,
    source_names: Mapping[str, str] | None = None,
    reference_now: float | None = None,
) -> Composed:
    """Answer about objects found by the open-vocabulary search (not tracks).

    `cameras_summary` rows: camera_id, camera_name, typical, peak, least, frames, seen_in, breakdown {label: n}.
    """
    sources = dict(source_names or {})
    shown = list(evidence)
    days = {datetime.fromtimestamp(e.t_peak, tz).date() for e in shown}
    ref_day = datetime.fromtimestamp(reference_now, tz).date() if reference_now is not None else None
    with_date = len(days) > 1 or (ref_day is not None and days != {ref_day} and bool(days))
    where = _where(plan, evidence, {r["camera_id"]: r["camera_name"] for r in cameras_summary})
    spot = f" of {where}" if where else ""
    frames = sum(int(r["frames"]) for r in cameras_summary)
    adjective = f"{colour} " if colour else ""
    ids = tuple(e.id for e in evidence)
    notes = [OBJECT_NOTE.format(what=("everyday objects" if noun in ("object", "objects") else f"'{noun}'"), frames=frames)]
    if colour:
        notes.append(f"Colour is read from the middle of each detected object ({colour} when at least a quarter of it is "
                     f"{colour}); something in front of it can change that.")
    if 0 < frames < 3:
        notes.append(f"Only {frames} stored frame(s) were available, so this is a thin sample.")
    sentences: list[Sentence] = []
    best = max(cameras_summary, key=lambda r: r["typical"], default=None)
    found = bool(evidence) and best is not None and best["peak"] > 0
    label = lambda n: (singular_word(noun) if n == 1 else pluralize(singular_word(noun)))  # noqa: E731

    if plan.intent == "count":
        n = best["typical"] if best is not None else 0
        if not found or n == 0:
            sentences.append(Sentence(f"No {adjective}{pluralize(singular_word(noun))} were found{_in(where)} in the "
                                      f"{frames} frames looked at.", (), "negative"))
            return Composed("count", sentences, notes, 0)
        peak = best["peak"]
        text = f"About {n} {adjective}{label(n)} {'was' if n == 1 else 'were'} in view{spot}"
        text += f" (between {best['least']} and {peak} depending on the frame)." if peak > best["least"] else "."
        sentences.append(Sentence(text, ids, "fact"))
        mix = best.get("breakdown") or {}
        if noun in ("object", "objects") and mix:
            parts = ", ".join(f"{k} {v}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1]))
            sentences.append(Sentence(f"In the clearest frame: {parts}.", ids))
        if len(cameras_summary) > 1:
            per = ", ".join(f"{r['camera_name']} {r['typical']}" for r in sorted(cameras_summary, key=lambda r: r["camera_id"]))
            sentences.append(Sentence(f"Per camera: {per}.", (), "note"))
            notes.append("These cameras may show the same place, so the largest single-camera number is given, not the sum.")
        return Composed("count", sentences, notes, n)

    first = evidence[0] if evidence else None
    if not found or first is None:
        sentences.append(Sentence(f"No {adjective}{singular_word(noun)} was found{_in(where)} in the {frames} frames "
                                  "looked at.", (), "negative"))
        return Composed("no" if plan.intent == "exists" else "not_found", sentences, notes)
    stamp_text = stamp(first, tz, with_date, sources.get(first.camera_id))
    seen = f"seen in {best['seen_in']} of {best['frames']} frames"
    if plan.intent == "exists":
        sentences.append(Sentence(f"Yes. {_sentence_case(_article(adjective + singular_word(noun)))} was in view{spot}: "
                                  f"{stamp_text} ({seen}).", (first.id,)))
        return Composed("yes", sentences, notes)
    n = best["typical"]
    sentences.append(Sentence(f"Found {adjective}{label(n)}{spot}: {stamp_text} ({seen}).", ids))
    return Composed("found", sentences, notes)


def _in(where: str) -> str:
    return f" in {where}" if where else ""


def singular_word(noun: str) -> str:
    return "object" if noun in ("object", "objects", "thing", "things", "item", "items", "stuff") else noun


def _article(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def compose_checked(plan: QueryPlan, evidence: Sequence[Evidence], **kw) -> Composed:
    """`compose` plus the validator. Cited ids must be in `evidence` (or be the nearest miss)."""
    out = compose(plan, evidence, **kw)
    known = {e.id for e in evidence}
    if kw.get("nearest_miss") is not None:
        known.add(kw["nearest_miss"].id)
    validate(out.sentences, known)
    return out
