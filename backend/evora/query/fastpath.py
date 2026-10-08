"""Deterministic parser for the common question shapes. No network, no model.

It is deliberately conservative: any word it does not understand makes it return
None so the planner falls through to the language model. A wrong fast-path plan is
worse than a slow correct one.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from contracts.models import QueryPlan, Referent, Target, TimeWindow


class CameraLike(Protocol):
    id: str
    name: str


# ----------------------------------------------------------------- lexicons
COLOURS = {
    "black": "black", "white": "white", "grey": "grey", "gray": "grey", "silver": "grey", "red": "red",
    "orange": "orange", "yellow": "yellow", "green": "green", "blue": "blue", "purple": "purple",
    "pink": "pink", "brown": "brown",
}

PERSON_GENERIC = {"person", "people", "someone", "anyone", "anybody", "everyone", "everybody", "somebody",
                  "pedestrian", "pedestrians"}
PERSON_SPECIFIC = {"man", "men", "woman", "women", "guy", "child", "children", "kid", "kids", "boy", "girl"}

# word -> (noun to report, detector classes, extra vehicle-type attribute)
VEHICLES: dict[str, tuple[str, list[str], str | None]] = {
    "car": ("car", ["car"], None), "cars": ("car", ["car"], None),
    "sedan": ("car", ["car"], None), "hatchback": ("car", ["car"], None),
    "taxi": ("car", ["car"], None), "cab": ("car", ["car"], None),
    "suv": ("suv", ["car"], "suv"), "suvs": ("suv", ["car"], "suv"),
    "van": ("van", ["car", "truck"], "van"), "vans": ("van", ["car", "truck"], "van"),
    "truck": ("truck", ["truck"], None), "trucks": ("truck", ["truck"], None),
    "lorry": ("truck", ["truck"], None),
    "bus": ("bus", ["bus"], None), "buses": ("bus", ["bus"], None),
    "motorcycle": ("motorcycle", ["motorcycle"], None), "motorcycles": ("motorcycle", ["motorcycle"], None),
    "motorbike": ("motorcycle", ["motorcycle"], None), "scooter": ("motorcycle", ["motorcycle"], None),
    "bicycle": ("bicycle", ["bicycle"], None), "bicycles": ("bicycle", ["bicycle"], None),
    "bike": ("bicycle", ["bicycle"], None), "bikes": ("bicycle", ["bicycle"], None),
    "rickshaw": ("rickshaw", ["car", "motorcycle"], "auto_rickshaw"),
    "vehicle": ("vehicle", ["car", "motorcycle", "bus", "truck"], None),
    "vehicles": ("vehicle", ["car", "motorcycle", "bus", "truck"], None),
}

CARRY = {"backpack": "backpack", "backpacks": "backpack", "rucksack": "backpack", "handbag": "handbag",
         "handbags": "handbag", "purse": "handbag", "suitcase": "suitcase", "suitcases": "suitcase",
         "luggage": "suitcase", "umbrella": "umbrella", "umbrellas": "umbrella"}
CARRY_VERBS = {"carrying", "holding", "with"}
SIZE_WORDS = {"large", "big", "huge"}

ARTICLES = {"a", "an", "the", "any", "some"}
SKIP = {"there", "who", "that", "which", "been", "is", "was", "were", "are", "did", "do", "does", "has", "have",
        "had", "seen"}
MOTION = {"go", "went", "gone", "going", "goes", "walk", "walked", "walking", "drive", "drove", "driven",
          "driving", "come", "came", "run", "ran", "move", "moved"}
ACTIONS = {
    "pass": "pass_through", "passed": "pass_through", "passing": "pass_through", "cross": "pass_through",
    "crossed": "pass_through", "crossing": "pass_through",
    "enter": "enter", "entered": "enter", "entering": "enter", "enters": "enter",
    "exit": "exit", "exited": "exit", "leave": "exit", "left": "exit", "leaving": "exit", "leaves": "exit",
    "loiter": "dwell", "loitered": "dwell", "loitering": "dwell", "wait": "dwell", "waited": "dwell",
    "waiting": "dwell", "stand": "dwell", "stood": "dwell", "standing": "dwell", "stay": "dwell",
    "stayed": "dwell", "staying": "dwell",
    "arrive": "appear", "arrived": "appear", "arrives": "appear", "appear": "appear", "appeared": "appear",
}
# preposition -> action implied when no verb gave one
PREPOSITIONS: dict[str, str] = {
    "at": "any", "near": "any", "by": "any", "in": "any", "inside": "any", "outside": "any", "around": "any",
    "on": "any", "through": "pass_through", "past": "pass_through", "across": "pass_through", "into": "enter",
}
DIRECT_OBJECT_ACTIONS = {"pass_through", "enter", "exit"}
PLACE_STOP ={"and", "or", "who", "that", "which", "with", "carrying", "holding", "but", "not", "my", "our"}
MAX_PLACE_WORDS = 4
# words that mean the "place" actually swallowed a time expression we could not parse
TIME_WORDS = {"after", "before", "between", "during", "ago", "since", "until", "till", "week", "weeks", "month",
              "months", "year", "years", "hour", "hours", "minute", "minutes", "last", "next", "yesterday",
              "tonight", "today", "now", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
              "sunday", "january", "february", "march", "april", "may", "june", "july", "august", "september",
              "october", "november", "december", "morning", "afternoon", "evening", "night", "noon", "midnight"}

NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}

# ------------------------------------------------------------- time phrases
_T = r"\d{1,2}(?::\d{2})?\s?(?:am|pm)?"
_RELATIVE = re.compile(
    r"\b(?:in|within|during|over)\s+the\s+(?:last|past)\s+(?:(?:\d+|a|an|one|two|three|four|five|six|seven|eight"
    r"|nine|ten|fifteen|twenty|thirty)\s+)?(?:second|minute|hour|day|week)s?\b"
)
_BETWEEN = re.compile(rf"\bbetween\s+({_T})\s+and\s+({_T})\b")
_AFTER_BEFORE = re.compile(r"\b(after|before)\s+(\d{1,2}(?::\d{2})?\s?(?:am|pm)|\d{1,2}:\d{2})\b")
_DAYPART = re.compile(
    r"\b(?:(?:yesterday|this|last)\s+(?:morning|afternoon|evening|night)|tonight|today|yesterday)\b"
)
_CLOCK = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s?(am|pm)?$")


def _to_minutes(token: str, default_suffix: str | None = None) -> tuple[int, bool] | None:
    """Return (minutes since midnight, had_explicit_suffix) or None."""
    m = _CLOCK.match(token.strip())
    if not m:
        return None
    hour, minute, suffix = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    explicit = suffix is not None
    suffix = suffix or default_suffix
    if minute > 59 or hour > 23 or (suffix and not 1 <= hour <= 12):
        return None
    if suffix == "pm" and hour != 12:
        hour += 12
    if suffix == "am" and hour == 12:
        hour = 0
    return hour * 60 + minute, explicit


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _extract_time(text: str) -> tuple[str, TimeWindow | None] | None:
    """Pull time phrases out of `text`. Returns (text without them, window) or None if a
    time-looking phrase could not be understood."""
    found: list[tuple[int, str]] = []
    tod_after = tod_before = None

    for m in _BETWEEN.finditer(text):
        second = _to_minutes(m.group(2))
        if second is None:
            return None
        first = _to_minutes(m.group(1), default_suffix=None if second[1] is False else m.group(2).strip()[-2:])
        if first is None:
            return None
        a, b = first[0], second[0]
        if not first[1] and a > b:  # "11 and 1 pm": the first is the morning one
            a -= 12 * 60
        if a >= b or a < 0:
            return None
        tod_after, tod_before = _hhmm(a), _hhmm(b)
        found.append((m.start(), m.group(0)))
    text = _BETWEEN.sub(" ", text)

    for m in _AFTER_BEFORE.finditer(text):
        parsed = _to_minutes(m.group(2))
        if parsed is None:
            return None
        if m.group(1) == "after":
            tod_after = _hhmm(parsed[0])
        else:
            tod_before = _hhmm(parsed[0])
        found.append((m.start(), m.group(0)))
    text = _AFTER_BEFORE.sub(" ", text)

    for rx in (_RELATIVE, _DAYPART):
        for m in rx.finditer(text):
            found.append((m.start(), m.group(0)))
        text = rx.sub(" ", text)

    if not found:
        return text, None
    text = re.sub(r"\b(?:and|or)\s*$", "", text.strip())  # "after 8pm and before 10pm" leaves a dangling "and"
    # keep the phrase in the order it appeared in the original sentence
    phrase = " ".join(p for _, p in sorted(found))
    return text, TimeWindow(phrase=phrase, tod_after=tod_after, tod_before=tod_before)


# -------------------------------------------------------------- normalizing
def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(\d{1,2})\s?([ap])\.m\.?", r"\1\2m", text)
    text = text.replace("’", "'").replace("auto-rickshaw", "rickshaw").replace("auto rickshaw", "rickshaw")
    text = re.sub(r"[?!.,;\"'()]", " ", text)
    text = re.sub(r"\b(please|hey|hi|ok|okay)\b", " ", text)
    text = re.sub(r"^\s*(?:can|could|would) you\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm_name(name: str) -> str:
    name = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    name = re.sub(r"\b(the|camera|cam)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _match_camera(place: str, cameras: Sequence[CameraLike]) -> str | None:
    wanted = _norm_name(place)
    if not wanted:
        return None
    for cam in cameras:
        if wanted == _norm_name(cam.name) or wanted == _norm_name(cam.id).replace("_", " "):
            return cam.id
    digits = re.fullmatch(r"(\d+)", wanted)
    if digits:
        for cam in cameras:
            tail = re.search(r"(\d+)$", cam.id)
            if tail and int(tail.group(1)) == int(digits.group(1)):
                return cam.id
    return None


# ------------------------------------------------------------------- intent
_EXISTS = re.compile(r"^(?:did|was|were|is|are|has|have|does|do)\s+(.+)$")
_COUNT = re.compile(r"^how many\s+(.+)$")
_LIST = re.compile(r"^(?:show|find|list|get|give)(?:\s+me)?\s+(?:all\s+|every\s+|any\s+)?(.+)$")
_WHEN = re.compile(r"^when\s+(?:did|was|were)\s+(.+)$")


def _split_intent(text: str) -> tuple[str, str, int] | None:
    """Return (intent, remainder, default limit)."""
    if m := re.match(r"^tell me (?:if|whether)\s+(.+)$", text):
        return "exists", m.group(1), 10
    if m := re.match(r"^tell me\s+(.+)$", text):
        return _split_intent(m.group(1))
    if m := _COUNT.match(text):
        return "count", m.group(1), 10
    if m := _WHEN.match(text):
        rest = m.group(1)
        tokens = rest.split()
        if "first" in tokens:
            return "first", " ".join(t for t in tokens if t != "first"), 1
        if "last" in tokens:
            return "last", " ".join(t for t in tokens if t != "last"), 1
        return None
    if m := _EXISTS.match(text):
        return "exists", m.group(1), 10
    if m := _LIST.match(text):
        return "list", m.group(1), 10
    return None


# ---------------------------------------------------------------- the parser
def _embed_text(noun: str, colours: list[str], carrying: list[str], is_person: bool) -> str:
    desc = noun if not is_person or noun in PERSON_SPECIFIC else "person"
    if is_person:
        text = f"a photo of a {desc}"
        if colours:
            text += f" in {' and '.join(colours)}"
        if carrying:
            text += " carrying " + " and ".join(
                "a large bag" if c == "large_bag" else f"a {c}" for c in carrying
            )
        return text
    shown = "auto rickshaw" if desc == "rickshaw" else desc
    return "a photo of a " + " ".join([*colours, shown])


def parse(text: str, cameras: Sequence[CameraLike] = ()) -> QueryPlan | None:
    """Return a plan for a recognised question shape, else None."""
    norm = _normalize(text)
    if not norm:
        return None
    timed = _extract_time(norm)
    if timed is None:
        return None
    norm, window = timed
    norm = re.sub(r"\s+", " ", norm).strip()
    split = _split_intent(norm)
    if split is None:
        return None
    intent, rest, limit = split

    tokens = rest.split()
    colours: list[str] = []
    carrying: list[str] = []
    extra_types: list[str] = []
    noun: str | None = None
    classes: list[str] = []
    is_person = False
    action = "any"
    saw_verb = False
    place_tokens: list[str] = []
    prep = ""

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ARTICLES or tok in SKIP or tok in MOTION:
            saw_verb = saw_verb or tok in MOTION
            i += 1
        elif tok in COLOURS and noun is None:
            colours.append(COLOURS[tok])
            i += 1
        elif noun is None and (tok in PERSON_GENERIC or tok in PERSON_SPECIFIC):
            noun, classes, is_person = tok, ["person"], True
            i += 1
        elif noun is None and tok in VEHICLES:
            noun, classes, vtype = VEHICLES[tok]
            if vtype:
                extra_types.append(vtype)
            i += 1
        elif noun is not None and tok in CARRY_VERBS and is_person:
            j = i + 1
            while j < len(tokens) and (tokens[j] in ARTICLES or tokens[j] in SIZE_WORDS):
                j += 1
            large = any(t in SIZE_WORDS for t in tokens[i + 1:j])
            if j < len(tokens) and tokens[j] == "bag" and large:
                carrying.append("large_bag")
            elif j < len(tokens) and tokens[j] in CARRY:
                carrying.append(CARRY[tokens[j]])
            else:
                return None
            i = j + 1
        elif tok in ACTIONS and noun is not None:
            if action != "any":
                return None
            action = ACTIONS[tok]
            saw_verb = True
            i += 1
            if action in DIRECT_OBJECT_ACTIONS and i < len(tokens) and tokens[i] not in PREPOSITIONS:
                prep = "direct"  # "enter the lobby", "leave the building": the object is the place
                place_tokens = [t for t in tokens[i:] if t not in ARTICLES]
                break
        elif tok in PREPOSITIONS and noun is not None:
            prep = tok
            place_tokens = [t for t in tokens[i + 1:] if t not in ARTICLES]
            break
        else:
            return None

    if noun is None:
        return None
    if norm.split()[0] in ("did", "do", "does") and not saw_verb:
        return None  # "did a car" is not a question
    if prep and action == "any":
        action = PREPOSITIONS[prep]
    if prep and (not place_tokens or len(place_tokens) > MAX_PLACE_WORDS or PLACE_STOP & set(place_tokens)
                 or any(t in ACTIONS or t in COLOURS or t in VEHICLES for t in place_tokens)):
        return None

    place_text = " ".join(place_tokens)
    camera_ids: list[str] = []
    place: Referent | None = None
    unresolved: list[Referent] = []
    if place_text:
        cam = _match_camera(place_text, cameras)
        if cam is not None:
            camera_ids = [cam]
        elif TIME_WORDS & set(place_tokens) or any(ch.isdigit() for ch in place_text):
            return None  # an unparsed time expression ended up in the place
        else:
            place = Referent(text=place_text, role="place")
            unresolved = [place]

    attributes = [*colours, *extra_types, *carrying]
    head = noun if noun not in PERSON_GENERIC else "person"
    target = Target(
        noun=head,
        cls=classes,
        attributes=attributes,
        embed_text=_embed_text(head, colours, carrying, is_person),
    )
    return QueryPlan(
        intent=intent,  # type: ignore[arg-type]
        targets=[target],
        place=place,
        action=action,  # type: ignore[arg-type]
        time=window,
        camera_ids=camera_ids,
        limit=limit,
        unresolved=unresolved,
        source="fastpath",
    )
