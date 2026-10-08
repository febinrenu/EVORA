"""Actions the system cannot recognise, so an answer about them must not claim them.

evora tracks people and vehicles, where they go (zones, lines, dwell), what they wear and carry. It has no action
recogniser: "did someone put something down" can only be answered with who was there, never with "yes". For a few
actions, how people, bags and vehicles moved points at a likely moment (see query/action_cues.py); the key returned
by `classify_action` says which cue applies.

The supported movements (walk, pass, cross, enter, exit, leave, wait, loiter, stand, stay, appear, park) and carrying
are not listed here on purpose, and neither are words that only look like actions in place names ("loading bay",
"drop-off zone") or everyday phrases ("close to the gate", "rush hour", "give me the cars").
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_VEHICLE = r"(?:car|cars|vehicle|vehicles|truck|trucks|van|vans|bus|buses|taxi|suv|auto|rickshaw)"
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}
_DET = r"(?:the|a|an|his|her|their|its|car|vehicle)"

# (pattern, key, readable phrase used in the answer: "I can't tell whether anyone was <phrase>"); first match wins
_ACTIONS: list[tuple[re.Pattern[str], str, str]] = [(re.compile(p, re.IGNORECASE), key, label) for p, key, label in [
    (r"\b(?:put|puts|putting|set|sets|setting|lay|lays|laid|laying)\b(?:\s+\w+){0,3}\s+down\b",
     "put_down", "putting something down"),
    (r"\b(?:pick|picks|picked|picking)\b(?:\s+\w+){0,3}\s+up\b", "pick_up", "picking something up"),
    (r"\b(?:drop|drops|dropped|dropping)\b(?!-?\s*off\s+(?:zone|area|point|bay|lane))", "drop", "dropping something"),
    (rf"\b(?:get|gets|got|getting|climb|climbs|climbed|climbing|step|steps|stepped|stepping|hop|hops|hopped)\s+"
     rf"(?:out\s+of|out|off)\b(?:\s+\w+){{0,2}}\s+{_VEHICLE}\b", "vehicle_out", "getting out of a vehicle"),
    (rf"\b(?:get|gets|got|getting|climb|climbs|climbed|climbing|step|steps|stepped|stepping|hop|hops|hopped)\s+"
     rf"(?:in|into|onto|on)\b(?:\s+\w+){{0,2}}\s+{_VEHICLE}\b", "vehicle_in", "getting into a vehicle"),
    (rf"\b(?:exit|exits|exited|exiting|leave|leaves|left|leaving)\s+(?:a|an|the|their|his|her)?\s*{_VEHICLE}\b",
     "vehicle_out", "getting out of a vehicle"),
    (rf"\b(?:enter|enters|entered|entering|board|boards|boarded|boarding)\s+(?:a|an|the|their|his|her)?\s*{_VEHICLE}\b",
     "vehicle_in", "getting into a vehicle"),
    (rf"\b(?:open|opens|opened|opening|close|closes|closed|closing|shut|shuts|shutting)\s+{_DET}\s+(?:\w+\s+)?"
     r"(?:door|doors|trunk|boot|hood|bonnet|window|gate)\b", "open_close", "opening or closing something"),
    (r"\bu-?turns?\b|\b(?:turn|turns|turned|turning)\s+(?:around|round)\b", "u_turn", "turning around"),
    (r"\b(?:turn|turns|turned|turning)\s+left\b", "turn_left", "turning left"),
    (r"\b(?:turn|turns|turned|turning)\s+right\b", "turn_right", "turning right"),
    (r"\b(?:un)?(?:load|loads|loaded)\b|\b(?:un)?loading\b(?!\s+(?:bay|bays|dock|docks|area|zone|ramp|point))",
     "load", "loading or unloading"),
    (r"\b(?:hand|hands|handed|handing)\s+(?:over|something|it|a|an|the)\b"
     r"|\b(?:give|gives|gave|giving)\b(?:\s+\w+){1,3}\s+to\s+(?:a|an|the|someone|somebody|another|him|her|them)\b",
     "hand_over", "handing something over"),
    (r"\b(?:texting|phoning|calling|on\s+(?:the|a|his|her|their)\s+phone|using\s+(?:a|the|his|her|their)\s+phone)\b",
     "phone", "using a phone"),
    (r"\b(?:talk|talks|talked|talking|chat|chats|chatted|chatting|speak|speaks|spoke|speaking)\b", "talk", "talking"),
    (r"\b(?:hug|hugs|hugged|hugging|embrac\w*)\b|\bshak\w*\s+hands\b", "greet", "greeting someone"),
    (r"\b(?:fight|fights|fought|fighting|punch\w*|hit|hits|hitting|kick|kicks|kicked|kicking|attack\w*)\b",
     "fight", "fighting"),
    (r"\b(?:push|pushes|pushed|pushing|pull|pulls|pulled|pulling)\b", "push_pull", "pushing or pulling something"),
    (r"\b(?:throw|throws|threw|throwing|toss|tosses|tossed|tossing)\b", "throw", "throwing something"),
    (r"\b(?:steal|steals|stole|stolen|stealing|rob|robs|robbed|robbing|shoplift\w*)\b", "steal", "stealing"),
    (r"\b(?:fall|falls|fell|fallen|falling|tripped|tripping|collaps\w*)\b", "fall", "falling"),
    (r"\b(?:run|runs|ran|running|jog|jogs|jogged|jogging|sprint\w*)\b", "run", "running"),
    (r"\b(?:sit|sits|sat|sitting|lie|lies|lying)\b|\blay\s+down\b", "sit", "sitting or lying down"),
    (r"\b(?:smok\w+|vap(?:e|es|ed|ing))\b", "smoke", "smoking"),
]]


@dataclass(frozen=True)
class DetectedAction:
    """An action perception writes to the `events` table, so an answer about it can come from a matching event."""

    key: str
    kinds: tuple[str, ...]     # event kinds that satisfy it (see ACTION_EVENT_KINDS in logic.py)
    subject: str               # "vehicle" or "person": the kind of track the event belongs to
    label: str                 # what the answer calls it: "a vehicle turning left"
    cue: str                   # how it is found, for the note: "from how the vehicle moved"


# (pattern, action); first match wins. Only what perception/actions.py detects: nothing here is a guess.
_DETECTED: list[tuple[re.Pattern[str], DetectedAction]] = [(re.compile(p, re.IGNORECASE), d) for p, d in [
    (rf"\b(?:get|gets|got|getting|climb|climbs|climbed|climbing|step|steps|stepped|stepping|hop|hops|hopped)\s+"
     rf"(?:out\s+of|out|off)\b(?:\s+\w+){{0,2}}\s+{_VEHICLE}\b"
     rf"|\b(?:exit|exits|exited|exiting|leave|leaves|left|leaving)\s+(?:a|an|the|their|his|her)?\s*{_VEHICLE}\b",
     DetectedAction("vehicle_out", ("person_exits_vehicle",), "person", "a person getting out of a vehicle",
                    "from a person appearing next to a vehicle that has stopped")),
    (rf"\b(?:get|gets|got|getting|climb|climbs|climbed|climbing|step|steps|stepped|stepping|hop|hops|hopped)\s+"
     rf"(?:in|into|onto|on)\b(?:\s+\w+){{0,2}}\s+{_VEHICLE}\b"
     rf"|\b(?:enter|enters|entered|entering|board|boards|boarded|boarding)\s+(?:a|an|the|their|his|her)?\s*{_VEHICLE}\b",
     DetectedAction("vehicle_in", ("person_enters_vehicle",), "person", "a person getting into a vehicle",
                    "from a person disappearing next to a vehicle that has stopped")),
    (r"\bu-?turns?\b|\b(?:turn|turns|turned|turning)\s+(?:around|round)\b",
     DetectedAction("u_turn", ("vehicle_u_turn",), "vehicle", "a vehicle making a U-turn", "from the vehicle's path")),
    (r"\b(?:turn|turns|turned|turning)\s+left\b",
     DetectedAction("turn_left", ("vehicle_turn_left",), "vehicle", "a vehicle turning left", "from the vehicle's path")),
    (r"\b(?:turn|turns|turned|turning)\s+right\b",
     DetectedAction("turn_right", ("vehicle_turn_right",), "vehicle", "a vehicle turning right", "from the vehicle's path")),
    (r"\b(?:reverse|reverses|reversed|reversing|back\s+up|backs\s+up|backed\s+up|backing\s+up)\b",
     DetectedAction("reverse", ("vehicle_reverse",), "vehicle", "a vehicle reversing", "from the vehicle's path")),
    (r"\b(?:start|starts|started|starting)\s+(?:to\s+)?(?:moving|move|driving|drive|off)\b"
     r"|\b(?:drive|drives|drove|driving|pull|pulls|pulled|pulling)\s+(?:off|away)\b|\bmove\s+off\b",
     DetectedAction("start", ("vehicle_start",), "vehicle", "a vehicle starting to move", "from the vehicle's speed")),
    (r"\b(?:stop|stops|stopped|stopping|halt|halts|halted|halting)\b",
     DetectedAction("stop", ("vehicle_stop",), "vehicle", "a vehicle stopping", "from the vehicle's speed")),
    (r"\b(?:talk|talks|talked|talking|chat|chats|chatted|chatting|conversation)\b",
     DetectedAction("talk", ("people_close",), "person", "two people standing together",
                    "from two people standing close together and still")),
]]
_PHONE = re.compile(r"\bphone|\bcall(?:ing)?\b|\btext(?:ing)?\b", re.IGNORECASE)
_VEHICLE_WORDS = re.compile(rf"\b{_VEHICLE}\b", re.IGNORECASE)
_PEOPLE_WORDS = re.compile(r"\b(?:person|people|man|men|woman|women|someone|somebody|anyone|anybody|pedestrians?|two)\b",
                           re.IGNORECASE)


def detected_action(text: str, target_classes: Sequence[str] = ()) -> DetectedAction | None:
    """The action perception detects that this question asks about, or None.

    A vehicle action needs a vehicle: "did a person turn left" is not "did a vehicle turn left" and stays unrecognised.
    "Talking on the phone" is not two people standing together.
    """
    for pattern, action in _DETECTED:
        if not pattern.search(text or ""):
            continue
        if action.key == "talk" and _PHONE.search(text):
            return None
        if action.subject == "vehicle":
            classes = set(target_classes)
            if classes and not classes & VEHICLE_CLASSES:
                return None                       # the question names something else (a person) as the one acting
            if not classes and not _VEHICLE_WORDS.search(text) and _PEOPLE_WORDS.search(text):
                return None
        return action
    return None


def classify_action(text: str) -> tuple[str, str] | None:
    """(key, phrase) of the action a question asks about that cannot be recognised, or None."""
    for pattern, key, label in _ACTIONS:
        if pattern.search(text or ""):
            return key, label
    return None


def unsupported_action(text: str) -> str | None:
    """The action a question asks about that cannot be recognised, as a phrase, or None."""
    found = classify_action(text)
    return found[1] if found else None


__all__ = ["DetectedAction", "classify_action", "detected_action", "unsupported_action"]
