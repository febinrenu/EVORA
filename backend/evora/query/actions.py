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

_VEHICLE = r"(?:car|cars|vehicle|vehicles|truck|trucks|van|vans|bus|buses|taxi|suv|auto|rickshaw)"
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
    (r"\b(?:texting|phoning|calling|on\s+(?:the|a|his|her|their)\s+phone|using\s+(?:a|the|his|her|their)\s+phone)\b",
     "phone", "using a phone"),
]]


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


__all__ = ["classify_action", "unsupported_action"]
