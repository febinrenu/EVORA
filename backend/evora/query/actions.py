"""Actions the system cannot recognise, so an answer about them must not claim them.

evora tracks people and vehicles, where they go (zones, lines, dwell), what they wear and carry. It has no action
recogniser: "did someone put something down" can only be answered with who was there, never with "yes". The
supported movements (walk, pass, cross, enter, exit, leave, wait, loiter, stand, stay, appear, park) and carrying are
not listed here on purpose, and neither are words that only look like actions in place names ("loading bay",
"drop-off zone") or everyday phrases ("close to the gate", "rush hour", "give me the cars").
"""
from __future__ import annotations

import re

_VEHICLE = r"(?:car|cars|vehicle|vehicles|truck|trucks|van|vans|bus|buses|taxi|suv|auto|rickshaw)"
_DET = r"(?:the|a|an|his|her|their|its|car|vehicle)"

# (pattern, readable phrase used in the answer: "I can't tell whether anyone was <phrase>")
_ACTIONS: list[tuple[re.Pattern[str], str]] = [(re.compile(p, re.IGNORECASE), label) for p, label in [
    (r"\b(?:put|puts|putting|set|sets|setting|lay|lays|laid|laying)\b(?:\s+\w+){0,3}\s+down\b",
     "putting something down"),
    (r"\b(?:pick|picks|picked|picking)\b(?:\s+\w+){0,3}\s+up\b", "picking something up"),
    (r"\b(?:drop|drops|dropped|dropping)\b(?!-?\s*off\s+(?:zone|area|point|bay|lane))", "dropping something"),
    (rf"\b(?:get|gets|got|getting|climb|climbs|climbed|climbing|step|steps|stepped|stepping|hop|hops|hopped)\s+"
     rf"(?:in|into|out\s+of|out|off)\b(?:\s+\w+){{0,2}}\s+{_VEHICLE}\b", "getting into or out of a vehicle"),
    (rf"\b(?:enter|enters|entered|entering|exit|exits|exited|exiting|leave|leaves|left|leaving)\s+"
     rf"(?:a|an|the|their|his|her)?\s*{_VEHICLE}\b", "getting into or out of a vehicle"),
    (rf"\b(?:open|opens|opened|opening|close|closes|closed|closing|shut|shuts|shutting)\s+{_DET}\s+(?:\w+\s+)?"
     r"(?:door|doors|trunk|boot|hood|bonnet|window|gate)\b", "opening or closing something"),
    (r"\b(?:turn|turns|turned|turning)\s+(?:left|right|around|round)\b|\bu-?turns?\b", "turning"),
    (r"\b(?:un)?(?:load|loads|loaded)\b|\b(?:un)?loading\b(?!\s+(?:bay|bays|dock|docks|area|zone|ramp|point))",
     "loading or unloading"),
    (r"\b(?:hand|hands|handed|handing)\s+(?:over|something|it|a|an|the)\b"
     r"|\b(?:give|gives|gave|giving)\b(?:\s+\w+){1,3}\s+to\s+(?:a|an|the|someone|somebody|another|him|her|them)\b",
     "handing something over"),
    (r"\b(?:talk|talks|talked|talking|chat|chats|chatted|chatting|speak|speaks|spoke|speaking)\b", "talking"),
    (r"\b(?:hug|hugs|hugged|hugging|embrac\w*)\b|\bshak\w*\s+hands\b", "greeting someone"),
    (r"\b(?:fight|fights|fought|fighting|punch\w*|hit|hits|hitting|kick|kicks|kicked|kicking|attack\w*)\b",
     "fighting"),
    (r"\b(?:push|pushes|pushed|pushing|pull|pulls|pulled|pulling)\b", "pushing or pulling something"),
    (r"\b(?:throw|throws|threw|throwing|toss|tosses|tossed|tossing)\b", "throwing something"),
    (r"\b(?:steal|steals|stole|stolen|stealing|rob|robs|robbed|robbing|shoplift\w*)\b", "stealing"),
    (r"\b(?:fall|falls|fell|fallen|falling|tripped|tripping|collaps\w*)\b", "falling"),
    (r"\b(?:run|runs|ran|running|jog|jogs|jogged|jogging|sprint\w*)\b", "running"),
    (r"\b(?:sit|sits|sat|sitting|lie|lies|lying)\b|\blay\s+down\b", "sitting or lying down"),
    (r"\b(?:smok\w+|vap(?:e|es|ed|ing))\b", "smoking"),
    (r"\b(?:texting|phoning|calling|on\s+(?:the|a|his|her|their)\s+phone|using\s+(?:a|the|his|her|their)\s+phone)\b",
     "using a phone"),
]]


def unsupported_action(text: str) -> str | None:
    """The action a question asks about that cannot be recognised, as a phrase, or None."""
    for pattern, label in _ACTIONS:
        if pattern.search(text or ""):
            return label
    return None


__all__ = ["unsupported_action"]
