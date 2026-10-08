"""Time-of-day phrases for time aliases: '8pm to 6am', '20:00-06:00', 'noon until 2:30pm'."""
from __future__ import annotations

import re

_CLOCK = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")
_SPLIT = re.compile(r"\s*(?:\bto\b|\buntil\b|\btill\b|\band\b|-|–|—)\s*")
_WORDS = {"noon": "12:00", "midday": "12:00", "midnight": "00:00"}


class TodError(ValueError):
    pass


def parse_tod(text: str) -> str:
    """One clock time to 'HH:MM'. Raises TodError when it is not a time."""
    t = text.strip().lower().replace(".", "")
    if t in _WORDS:
        return _WORDS[t]
    m = _CLOCK.match(t)
    if not m:
        raise TodError(f"not a time: {text!r}")
    hour, minute, meridiem = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if minute > 59:
        raise TodError(f"not a time: {text!r}")
    if meridiem:
        if not 1 <= hour <= 12:
            raise TodError(f"not a time: {text!r}")
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    elif hour > 23:
        raise TodError(f"not a time: {text!r}")
    return f"{hour:02d}:{minute:02d}"


def parse_tod_range(text: str) -> tuple[str, str]:
    """'8pm to 6am' -> ('20:00', '06:00'). The range may wrap midnight."""
    cleaned = re.sub(r"^\s*(?:from|between)\s+", "", text.strip(), flags=re.IGNORECASE)
    parts = [p for p in _SPLIT.split(cleaned) if p]
    if len(parts) != 2:
        raise TodError("give the hours as a start and an end, for example '8pm to 6am'")
    return parse_tod(parts[0]), parse_tod(parts[1])
