"""Resolve the time phrase in a plan against the footage clock.

Plans carry the phrase the user said ("in the last hour", "yesterday evening") and the
server turns it into epoch bounds. "Now" is the workspace's reference_now (the end of
the latest footage by default), never the wall clock, so recorded footage answers the
same way tomorrow as today.
"""
from __future__ import annotations

import re
from datetime import UTC, date, datetime, time, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from contracts.models import TimeWindow

UNIT_S = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800}
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"])}
# part of day -> (start hour, end hour); night runs past midnight
PARTS = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 21), "night": (21, 29)}

_MONTH_RX = "|".join(MONTHS)
_RELATIVE = re.compile(
    r"\b(?:in|within|during|over)\s+the\s+(?:last|past)\s+(?:(\d+|[a-z]+)\s+)?(second|minute|hour|day|week)s?\b"
)
_PERIOD = re.compile(r"\b(this|last)\s+(week|month)\b")
_DAY = re.compile(r"\b(today|tonight|yesterday|this|last)(?:\s+(morning|afternoon|evening|night))?\b")
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RX})(?:\s+(\d{{4}}))?\b")
_MDY = re.compile(rf"\b({_MONTH_RX})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_OFFSET = re.compile(r"^(?:utc|gmt)?\s*([+-])(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)


def _tod_only(window: TimeWindow) -> bool:
    """A phrase we cannot place on the calendar is still fine when it came with time-of-day bounds."""
    return bool(window.tod_after or window.tod_before)


def parse_tz(name: str | None) -> tzinfo:
    """IANA name, or a fixed offset such as '+05:30' / 'UTC+5:30'; UTC when unknown."""
    if not name:
        return UTC
    name = name.strip()
    if name.upper() in ("UTC", "GMT", "Z"):
        return UTC
    if m := _OFFSET.match(name):
        delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
        return timezone(delta if m.group(1) == "+" else -delta)
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return UTC


def _midnight(day: date, tz: tzinfo) -> datetime:
    return datetime.combine(day, time(), tzinfo=tz)


def _tod_seconds(hhmm: str) -> float:
    hour, minute = hhmm.split(":")
    return int(hour) * 3600 + int(minute) * 60


def _absolute_date(phrase: str, ref_day: date) -> date | None:
    if m := _ISO.search(phrase):
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    if m := _DMY.search(phrase):
        day, month, year = int(m.group(1)), MONTHS[m.group(2)], m.group(3)
    elif m := _MDY.search(phrase):
        month, day, year = MONTHS[m.group(1)], int(m.group(2)), m.group(3)
    else:
        return None
    try:
        if year:
            return date(int(year), month, day)
        guess = date(ref_day.year, month, day)
        return guess if guess <= ref_day else date(ref_day.year - 1, month, day)  # no future footage
    except ValueError:
        return None


def resolve_window(window: TimeWindow | None, reference_now: float, tz: tzinfo = UTC) -> tuple[TimeWindow | None, bool]:
    """Fill `start`/`end` from `window.phrase`. Returns (window, understood).

    `understood` is False when there was a phrase but it could not be interpreted; the
    window is then returned unchanged so the caller can add a note. Time-of-day bounds
    alone ("after 8pm") stay as filters and do not set start/end.
    """
    if window is None or not window.phrase:
        return window, True
    phrase = window.phrase.lower()
    ref_dt = datetime.fromtimestamp(reference_now, tz)
    today = ref_dt.date()

    start: float | None = None
    end: float | None = None
    day_based = False  # a single calendar day, so time-of-day bounds can be applied to it

    if m := _RELATIVE.search(phrase):
        raw = m.group(1)
        count = 1 if raw is None else int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw)
        if count is None:
            return window, _tod_only(window)
        end, start = reference_now, reference_now - count * UNIT_S[m.group(2)]
    elif d := _absolute_date(phrase, today):
        start = _midnight(d, tz).timestamp()
        end = _midnight(d + timedelta(days=1), tz).timestamp()
        day_based = True
    elif m := _PERIOD.search(phrase):
        this = m.group(1) == "this"
        if m.group(2) == "week":  # weeks run Monday to Sunday
            monday = today - timedelta(days=today.weekday())
            first = monday if this else monday - timedelta(days=7)
            after = None if this else monday
        else:
            first = today.replace(day=1) if this else (today.replace(day=1) - timedelta(days=1)).replace(day=1)
            after = None if this else today.replace(day=1)
        start = _midnight(first, tz).timestamp()
        end = reference_now if after is None else _midnight(after, tz).timestamp()
    elif m := _DAY.search(phrase):
        word, part = m.group(1), m.group(2)
        if word == "this" and part is None:  # "this week", "this year": not handled
            return window, _tod_only(window)
        if word == "last" and part is None:
            return window, _tod_only(window)
        if word == "tonight":
            word, part = "today", part or "night"
        if word == "last":  # "last night", "last evening"
            base = today - timedelta(days=1)
        elif word == "yesterday":
            base = today - timedelta(days=1)
        else:
            base = today
        d0 = _midnight(base, tz).timestamp()
        if part:
            lo, hi = PARTS[part]
            start, end = d0 + lo * 3600, d0 + hi * 3600
            if hi > 24:  # night ends the next morning; recompute for DST-safe midnight
                end = _midnight(base + timedelta(days=1), tz).timestamp() + (hi - 24) * 3600
        else:
            start, end = d0, _midnight(base + timedelta(days=1), tz).timestamp()
            day_based = True
        if base == today and end > reference_now > start:
            end = reference_now  # the footage ends at reference_now
    else:
        return window, _tod_only(window)

    if day_based:
        if window.tod_after:
            start = max(start, start_of_day(start, tz) + _tod_seconds(window.tod_after))
        if window.tod_before:
            end = min(end, start_of_day(start, tz) + _tod_seconds(window.tod_before))
    if start is not None and end is not None and end < start:
        return window, _tod_only(window)
    return window.model_copy(update={"start": start, "end": end}), True


def start_of_day(ts: float, tz: tzinfo) -> float:
    return _midnight(datetime.fromtimestamp(ts, tz).date(), tz).timestamp()
