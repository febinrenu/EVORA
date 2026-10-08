from datetime import datetime

import pytest
from contracts.models import TimeWindow

from evora.query.timeparse import parse_tz, resolve_window

IST = parse_tz("+05:30")
# the footage ends Friday 9 Oct 2026, 10:30 IST
REF = datetime(2026, 10, 9, 10, 30, tzinfo=IST).timestamp()


def at(y, mo, d, h=0, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=IST).timestamp()


def resolve(phrase, **kw):
    out, ok = resolve_window(TimeWindow(phrase=phrase, **kw), REF, IST)
    return out, ok


@pytest.mark.parametrize(
    "phrase,start,end",
    [
        ("in the last hour", REF - 3600, REF),
        ("in the past 2 hours", REF - 7200, REF),
        ("within the last ten minutes", REF - 600, REF),
        ("in the last day", REF - 86400, REF),
        ("in the last week", REF - 604800, REF),
        ("in the last a minute", REF - 60, REF),
        ("today", at(2026, 10, 9), REF),
        ("yesterday", at(2026, 10, 8), at(2026, 10, 9)),
        ("this morning", at(2026, 10, 9, 5), REF),  # morning would end at 12:00 but the footage ends at 10:30
        ("yesterday morning", at(2026, 10, 8, 5), at(2026, 10, 8, 12)),
        ("yesterday afternoon", at(2026, 10, 8, 12), at(2026, 10, 8, 17)),
        ("yesterday evening", at(2026, 10, 8, 17), at(2026, 10, 8, 21)),
        ("last night", at(2026, 10, 8, 21), at(2026, 10, 9, 5)),
        ("tonight", at(2026, 10, 9, 21), at(2026, 10, 10, 5)),
        ("on 3 October", at(2026, 10, 3), at(2026, 10, 4)),
        ("on october 3rd", at(2026, 10, 3), at(2026, 10, 4)),
        ("3 October 2025", at(2025, 10, 3), at(2025, 10, 4)),
        ("on 2026-10-05", at(2026, 10, 5), at(2026, 10, 6)),
        ("on 20 december", at(2025, 12, 20), at(2025, 12, 21)),  # a future date means last year's
    ],
)
def test_phrases(phrase, start, end):
    out, ok = resolve(phrase)
    assert ok
    assert (out.start, out.end) == (pytest.approx(start), pytest.approx(end))
    assert out.phrase == phrase


def test_time_of_day_bounds_narrow_a_single_day():
    out, _ = resolve("yesterday after 8pm", tod_after="20:00")
    assert (out.start, out.end) == (at(2026, 10, 8, 20), at(2026, 10, 9))
    out, _ = resolve("on 3 October between 14:00 and 14:30", tod_after="14:00", tod_before="14:30")
    assert (out.start, out.end) == (at(2026, 10, 3, 14), at(2026, 10, 3, 14, 30))
    out, _ = resolve("today before 9am", tod_before="09:00")
    assert (out.start, out.end) == (at(2026, 10, 9), at(2026, 10, 9, 9))
    assert out.tod_before == "09:00"  # the filter fields are kept


def test_time_of_day_alone_leaves_bounds_open():
    out, ok = resolve_window(TimeWindow(phrase="after 8pm", tod_after="20:00"), REF, IST)
    assert ok is True  # a time-of-day filter is a complete answer: nothing to anchor, nothing to ask
    assert out.start is None and out.end is None and out.tod_after == "20:00"


def test_unparseable_phrases_are_reported_not_guessed():
    for phrase in ("after hours", "night shift", "sometime", "in the last few hours", "lunch time"):
        out, ok = resolve(phrase)
        assert ok is False and out.start is None and out.end is None, phrase


def test_no_phrase_or_no_window_is_a_noop():
    assert resolve_window(None, REF, IST) == (None, True)
    w = TimeWindow(tod_after="20:00")
    assert resolve_window(w, REF, IST) == (w, True)


def test_does_not_depend_on_the_wall_clock():
    a = resolve_window(TimeWindow(phrase="in the last hour"), REF, IST)[0]
    b = resolve_window(TimeWindow(phrase="in the last hour"), REF, IST)[0]
    assert a == b and a.end == REF


def test_timezone_changes_the_calendar_day():
    utc_ref = datetime(2026, 10, 9, 20, 0, tzinfo=parse_tz("UTC")).timestamp()  # 01:30 on the 10th in IST
    in_utc, _ = resolve_window(TimeWindow(phrase="today"), utc_ref, parse_tz("UTC"))
    in_ist, _ = resolve_window(TimeWindow(phrase="today"), utc_ref, IST)
    assert in_utc.start == at_utc(2026, 10, 9)
    assert in_ist.start == at(2026, 10, 10)


def at_utc(y, mo, d):
    return datetime(y, mo, d, tzinfo=parse_tz("UTC")).timestamp()


def test_parse_tz_forms():
    assert parse_tz(None).utcoffset(None).total_seconds() == 0
    assert parse_tz("UTC+5:30").utcoffset(None).total_seconds() == 19800
    assert parse_tz("-08:00").utcoffset(None).total_seconds() == -28800
    assert parse_tz("Not/AZone").utcoffset(None).total_seconds() == 0


def test_weeks_and_months():
    # REF is Friday 9 Oct 2026 10:30 IST: this week began Monday 5 Oct
    out, ok = resolve("this week")
    assert ok and (out.start, out.end) == (pytest.approx(at(2026, 10, 5)), pytest.approx(REF))
    out, ok = resolve("last week")
    assert ok and (out.start, out.end) == (pytest.approx(at(2026, 9, 28)), pytest.approx(at(2026, 10, 5)))
    out, ok = resolve("this month")
    assert ok and (out.start, out.end) == (pytest.approx(at(2026, 10, 1)), pytest.approx(REF))
    out, ok = resolve("last month")
    assert ok and (out.start, out.end) == (pytest.approx(at(2026, 9, 1)), pytest.approx(at(2026, 10, 1)))


def test_last_month_across_a_year_boundary():
    jan = datetime(2027, 1, 15, 12, 0, tzinfo=IST).timestamp()
    out, ok = resolve_window(TimeWindow(phrase="last month"), jan, IST)
    assert ok and out.start == pytest.approx(datetime(2026, 12, 1, tzinfo=IST).timestamp())
    assert out.end == pytest.approx(datetime(2027, 1, 1, tzinfo=IST).timestamp())


def test_a_phrase_with_time_of_day_bounds_is_never_reported_as_unknown():
    out, ok = resolve_window(TimeWindow(phrase="after hours", tod_after="20:00", tod_before="06:00"), REF, IST)
    assert ok is True and out.start is None
