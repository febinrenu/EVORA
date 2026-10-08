"""Clock alignment: find the wall-clock epoch (UTC seconds) of a video's first frame.

Order, first hit wins (PLAN.md section 8.2, P2.5):
  1. filename pattern (MEVA and common NVR / phone names)
  2. container `creation_time` metadata
  3. on-screen timestamp read by a local vision model (`osd.read_burned_in_time`, or an injected `osd_reader`)
  4. a filmed phone clock at the start of the recording, read the same way (source "slate")
  5. manual: the file's modification time, flagged so the UI can warn

Times written without a zone in a filename are read in `default_tz` (config `clock.default_tz`,
default Asia/Kolkata). `detect_clock` never raises.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path

import av

from evora.perception.vision import VisionClient, get_vision_client

log = logging.getLogger("evora.perception.clock")

DEFAULT_TZ = "Asia/Kolkata"
VISION_BUDGET_S = 40.0
OsdReader = Callable[[Path], "float | None"]  # returns epoch seconds, or None when no timestamp is readable

_MIN_YEAR, _MAX_YEAR = 2000, 2100

# (name, regex with named groups y, mo, d, h, mi, s). MEVA first because its names also contain date-like runs.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("meva", re.compile(r"(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})\.(?P<h>\d{2})-(?P<mi>\d{2})-(?P<s>\d{2})\.")),
    ("dashed", re.compile(
        r"(?P<y>\d{4})[-_.](?P<mo>\d{2})[-_.](?P<d>\d{2})[ T_.-]+(?P<h>\d{2})[-_.:](?P<mi>\d{2})[-_.:](?P<s>\d{2})"
    )),
    ("compact", re.compile(r"(?<!\d)(?P<y>\d{4})(?P<mo>\d{2})(?P<d>\d{2})[ T_.-]?(?P<h>\d{2})(?P<mi>\d{2})(?P<s>\d{2})(?!\d)")),
]


def _tz(name: str) -> tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - tzdata missing on some Windows installs: fall back to the fixed IST offset
        log.warning("timezone %s unavailable; using UTC+05:30", name)
        return timezone(timedelta(hours=5, minutes=30))


def default_tz_offset(when: float | None = None, tz_name: str = DEFAULT_TZ) -> str:
    """The clock zone as a fixed offset such as '+05:30', which every reader of `meta.tz` can parse without tzdata."""
    moment = datetime.fromtimestamp(when if when is not None else 0.0, tz=_tz(tz_name))
    offset = moment.utcoffset() or timedelta(0)
    minutes = int(offset.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    return f"{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def parse_filename(name: str, tz_name: str = DEFAULT_TZ) -> float | None:
    """Epoch seconds from a timestamp embedded in a file name, or None."""
    for _, pat in _PATTERNS:
        m = pat.search(name)
        if not m:
            continue
        try:
            parts = {k: int(v) for k, v in m.groupdict().items()}
            if not _MIN_YEAR <= parts["y"] <= _MAX_YEAR:
                continue
            local = datetime(parts["y"], parts["mo"], parts["d"], parts["h"], parts["mi"], parts["s"], tzinfo=_tz(tz_name))
        except ValueError:
            continue
        return local.timestamp()
    return None


def read_creation_time(path: Path) -> float | None:
    """Container `creation_time` as epoch seconds, ignoring unset or implausible values."""
    try:
        with av.open(str(path)) as container:
            raw = container.metadata.get("creation_time")
            if not raw and container.streams.video:
                raw = container.streams.video[0].metadata.get("creation_time")
    except (av.error.FFmpegError, OSError) as exc:
        log.warning("cannot read metadata of %s: %s", path, exc)
        return None
    if not raw:
        return None
    try:
        when = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        log.warning("unparseable creation_time %r in %s", raw, path)
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    if not _MIN_YEAR <= when.year <= _MAX_YEAR or when.year < 2005:
        return None
    return when.timestamp()


def detect_clock(path: str | Path, *, osd_reader: OsdReader | None = None, tz_name: str = DEFAULT_TZ,
                 vision: VisionClient | None = None) -> tuple[float, str]:
    """Return `(t0, source)` for the first frame of `path`. Source is one of filename|metadata|osd|slate|manual.

    The on-screen and slate readings use `vision`, else the client registered with `perception.vision`, and are
    skipped when there is none. A reading is trusted only when two frames agree with the video's own timeline.
    """
    p = Path(path)
    if (t0 := parse_filename(p.name, tz_name)) is not None:
        return t0, "filename"
    if (t0 := read_creation_time(p)) is not None:
        return t0, "metadata"
    if osd_reader is not None:
        try:
            t0 = osd_reader(p)
        except Exception:  # noqa: BLE001 - a failing vision model must never block ingest
            log.exception("on-screen timestamp reader failed for %s", p)
            t0 = None
        if t0 is not None:
            return t0, "osd"
    client = vision or get_vision_client()
    if client is not None and osd_reader is None:
        from evora.perception.osd import read_burned_in_time

        deadline = time.monotonic() + VISION_BUDGET_S      # one budget for both attempts, so an upload is never held up long
        for source in ("osd", "slate"):
            try:
                t0 = read_burned_in_time(p, client, mode=source, tz_name=tz_name, budget_s=deadline - time.monotonic())
            except Exception:  # noqa: BLE001 - a failing vision model must never block ingest
                log.exception("%s clock reading failed for %s", source, p)
                t0 = None
            if t0 is not None:
                return t0, source
    try:
        return p.stat().st_mtime, "manual"
    except OSError:
        log.warning("cannot stat %s; using the current time", p)
        return datetime.now(tz=UTC).timestamp(), "manual"
