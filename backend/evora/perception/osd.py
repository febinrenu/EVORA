"""Read a camera's burned-in date and time (or a filmed phone clock) with the local vision model.

Vision models misread digits and sometimes invent a plausible timestamp, so a reading is only trusted when
two frames a few seconds apart give times that differ by what the video itself says (their presentation
timestamps), within 2 s. One reading, an implausible year, or two inconsistent readings all return None, and
`detect_clock` then falls back to the file time with the "manual" warning.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from evora.perception.clock import DEFAULT_TZ, _tz
from evora.perception.decode import DecodeError, read_frames
from evora.perception.vision import VisionClient

log = logging.getLogger("evora.perception.osd")

MIN_YEAR, MAX_YEAR = 2000, 2100
TOLERANCE_S = 2.0
MIN_SPAN_S = 2.0
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

OSD_PROMPT = (
    "These four crops are the corners of one video frame (top-left TL, top-right TR, bottom-left BL, bottom-right BR). "
    "A camera may have burned the current date and time into one of them. If you can read a date and time, answer with it "
    "in ISO 8601 as YYYY-MM-DDTHH:MM:SS. If there is no date and time text, answer NONE. Answer with the timestamp only."
)
SLATE_PROMPT = (
    "This frame may show a phone or clock that displays the current time. Read it. Answer in ISO 8601 as "
    "YYYY-MM-DDTHH:MM:SS if the date is shown, otherwise as HH:MM:SS in 24-hour time. If no clock is readable answer NONE. "
    "Answer with the timestamp only."
)

_CLOCK = r"[T\s,]+(?P<h>\d{1,2})[:.](?P<mi>\d{2})(?:[:.](?P<s>\d{2}))?\s*(?P<ap>[AaPp][Mm])?"
_ISO = re.compile(r"(?P<y>\d{4})[-/.](?P<mo>\d{1,2})[-/.](?P<d>\d{1,2})" + _CLOCK)
_DMY = re.compile(r"(?P<a>\d{1,2})[-/.](?P<b>\d{1,2})[-/.](?P<y>\d{4})" + _CLOCK)
_NAMED = re.compile(r"(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]{3})[a-z]*\.?,?\s+(?P<y>\d{4})" + _CLOCK)
_NAMED_US = re.compile(r"(?P<mon>[A-Za-z]{3})[a-z]*\.?\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{4})" + _CLOCK)
_TIME_ONLY = re.compile(r"(?<!\d)(?P<h>\d{1,2}):(?P<mi>\d{2})(?::(?P<s>\d{2}))?\s*(?P<ap>[AaPp][Mm])?(?!\d)")


def _hour(h: int, ap: str | None) -> int | None:
    if ap:
        if not 1 <= h <= 12:
            return None
        return h % 12 + (12 if ap.lower() == "pm" else 0)
    return h if 0 <= h <= 23 else None


def _make(tz, y: int, mo: int, d: int, h: int, mi: int, s: int, ap: str | None) -> float | None:
    hour = _hour(h, ap)
    if hour is None or not MIN_YEAR <= y <= MAX_YEAR or not 0 <= mi <= 59 or not 0 <= s <= 59:
        return None
    try:
        return datetime(y, mo, d, hour, mi, s, tzinfo=tz).timestamp()
    except ValueError:
        return None


def parse_timestamp(text: str, tz_name: str = DEFAULT_TZ, default_day: datetime | None = None) -> float | None:
    """Epoch seconds from a model reply, or None. Understands ISO, DD/MM/YYYY (month first only if the day is > 12),
    named months and 12-hour times. A time with no date uses `default_day` (a filmed phone clock shows no date)."""
    if not text or re.search(r"\bnone\b", text, re.I):
        return None
    tz = _tz(tz_name)
    if m := _ISO.search(text):
        return _make(tz, int(m["y"]), int(m["mo"]), int(m["d"]), int(m["h"]), int(m["mi"]), int(m["s"] or 0), m["ap"])
    for pattern in (_NAMED, _NAMED_US):
        if (m := pattern.search(text)) and m["mon"].lower()[:3] in _MONTHS:
            return _make(tz, int(m["y"]), _MONTHS[m["mon"].lower()[:3]], int(m["d"]), int(m["h"]), int(m["mi"]),
                         int(m["s"] or 0), m["ap"])
    if m := _DMY.search(text):
        a, b = int(m["a"]), int(m["b"])
        day, month = (b, a) if a <= 12 < b else (a, b)      # month-first only when it is the only valid reading
        return _make(tz, int(m["y"]), month, day, int(m["h"]), int(m["mi"]), int(m["s"] or 0), m["ap"])
    if default_day is not None and (m := _TIME_ONLY.search(text)):
        day = default_day.astimezone(tz)
        return _make(tz, day.year, day.month, day.day, int(m["h"]), int(m["mi"]), int(m["s"] or 0), m["ap"])
    return None


def corner_sheet(frame: np.ndarray, *, min_strip_h: int = 120) -> bytes:
    """A JPEG with the four corner strips of a frame stacked 2 x 2 and enlarged, so small burned-in text is legible."""
    h, w = frame.shape[:2]
    sh, sw = max(int(h * 0.16), 8), max(int(w * 0.46), 8)
    boxes = {"TL": (0, 0), "TR": (w - sw, 0), "BL": (0, h - sh), "BR": (w - sw, h - sh)}
    scale = max(1.0, min_strip_h / sh)
    tiles = []
    for name, (x, y) in boxes.items():
        tile = cv2.resize(frame[y:y + sh, x:x + sw], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.putText(tile, name, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(tile)
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError("could not encode the corner sheet")
    return buf.tobytes()


def full_frame_jpeg(frame: np.ndarray, max_side: int = 1280) -> bytes:
    h, w = frame.shape[:2]
    if max(h, w) > max_side:
        frame = cv2.resize(frame, None, fx=max_side / max(h, w), fy=max_side / max(h, w), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError("could not encode the frame")
    return buf.tobytes()


def _frames_near(path: Path, targets: list[float]) -> list[tuple[float, np.ndarray]]:
    """(presentation time, image) of the first frame at or after each target time."""
    found: list[tuple[float, np.ndarray]] = []
    pending = sorted(targets)
    for frame in read_frames(path, 1920, end_s=max(targets) + 2.0):
        while pending and frame.pts_s + 1e-6 >= pending[0]:
            found.append((frame.pts_s, frame.image))
            pending.pop(0)
        if not pending:
            break
    return found


def read_burned_in_time(path: str | Path, vision: VisionClient, *, mode: str = "osd", tz_name: str = DEFAULT_TZ,
                        budget_s: float = 20.0) -> float | None:
    """Epoch seconds of the video's first frame read from on-screen text (`mode='osd'`) or a filmed clock (`'slate'`).

    Returns None unless two readings agree with the video's own timeline.
    """
    p = Path(path)
    started = time.monotonic()
    targets = [0.5, 5.5] if mode == "osd" else [0.5, 3.5]
    prompt = OSD_PROMPT if mode == "osd" else SLATE_PROMPT
    try:
        frames = _frames_near(p, targets)
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
    except (DecodeError, OSError) as exc:
        log.warning("cannot read frames of %s for the clock: %s", p, exc)
        return None
    if len(frames) < 2 or frames[-1][0] - frames[0][0] < MIN_SPAN_S:
        log.info("%s is too short to verify an on-screen clock", p.name)
        return None
    readings: list[tuple[float, float]] = []
    for pts, image in frames[:2]:
        if time.monotonic() - started > budget_s:
            log.info("clock reading for %s ran out of time", p.name)
            return None
        jpeg = corner_sheet(image) if mode == "osd" else full_frame_jpeg(image)
        reply = vision.describe(jpeg, prompt, max_tokens=32)
        epoch = parse_timestamp(reply or "", tz_name, default_day=mtime.astimezone() if mode == "slate" else None)
        if epoch is None:
            log.info("no readable clock in %s at %.1f s (reply %r)", p.name, pts, (reply or "")[:60])
            return None
        readings.append((pts, epoch))
    (p1, e1), (p2, e2) = readings
    if abs((e2 - e1) - (p2 - p1)) > TOLERANCE_S:
        log.warning("clock readings of %s disagree with the video timeline (%.1f s vs %.1f s apart): ignored",
                    p.name, e2 - e1, p2 - p1)
        return None
    return e1 - p1


__all__ = ["read_burned_in_time", "parse_timestamp", "corner_sheet", "OSD_PROMPT", "SLATE_PROMPT"]
