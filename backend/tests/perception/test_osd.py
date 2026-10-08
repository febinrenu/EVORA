import subprocess
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")

from evora.perception import clock, osd, vision  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))


def ist(*a):
    return datetime(*a, tzinfo=IST).timestamp()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2018-03-09T10:10:03", ist(2018, 3, 9, 10, 10, 3)),
        ("2018-03-09 10:10:03", ist(2018, 3, 9, 10, 10, 3)),
        ("2018/03/09 10:10:03", ist(2018, 3, 9, 10, 10, 3)),
        ("Date and time: 2018.03.09 22:10:03.", ist(2018, 3, 9, 22, 10, 3)),
        ("`09/03/2018 10:10:03`", ist(2018, 3, 9, 10, 10, 3)),             # day first
        ("03/25/2018 10:10:03", ist(2018, 3, 25, 10, 10, 3)),              # only month-first is valid
        ("2018-03-09 10:10:03 PM", ist(2018, 3, 9, 22, 10, 3)),
        ("2018-03-09 12:05:00 AM", ist(2018, 3, 9, 0, 5, 0)),
        ("9 Mar 2018 10:10:03", ist(2018, 3, 9, 10, 10, 3)),
        ("March 9, 2018 10:10 PM", ist(2018, 3, 9, 22, 10, 0)),
    ],
)
def test_parse_timestamp_formats(text, expected):
    assert osd.parse_timestamp(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    ["NONE", "none.", "", "no clock visible", "1999-01-01 00:00:00", "2018-13-45 10:10:10", "2018-03-09 25:10:10",
     "2018-03-09 13:10:10 PM"],
)
def test_parse_timestamp_rejects(text):
    assert osd.parse_timestamp(text) is None


def test_time_only_needs_a_day():
    assert osd.parse_timestamp("14:05:09") is None
    day = datetime(2026, 10, 8, 3, 0, tzinfo=IST)
    assert osd.parse_timestamp("14:05:09", default_day=day) == pytest.approx(ist(2026, 10, 8, 14, 5, 9))


class ScriptedVision:
    """Answers each describe() call from a list, and records the sizes of the images it was shown."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def describe(self, image_jpeg, prompt, *, max_tokens=64):
        self.calls.append((len(image_jpeg), prompt[:20]))
        return self.replies.pop(0) if self.replies else None


@pytest.fixture
def clip(tmp_path):
    out = tmp_path / "cam.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:duration=8",
         "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    return out


def test_two_consistent_readings_give_the_start_time(clip):
    # the frames sit at about 0.5 s and 5.5 s; they read 10:10:00 and 10:10:05, so frame 0 was at 10:10:00 - 0.5 s
    v = ScriptedVision(["2018-03-09T10:10:00", "2018-03-09T10:10:05"])
    t0 = osd.read_burned_in_time(clip, v)
    assert t0 == pytest.approx(ist(2018, 3, 9, 10, 10, 0) - 0.5, abs=0.2)
    assert len(v.calls) == 2 and all(size > 1000 for size, _ in v.calls)


def test_inconsistent_readings_are_rejected(clip):
    v = ScriptedVision(["2018-03-09T10:10:00", "2018-03-09T10:30:00"])
    assert osd.read_burned_in_time(clip, v) is None


def test_no_text_or_a_failed_model_gives_none(clip):
    assert osd.read_burned_in_time(clip, ScriptedVision(["NONE"])) is None
    assert osd.read_burned_in_time(clip, ScriptedVision([None])) is None
    assert osd.read_burned_in_time(clip, ScriptedVision(["2018-03-09T10:10:00", "garbage"])) is None


def test_a_clip_too_short_to_check_is_not_trusted(tmp_path):
    short = tmp_path / "short.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=1",
         "-pix_fmt", "yuv420p", str(short)],
        check=True,
    )
    v = ScriptedVision(["2018-03-09T10:10:00", "2018-03-09T10:10:01"])
    assert osd.read_burned_in_time(short, v) is None and not v.calls


def test_slate_mode_uses_the_file_day_for_a_time_only_clock(clip):
    v = ScriptedVision(["14:05:09", "14:05:12"])
    t0 = osd.read_burned_in_time(clip, v, mode="slate")
    assert t0 is not None
    assert datetime.fromtimestamp(t0, IST).strftime("%H:%M:%S") in ("14:05:08", "14:05:09")
    assert v.calls[0][1].startswith("This frame may sh")


def test_corner_sheet_is_a_decodable_enlarged_image():
    import cv2
    import numpy as np

    frame = np.full((360, 640, 3), 90, dtype=np.uint8)
    img = cv2.imdecode(np.frombuffer(osd.corner_sheet(frame), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] >= 240 and img.shape[1] > 400


def test_detect_clock_prefers_filename_then_metadata_then_vision_then_manual(clip, tmp_path):
    t0, source = clock.detect_clock(clip, vision=ScriptedVision(["2018-03-09T10:10:00", "2018-03-09T10:10:05"]))
    assert source == "osd" and t0 == pytest.approx(ist(2018, 3, 9, 10, 10, 0) - 0.5, abs=0.2)
    t0, source = clock.detect_clock(clip, vision=ScriptedVision(["NONE", "NONE", "NONE", "NONE"]))
    assert source == "manual" and t0 == pytest.approx(clip.stat().st_mtime)
    named = tmp_path / "2026-10-08.12-00-00.12-05-00.site.G001.r13.mp4"
    named.write_bytes(clip.read_bytes())
    unused = ScriptedVision([])
    assert clock.detect_clock(named, vision=unused)[1] == "filename" and not unused.calls


def test_the_registered_client_is_used_automatically(clip):
    vision.register(ScriptedVision(["2018-03-09T10:10:00", "2018-03-09T10:10:05"]))
    try:
        assert clock.detect_clock(clip)[1] == "osd"
    finally:
        vision.register(None)
    assert clock.detect_clock(clip)[1] == "manual"


def test_a_failing_vision_client_never_breaks_clock_detection(clip):
    class Boom:
        def describe(self, *a, **k):
            raise RuntimeError("model crashed")

    assert clock.detect_clock(clip, vision=Boom())[1] == "manual"


def test_loop_client_runs_the_async_call_on_its_loop_and_survives_errors():
    import asyncio
    import threading

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    seen = {}

    async def ok(image, prompt, *, local_only, max_tokens):
        seen.update(local_only=local_only, max_tokens=max_tokens, in_loop=asyncio.get_running_loop() is loop)
        return "hello"

    async def bad(image, prompt, *, local_only, max_tokens):
        raise RuntimeError("down")

    try:
        assert vision.LoopVisionClient(ok, loop).describe(b"x", "p", max_tokens=12) == "hello"
        assert seen == {"local_only": True, "max_tokens": 12, "in_loop": True}      # people crops are always local-only
        assert vision.LoopVisionClient(bad, loop).describe(b"x", "p") is None
    finally:
        loop.call_soon_threadsafe(loop.stop)
