from datetime import UTC, datetime, timedelta, timezone

import pytest

pytest.importorskip("av")

from evora.perception import clock  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("2018-03-05.11-05-00.11-10-00.hospital.G436.r13.avi", datetime(2018, 3, 5, 11, 5, 0, tzinfo=IST)),
        ("VID_20261008_091503.mp4", datetime(2026, 10, 8, 9, 15, 3, tzinfo=IST)),
        ("IMG_20261008-091503.mov", datetime(2026, 10, 8, 9, 15, 3, tzinfo=IST)),
        ("cam2_2026-10-08_09-15-03.mp4", datetime(2026, 10, 8, 9, 15, 3, tzinfo=IST)),
        ("NVR 2026.10.08 09.15.03 gate.mkv", datetime(2026, 10, 8, 9, 15, 3, tzinfo=IST)),
    ],
)
def test_filename_patterns(name, expected):
    assert clock.parse_filename(name, "Asia/Kolkata") == pytest.approx(expected.timestamp())


@pytest.mark.parametrize(
    "name",
    ["gate.mp4", "terrace1-c0.mp4", "20269999_999999.mp4", "1999-01-01.00-00-00.x.avi", "clip_12345678901234.mp4"],
)
def test_filename_without_a_plausible_timestamp(name):
    assert clock.parse_filename(name) is None


def test_metadata_creation_time(sample_mp4):
    t0, source = clock.detect_clock(sample_mp4)
    assert source == "metadata"
    assert t0 == pytest.approx(datetime(2026, 10, 8, 9, 0, 0, tzinfo=UTC).timestamp())


def test_filename_wins_over_metadata(sample_mp4, tmp_path):
    named = tmp_path / "2026-10-08.12-00-00.12-05-00.site.G001.r13.mp4"
    named.write_bytes(sample_mp4.read_bytes())
    _, source = clock.detect_clock(named)
    assert source == "filename"


def test_osd_reader_used_when_nothing_else(tmp_path):
    f = tmp_path / "plain.mp4"
    f.write_bytes(b"not a real video")
    t0, source = clock.detect_clock(f, osd_reader=lambda p: 1_790_000_000.0)
    assert (t0, source) == (1_790_000_000.0, "osd")


def test_osd_failure_falls_back_to_manual(tmp_path):
    f = tmp_path / "plain.mp4"
    f.write_bytes(b"not a real video")

    def boom(_):
        raise RuntimeError("vlm offline")

    t0, source = clock.detect_clock(f, osd_reader=boom)
    assert source == "manual"
    assert t0 == pytest.approx(f.stat().st_mtime)


def test_missing_file_never_raises(tmp_path):
    t0, source = clock.detect_clock(tmp_path / "gone.mp4")
    assert source == "manual" and t0 > 0
