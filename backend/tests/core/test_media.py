from datetime import UTC, datetime

import pytest

from evora.core import media

EXT = [".mp4", ".mov"]


def test_probe_reads_video_properties(sample_mp4):
    p = media.probe(sample_mp4)
    assert (p.width, p.height) == (320, 240)
    assert p.fps == pytest.approx(10.0)
    assert p.duration_s == pytest.approx(2.0, abs=0.2)
    assert p.codec == "h264" and p.rotation == 0
    assert datetime.fromisoformat(p.creation_time.replace("Z", "+00:00")) == datetime(2026, 10, 8, 9, 0, tzinfo=UTC)


def test_probe_rejects_non_video(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    with pytest.raises(media.UploadError) as e:
        media.probe(bad)
    assert e.value.status == 422


@pytest.mark.parametrize(
    "raw,expected",
    [("../../evil.mp4", "evil.mp4"), ("C:\\temp\\My Clip (1).MOV", "My_Clip_1.mov"), ("..mp4", "video.mp4")],
)
def test_safe_filename_strips_paths_and_odd_characters(raw, expected):
    assert media.safe_filename(raw, EXT) == expected


@pytest.mark.parametrize("raw", ["notes.txt", "noext", "clip.mp4.exe"])
def test_safe_filename_rejects_extensions(raw):
    with pytest.raises(media.UploadError) as e:
        media.safe_filename(raw, EXT)
    assert e.value.status == 422


def test_sink_hashes_and_stays_inside_uploads(tmp_path):
    sink = media.UploadSink(tmp_path / "up", "../../evil.mp4", 1000, EXT)
    sink.write(b"abc")
    sink.write(b"def")
    path, sha = sink.finish()
    assert path.parent == tmp_path / "up" and path.read_bytes() == b"abcdef"
    import hashlib

    assert sha == hashlib.sha256(b"abcdef").hexdigest()


def test_sink_enforces_size_cap_and_cleans_up(tmp_path):
    sink = media.UploadSink(tmp_path / "up", "big.mp4", 10, EXT)
    sink.write(b"12345")
    with pytest.raises(media.UploadError) as e:
        sink.write(b"678901")
    assert e.value.status == 413
    assert list((tmp_path / "up").iterdir()) == []


def test_sink_rejects_empty_upload(tmp_path):
    sink = media.UploadSink(tmp_path / "up", "e.mp4", 10, EXT)
    with pytest.raises(media.UploadError):
        sink.finish()
    assert list((tmp_path / "up").iterdir()) == []


@pytest.mark.parametrize("uri", ["http://x/y", "rtsp://", "rtsp://host/a b", "file:///etc/passwd", ""])
def test_rtsp_uri_validation_rejects(uri):
    with pytest.raises(media.UploadError):
        media.validate_rtsp_uri(uri)


def test_rtsp_uri_accepts_valid():
    assert media.validate_rtsp_uri(" rtsp://10.0.0.5:8554/cam1 ") == "rtsp://10.0.0.5:8554/cam1"
