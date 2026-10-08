import subprocess

import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")

from evora.perception.decode import DecodeError, probe_video, read_frames  # noqa: E402


def test_probe(sample_mp4):
    info = probe_video(sample_mp4)
    assert (info.width, info.height, info.rotation) == (320, 240, 0)
    assert info.fps == pytest.approx(10.0)
    assert info.duration_s == pytest.approx(2.0, abs=0.2)


def test_frames_use_pts_and_are_monotonic(sample_mp4):
    frames = list(read_frames(sample_mp4))
    assert len(frames) == 20
    times = [f.pts_s for f in frames]
    assert times == sorted(times)
    assert times[1] - times[0] == pytest.approx(0.1, abs=1e-3)
    assert frames[0].image.shape == (240, 320, 3)


def test_downscale_keeps_aspect(sample_mp4):
    f = next(iter(read_frames(sample_mp4, max_width=160)))
    assert f.image.shape[1] == 160 and f.image.shape[0] == 120


def test_window(sample_mp4):
    times = [f.pts_s for f in read_frames(sample_mp4, start_s=0.5, end_s=1.0)]
    assert times and min(times) >= 0.5 and max(times) <= 1.0


def test_variable_frame_rate_keeps_real_timestamps(tmp_path):
    out = tmp_path / "vfr.mp4"
    # keep frames 0-2 and 13-19: the source has a gap, so index / fps would put later frames at the wrong time
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
         "-vf", "select=lt(n\\,3)+gte(n\\,13)", "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    times = [f.pts_s for f in read_frames(out)]
    assert len(times) == 10
    assert times == sorted(times)
    assert max(b - a for a, b in zip(times, times[1:], strict=False)) > 0.5  # the real gap survives


def test_rotation_metadata_is_applied(tmp_path, sample_mp4):
    out = tmp_path / "rot.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(sample_mp4), "-c", "copy", str(out)], check=True,
    )
    info = probe_video(out)
    f = next(iter(read_frames(out)))
    assert info.rotation in (90, 270)
    assert (info.width, info.height) == (240, 320)
    assert f.image.shape[:2] == (320, 240)


def test_unreadable_file_raises_decode_error(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage" * 100)
    with pytest.raises(DecodeError):
        list(read_frames(bad))
