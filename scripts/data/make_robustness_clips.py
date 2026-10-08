"""Build awkward test videos for the robustness drills (PLAN P2.19) from one normal clip.

Usage: python scripts/data/make_robustness_clips.py [--source data/norm/epfl/terrace1-c0.mp4] [--out data/robust] [--seconds 20]

Clips: rotated phone video, variable frame rate, truncated files (with and without the index up front),
night/infrared look, 4K, high fps, a one-frame video, matroska + HEVC, an audio-only file and an empty file. Needs ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("make_robustness_clips")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def build(source: Path, out: Path, seconds: int) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    base = out / "base.mp4"
    ffmpeg("-i", str(source), "-t", str(seconds), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(base))
    clips: dict[str, Path] = {"base": base}

    def make(name: str, *args: str, suffix: str = ".mp4") -> None:
        path = out / f"{name}{suffix}"
        ffmpeg(*args, str(path))
        clips[name] = path

    # phone video: pixels stored sideways with a display-rotation flag
    make("rotated90", "-display_rotation", "90", "-i", str(base), "-c", "copy")
    # variable frame rate: keep frames 0-99 and every third frame after that, timestamps untouched
    make("vfr", "-i", str(base), "-vf", "select=lt(n\\,100)+not(mod(n\\,3))", "-fps_mode", "passthrough",
         "-c:v", "libx264", "-pix_fmt", "yuv420p")
    # night / infrared look: grey, dark, noisy
    make("night_ir", "-i", str(base), "-vf", "eq=brightness=-0.25:contrast=1.3,noise=c0s=12:c0f=t,hue=s=0,format=yuv420p",
         "-c:v", "libx264")
    # 4K input from an upscaled copy
    make("uhd4k", "-i", str(base), "-vf", "scale=3840:2160", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p")
    # 60 fps
    make("fps60", "-i", str(base), "-vf", "fps=60", "-c:v", "libx264", "-pix_fmt", "yuv420p")
    # a camera that delivers a single frame
    make("single_frame", "-i", str(base), "-frames:v", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p")
    # matroska + HEVC
    make("hevc_mkv", "-i", str(base), "-c:v", "libx265", "-pix_fmt", "yuv420p", suffix=".mkv")
    # audio only
    make("audio_only", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:a", "aac")
    # copies that stop in the middle of the data: without the index at the front the file cannot be opened at all,
    # with it (faststart) the first part is still playable
    data = base.read_bytes()
    trunc = out / "truncated.mp4"
    trunc.write_bytes(data[: int(len(data) * 0.6)])
    clips["truncated"] = trunc
    make("faststart_full", "-i", str(base), "-c", "copy", "-movflags", "+faststart")
    fast = out / "truncated_faststart.mp4"
    fast_data = clips.pop("faststart_full").read_bytes()
    fast.write_bytes(fast_data[: int(len(fast_data) * 0.6)])
    (out / "faststart_full.mp4").unlink()
    clips["truncated_faststart"] = fast
    # an empty upload
    empty = out / "empty.mp4"
    empty.write_bytes(b"")
    clips["empty"] = empty
    return clips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("data/norm/epfl/terrace1-c0.mp4"))
    ap.add_argument("--out", type=Path, default=Path("data/robust"))
    ap.add_argument("--seconds", type=int, default=20)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.source.is_file():
        log.error("source clip missing: %s", args.source)
        return 1
    for name, path in build(args.source, args.out, args.seconds).items():
        log.info("%-14s %8.1f KB  %s", name, path.stat().st_size / 1024, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
