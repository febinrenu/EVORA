"""Write data/manifest.json: ffprobe summary and SHA-256 for every video under a directory.

Usage: python scripts/data/manifest.py [--root data/norm] [--out data/manifest.json]
Only ffprobe summaries are produced; video bytes are hashed in chunks and never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("manifest")
EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path) -> dict | None:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout
        info = json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError) as exc:
        log.error("ffprobe failed for %s: %s", path, exc)
        return None
    streams = info.get("streams") or []
    if not streams:
        log.error("no video stream in %s", path)
        return None
    s, fmt = streams[0], info.get("format", {})
    num, _, den = s.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if den and float(den) else None
    nb = s.get("nb_frames", "")
    return {
        "codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"),
        "fps": fps, "frames": int(nb) if nb.isdigit() else None,
        "duration_s": float(fmt["duration"]) if "duration" in fmt else None,
        "size_bytes": int(fmt["size"]) if "size" in fmt else path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/norm"))
    ap.add_argument("--out", type=Path, default=Path("data/manifest.json"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    entries = {}
    for p in sorted(args.root.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        info = probe(p)
        if info is None:
            continue
        info["sha256"] = sha256(p)
        entries[p.relative_to(args.root).as_posix()] = info
        log.info("%s %sx%s %.1fs", p.name, info["width"], info["height"], info["duration_s"] or 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"root": args.root.as_posix(), "files": entries}, indent=2))
    log.info("wrote %s (%d files)", args.out, len(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
