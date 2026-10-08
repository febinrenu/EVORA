"""Download WILDTRACK (7 overlapping 1080p cameras, EPFL).

Usage:
  python scripts/data/fetch_wildtrack.py --zip [--out data/raw/wildtrack]     # frames + annotations, ~6.8 GB
  python scripts/data/fetch_wildtrack.py --videos 1sGUnExmJM2_tFuBd9LNlexf0LN2m0_c-  # Drive ids per camera
  python scripts/data/fetch_wildtrack.py --frames-to-mp4 data/raw/wildtrack/Wildtrack_dataset/Image_subsets

The zip is large, so nothing downloads unless you pass --zip. Resumable. If Google Drive refuses
(quota), build mp4s from the 10 fps frames with --frames-to-mp4.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("fetch_wildtrack")
ZIP_URL = "http://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip"
CHUNK = 1 << 20


def download_zip(dest: Path, attempts: int = 12) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(ZIP_URL, method="HEAD"), timeout=30) as r:
        total = int(r.headers["Content-Length"])
    for i in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == total:
            return True
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(ZIP_URL, headers=headers), timeout=60) as r, \
                    open(dest, "ab" if have else "wb") as f:
                while chunk := r.read(CHUNK):
                    f.write(chunk)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("attempt %d/%d interrupted at %.2f GB: %s", i, attempts, have / 1e9, exc)
            time.sleep(min(3 * i, 20))
    return dest.exists() and dest.stat().st_size == total


def extract(zip_path: Path, out: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    log.info("extracted to %s", out)


def drive_download(file_id: str, dest: Path) -> bool:
    try:
        import gdown
    except ImportError:
        log.error("gdown is not installed: pip install gdown")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    return bool(gdown.download(id=file_id, output=str(dest), quiet=False))


def frames_to_mp4(subsets: Path, out: Path, fps: int = 10) -> int:
    """Build one mp4 per camera folder (C1..C7) from numbered PNG frames."""
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for cam_dir in sorted(p for p in subsets.iterdir() if p.is_dir() and p.name.upper().startswith("C")):
        target = out / f"{cam_dir.name.lower()}.mp4"
        cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-framerate", str(fps),
               "-pattern_type", "glob", "-i", str(cam_dir / "*.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
        if subprocess.run(cmd).returncode:
            log.error("ffmpeg failed for %s", cam_dir)
            failures += 1
        else:
            log.info("wrote %s", target)
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw/wildtrack"))
    ap.add_argument("--zip", action="store_true", help="download and extract the full frames zip (~6.8 GB)")
    ap.add_argument("--videos", nargs="*", metavar="DRIVE_ID", help="Drive ids in camera order (c1, c2, ...)")
    ap.add_argument("--frames-to-mp4", type=Path, metavar="SUBSETS_DIR")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rc = 0
    if args.zip:
        z = args.out / "Wildtrack_dataset_full.zip"
        if download_zip(z):
            extract(z, args.out)
        else:
            log.error("zip download incomplete")
            rc = 1
    for i, fid in enumerate(args.videos or [], start=1):
        if not drive_download(fid, args.out / "videos" / f"c{i}.mp4"):
            log.error("Drive download failed for camera %d; use --frames-to-mp4 instead", i)
            rc = 1
    if args.frames_to_mp4:
        rc |= 1 if frames_to_mp4(args.frames_to_mp4, args.out / "videos") else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
