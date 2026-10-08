"""Download the EPFL multi-camera pedestrian sequences (terrace, passageway, laboratory 6p).

Usage: python scripts/data/fetch_epfl.py [--out data/raw/epfl] [--only terrace1 6p ...]

Downloads are resumable (HTTP Range) and skipped when the file is already complete.
Standard library only, so it runs before the backend environment exists.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger("fetch_epfl")

BASE = "https://documents.epfl.ch/groups/c/cv/cvlab-pom-{bucket}/www/{name}.avi"

# sequence -> list of (camera suffix, bucket)
SEQUENCES: dict[str, list[tuple[str, str]]] = {
    "terrace1": [(f"c{i}", "video3") for i in range(4)],
    "passageway1": [("c0", "video2"), ("c1", "video2"), ("c2", "video2"), ("c3", "video3")],
    "6p": [(f"c{i}", "video1") for i in range(4)],
}

CHUNK = 1 << 20


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            length = r.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        log.warning("HEAD failed for %s: %s", url, exc)
        return None


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if total is not None and have == total:
        log.info("skip %s (complete, %.1f MB)", dest.name, have / 1e6)
        return True
    headers = {"Range": f"bytes={have}-"} if have and total and have < total else {}
    mode = "ab" if headers else "wb"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, mode) as f:
            while chunk := r.read(CHUNK):
                f.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.error("download failed for %s: %s", url, exc)
        return False
    size = dest.stat().st_size
    ok = total is None or size == total
    log.info("%s %s (%.1f MB)", "done" if ok else "INCOMPLETE", dest.name, size / 1e6)
    return ok


def download_with_retry(url: str, dest: Path, attempts: int = 8) -> bool:
    """The EPFL server sometimes drops connections mid-file; resume until complete."""
    for i in range(1, attempts + 1):
        if download(url, dest):
            return True
        log.warning("retry %d/%d for %s", i, attempts, dest.name)
        time.sleep(min(2 * i, 15))
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/raw/epfl"))
    ap.add_argument("--only", nargs="*", choices=sorted(SEQUENCES), help="sequences to fetch (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    jobs: list[tuple[str, Path]] = []
    for seq in args.only or SEQUENCES:
        for cam, bucket in SEQUENCES[seq]:
            name = f"{seq}-{cam}"
            jobs.append((BASE.format(bucket=bucket, name=name), args.out / f"{name}.avi"))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda j: download_with_retry(*j), jobs))
    failed = [j[1].name for j, ok in zip(jobs, results, strict=True) if not ok]
    if failed:
        log.error("failed: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
