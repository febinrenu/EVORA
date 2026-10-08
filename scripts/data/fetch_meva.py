"""Find and download a small multi-camera slice of MEVA (AWS Open Data, no account needed).

Key format (checked 8 Oct 2026):
  drops-123-r13/<date>/<hour>/<date>.<HH-MM-SS>.<HH-MM-SS>.<site>.<camera>.r13.avi

Usage:
  python scripts/data/fetch_meva.py --list-days
  python scripts/data/fetch_meva.py --find --date 2018-03-05 --hours 09 10 11 13 14 [--min-cams 4]
  python scripts/data/fetch_meva.py --download --date 2018-03-05 --hours 09 --window-start 09:50:00 \
         [--max-clips 12] [--out data/raw/meva]

--find prints candidate windows (>= min-cams cameras recording at the same time) without downloading.
--download fetches the clips that cover one window, at most --max-clips of them. The camera set is
chosen from the cameras with the largest files, since near-empty files carry little activity.
The file name gives t0 (local site time); the annotation intersection (activity labels) is a
separate step against data/raw/meva-annotations and is not done here.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger("fetch_meva")
BUCKET = "mevadata-public-01"
PREFIX = "drops-123-r13/"
HTTPS = f"https://{BUCKET}.s3.amazonaws.com/"
KEY_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\.(?P<s>\d{2}-\d{2}-\d{2})\.(?P<e>\d{2}-\d{2}-\d{2})\.(?P<site>[a-z0-9]+)\.(?P<cam>G\d+)\.r\d+\.avi$"
)


@dataclass(frozen=True)
class Clip:
    key: str
    size: int
    start: datetime
    end: datetime
    site: str
    cam: str


def s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", region_name="us-east-1",
                        config=Config(signature_version=UNSIGNED, connect_timeout=15, read_timeout=60,
                                      retries={"max_attempts": 3}))


def parse_key(key: str, size: int) -> Clip | None:
    m = KEY_RE.search(key)
    if not m:
        return None
    d = m["date"]
    return Clip(key, size, datetime.strptime(f"{d} {m['s']}", "%Y-%m-%d %H-%M-%S"),
                datetime.strptime(f"{d} {m['e']}", "%Y-%m-%d %H-%M-%S"), m["site"], m["cam"])


def list_clips(s3, date: str, hours: list[str]) -> list[Clip]:
    clips: list[Clip] = []
    for h in hours:
        pages = s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{PREFIX}{date}/{h}/")
        for page in pages:
            for o in page.get("Contents", []):
                c = parse_key(o["Key"], o["Size"])
                if c:
                    clips.append(c)
    return clips


def find_windows(clips: list[Clip], min_cams: int, min_len_s: float = 240.0):
    """Group clips by identical (start, end) window; keep windows long enough with enough cameras."""
    groups: dict[tuple[datetime, datetime], list[Clip]] = defaultdict(list)
    for c in clips:
        if (c.end - c.start).total_seconds() >= min_len_s:
            groups[(c.start, c.end)].append(c)
    rows = [(k, v) for k, v in groups.items() if len({c.cam for c in v}) >= min_cams]
    return sorted(rows, key=lambda kv: sum(c.size for c in kv[1]), reverse=True)


def download(key: str, dest: Path, attempts: int = 6) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = HTTPS + key
    for i in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        try:
            head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30)
            total = int(head.headers["Content-Length"])
            if have == total:
                return True
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab" if have else "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError as exc:
            log.warning("attempt %d/%d for %s: %s", i, attempts, dest.name, exc)
            time.sleep(min(3 * i, 15))
    return dest.exists() and dest.stat().st_size == total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-days", action="store_true")
    ap.add_argument("--find", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--date")
    ap.add_argument("--hours", nargs="*", default=["09", "10"])
    ap.add_argument("--min-cams", type=int, default=4)
    ap.add_argument("--window-start", help="HH:MM:SS start of the window to download")
    ap.add_argument("--max-clips", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("data/raw/meva"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s3 = s3_client()

    if args.list_days:
        r = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, Delimiter="/")
        for c in r.get("CommonPrefixes", []):
            print(c["Prefix"])
        return 0
    if not args.date:
        ap.error("--date is required for --find/--download")

    clips = list_clips(s3, args.date, args.hours)
    log.info("%d clips under %s hours %s", len(clips), args.date, args.hours)
    windows = find_windows(clips, args.min_cams)
    if args.find or not args.download:
        for (s, e), v in windows[:15]:
            cams = sorted({c.cam for c in v})
            print(f"{s:%H:%M:%S}-{e:%H:%M:%S} cams={len(cams)} total={sum(c.size for c in v)/1e6:.0f} MB {cams}")
        return 0

    if not args.window_start:
        ap.error("--window-start is required with --download")
    pick = [(k, v) for k, v in windows if f"{k[0]:%H:%M:%S}" == args.window_start]
    if not pick:
        log.error("no window starting at %s with >= %d cameras", args.window_start, args.min_cams)
        return 1
    clips_sel = sorted(pick[0][1], key=lambda c: c.size, reverse=True)[: args.max_clips]
    rc = 0
    for c in clips_sel:
        if download(c.key, args.out / Path(c.key).name):
            log.info("done %s (%.0f MB)", Path(c.key).name, c.size / 1e6)
        else:
            log.error("failed %s", c.key)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
