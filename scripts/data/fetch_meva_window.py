"""Download chosen cameras of one MEVA 5-minute window (the clips that start at the same time).

Usage: python scripts/data/fetch_meva_window.py --date 2018-03-09 --start 10-10-00 --site school \
           --cams G419 G423 G421 G424 G300 G328 G299 G330 --out data/raw/meva-2018-03-09

Pick cameras with `python scripts/meva_to_queries.py find` (windows with many annotated cameras), then
pass the annotated camera codes here. Resumable; skips files that are already complete.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_meva as fm  # noqa: E402

log = logging.getLogger("fetch_meva_window")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--start", required=True, help="HH-MM-SS start of the window, as in the clip names")
    ap.add_argument("--site", required=True)
    ap.add_argument("--cams", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    hour = args.start.split("-")[0]
    clips = [
        c for c in fm.list_clips(fm.s3_client(), args.date, [hour])
        if c.start.strftime("%H-%M-%S") == args.start and c.site == args.site and c.cam in set(args.cams)
    ]
    missing = set(args.cams) - {c.cam for c in clips}
    if missing:
        log.warning("not found in the bucket: %s", ", ".join(sorted(missing)))
    rc = 0
    for c in sorted(clips, key=lambda c: c.size):
        dest = args.out / Path(c.key).name
        if fm.download(c.key, dest):
            log.info("done %s (%.0f MB)", dest.name, c.size / 1e6)
        else:
            log.error("failed %s", c.key)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
