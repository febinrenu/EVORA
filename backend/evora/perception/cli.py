"""Command line ingest, for building indexes without the API.

    python -m evora.perception.cli ingest data/norm/epfl/terrace1-c0.mp4 data/norm/epfl/terrace1-c1.mp4 \\
        --workspace mini --profile gpu --t0 2026-10-01T09:00:00+05:30

Files that are already registered in the workspace (same path) are re-ingested under their existing id.
`--t0` sets the same start time for every file, which is how synchronized sequences (EPFL) are aligned.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from contracts.models import CameraInfo, IngestJob

from evora.core import cameras as cams
from evora.core.db import open_db
from evora.core.vectors import open_store
from evora.perception.clock import detect_clock
from evora.perception.decode import probe_video
from evora.perception.pipeline import ingest, resolve_workspace

log = logging.getLogger("evora.perception.cli")


def _register(db, path: Path, name: str | None, t0_override: float | None) -> CameraInfo:
    uri = str(path.resolve())
    for cam in cams.list_cameras(db):
        if cam.source_uri == uri:
            return cam
    info = probe_video(path)
    if t0_override is not None:
        t0, source = t0_override, "manual"
    else:
        t0, source = detect_clock(path)
    return cams.insert_camera(
        db, name=name or path.stem, kind="file", source_uri=uri, t0=t0, t0_source=source,
        fps=info.fps, width=info.width, height=info.height, duration_s=info.duration_s,
    )


def _summary(db, ws) -> str:
    with db.read() as c:
        n_tracks = c.execute("SELECT count(*) FROM tracks").fetchone()[0]
        n_points = c.execute("SELECT count(*) FROM track_points").fetchone()[0]
    store = open_store(ws.vectors_dir)
    existing = set(store.list_tables().tables)
    counts = {t: store.open_table(t).count_rows() for t in ("crops", "scenes") if t in existing}
    return f"tracks={n_tracks} track_points={n_points} vectors={counts}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="register and ingest video files")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--workspace", help="workspace name (default: evora_WORKSPACE or the configured default)")
    p.add_argument("--profile", default="cpu", choices=["cpu", "gpu", "mps"])
    p.add_argument("--layers", nargs="*", default=["L0", "L1"], choices=["L0", "L1", "L2", "L3"])
    p.add_argument("--t0", help="ISO start time for every file, e.g. 2026-10-01T09:00:00+05:30")
    p.add_argument("--names", nargs="*", help="camera names, one per file")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    t0_override = datetime.fromisoformat(args.t0).timestamp() if args.t0 else None
    if args.names and len(args.names) != len(args.files):
        ap.error("--names needs one name per file")
    ws = resolve_workspace(args.workspace)
    db = open_db(ws.db_path)
    rc = 0
    last: dict[str, float] = {}
    for i, f in enumerate(args.files):
        if not f.is_file():
            log.error("not a file: %s", f)
            rc = 1
            continue
        cam = _register(db, f, args.names[i] if args.names else None, t0_override)
        def show(job: IngestJob, cam_id: str = cam.id) -> None:
            key = f"{cam_id}:{job.layer}"
            if job.progress >= 1.0 or job.progress - last.get(key, -1.0) >= 0.1:
                last[key] = job.progress
                rate = f" {job.video_s_per_s:.1f} video-s/s" if job.video_s_per_s else ""
                print(f"{cam_id} {job.layer} {job.progress * 100:5.1f}%{rate}", flush=True)

        try:
            ingest(cam, args.profile, set(args.layers), show, ws=ws)
        except Exception:  # noqa: BLE001 - report which file failed, keep going with the rest
            log.exception("ingest failed for %s", f)
            rc = 1
            continue
        cams.add_layers(db, cam.id, [ly for ly in args.layers if ly in ("L0", "L1")])
        cams.set_status(db, cam.id, "ready")
    print(_summary(db, ws))
    return rc


if __name__ == "__main__":
    sys.exit(main())
