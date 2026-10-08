"""Robustness drill (PLAN P2.19): run every awkward clip through the real pipeline and report what happens.

Usage: python scripts/robustness_drill.py [--clips data/robust] [--workspace robust] [--profile gpu] [--layers L0 L1 L2]
Build the clips first with scripts/data/make_robustness_clips.py. Exit code 1 if any clip behaves unexpectedly.

Expected outcomes: readable video -> ingest completes; unreadable input (empty, audio only) -> a clear
DecodeError at registration, never a crash or a half-registered camera.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO))

OK, REJECT = "ingests", "rejected cleanly"
EXPECT = {
    "base": OK, "rotated90": OK, "vfr": OK, "night_ir": OK, "uhd4k": OK, "fps60": OK, "single_frame": OK,
    "hevc_mkv": OK, "audio_only": REJECT, "empty": REJECT, "truncated": REJECT, "truncated_faststart": OK,
}


@dataclass
class Result:
    name: str
    outcome: str
    seconds: float
    detail: str


def run_one(name: str, path: Path, ws, db, profile: str, layers: set[str]) -> Result:
    from evora.core import cameras as cams
    from evora.perception.cli import _register
    from evora.perception.decode import DecodeError
    from evora.perception.pipeline import ingest

    started = time.monotonic()
    try:
        cam = _register(db, path, name, None)
    except DecodeError as exc:
        return Result(name, REJECT, time.monotonic() - started, str(exc)[:90])
    except Exception as exc:  # noqa: BLE001 - the drill reports any crash instead of stopping
        return Result(name, f"CRASH {type(exc).__name__}", time.monotonic() - started, str(exc)[:90])
    try:
        ingest(cam, profile, layers, lambda job: None, ws=ws)
    except DecodeError as exc:
        return Result(name, REJECT, time.monotonic() - started, f"after registering: {str(exc)[:70]}")
    except Exception as exc:  # noqa: BLE001
        return Result(name, f"CRASH {type(exc).__name__}", time.monotonic() - started, str(exc)[:90])
    cams.add_layers(db, cam.id, sorted(layers))
    cams.set_status(db, cam.id, "ready")
    with db.read() as c:
        n_tracks = c.execute("SELECT count(*) FROM tracks WHERE camera_id=?", (cam.id,)).fetchone()[0]
        ir = c.execute("SELECT ir_fraction FROM cameras WHERE id=?", (cam.id,)).fetchone()[0]
        attrs = [json.loads(r[0]) for r in c.execute("SELECT attrs FROM tracks WHERE camera_id=? AND cls='person'", (cam.id,))]
    scenes = sorted((ws.media_dir / "scenes" / cam.id).glob("*.jpg"))
    shape = ""
    if scenes:
        import cv2

        h, w = cv2.imread(str(scenes[0])).shape[:2]
        shape = f"scene {w}x{h}"
    ir_tracks = sum(1 for a in attrs if a.get("is_ir"))
    coloured = sum(1 for a in attrs if a.get("upper_color"))
    detail = f"{shape}, {n_tracks} tracks, ir_fraction={ir}, ir tracks={ir_tracks}, coloured persons={coloured}"
    return Result(name, OK, time.monotonic() - started, detail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=Path, default=Path("data/robust"))
    ap.add_argument("--workspace", default="robust")
    ap.add_argument("--profile", default="gpu")
    ap.add_argument("--layers", nargs="*", default=["L0", "L1", "L2"])
    ap.add_argument("--fresh", action="store_true", help="delete the drill workspace first")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import shutil

    from evora.core.db import open_db
    from evora.perception.pipeline import resolve_workspace

    ws = resolve_workspace(args.workspace)
    if args.fresh:
        from evora.core.db import close_all

        close_all()
        shutil.rmtree(ws.root, ignore_errors=True)
        ws = resolve_workspace(args.workspace)
    db = open_db(ws.db_path)
    results = []
    for path in sorted(args.clips.glob("*.*"), key=lambda p: (p.stem != "base", p.stem)):
        if path.suffix not in (".mp4", ".mkv"):
            continue
        results.append(run_one(path.stem, path, ws, db, args.profile, set(args.layers)))
        r = results[-1]
        print(f"{r.name:<13} {r.outcome:<18} {r.seconds:6.1f}s  {r.detail}", flush=True)
    bad = [r for r in results if EXPECT.get(r.name) not in (None, r.outcome)]
    for r in bad:
        print(f"UNEXPECTED {r.name}: wanted '{EXPECT[r.name]}', got '{r.outcome}'")
    with sqlite3.connect(ws.db_path) as c:
        orphans = c.execute("SELECT count(*) FROM cameras WHERE status='pending'").fetchone()[0]
    print(f"{len(results)} clips, {len(bad)} unexpected, cameras left pending: {orphans}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
