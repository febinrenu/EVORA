"""Shared fixtures for tests that need a populated workspace (SQLite + LanceDB) and toy embeddings."""
from __future__ import annotations

import json

import numpy as np

from evora.core.db import Database, close_all, open_db
from evora.core.vectors import ensure_tables, open_store

DIM = 4
E = np.eye(DIM, dtype=np.float32)  # axis 0 red-car, 1 blue-car, 2 generic car, 3 sedan-only


class ToyEmbedder:
    def embed_text(self, text: str) -> np.ndarray:
        t = text.lower()
        if "sedan" in t:
            return E[3]
        if "red" in t:
            return E[0]
        if "blue" in t:
            return E[1]
        return E[2]


class Workspace:
    def __init__(self, path=None, db: Database | None = None, vectors_dir=None) -> None:
        """A fresh workspace under `path`, or wrap an existing database and vector folder."""
        self.db: Database = db if db is not None else open_db(path / "evora.db")
        self.store = open_store(vectors_dir if vectors_dir is not None else path / "vectors")
        ensure_tables(self.store, {"embed_dim_image": DIM, "embed_dim_text": 384})

    def close(self) -> None:
        close_all()

    def camera(self, cid: str, name: str | None = None, t0: float = 1000.0, duration: float = 300.0,
               layers=("L0", "L1"), ir: float | None = None, source: str | None = None) -> None:
        with self.db.write() as c:
            c.execute(
                "INSERT INTO cameras(id,name,kind,source_uri,t0,t0_source,duration_s,layers,ir_fraction,created_at) "
                "VALUES(?,?, 'file',?,?, 'manual',?,?,?,0)",
                (cid, name or cid, source or f"{cid}.mp4", t0, duration, json.dumps(list(layers)), ir))

    def track(self, tid: str, cam: str, cls: str = "car", t0: float = 1100.0, t1: float = 1110.0, attrs=None,
              gid: str | None = None, crops=(E[0],), best_t: float | None = None, bbox=None,
              points=None) -> None:
        """`bbox` gives the track stored points at 4 Hz over its span (like real tracks); `points` lists exact times."""
        with self.db.write() as c:
            c.execute("INSERT INTO tracks(id,camera_id,cls,t_start,t_end,n_obs,best_t,attrs,global_id) "
                      "VALUES(?,?,?,?,?,?,?,?,?)", (tid, cam, cls, t0, t1, 10, best_t, json.dumps(attrs or {}), gid))
            box = bbox if bbox is not None else ((0.1, 0.1, 0.2, 0.3) if points is not None else None)
            if box is not None:
                times = list(points) if points is not None else [t0 + k * 0.25 for k in range(int((t1 - t0) * 4) + 1)]
                for t in times:
                    c.execute("INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,?,?,?,?,1)",
                              (tid, t, *box))
        rows = [{"vector": [float(x) for x in v], "track_id": tid, "camera_id": cam, "cls": cls, "t": t0 + i,
                 "quality": 0.5, "crop_path": f"{tid}_{i}.jpg"} for i, v in enumerate(crops)]
        if rows:
            self.store.open_table("crops").add(rows)

    def event(self, eid: str, cam: str, tid: str, kind: str, t: float, zone: str | None = "z1", **payload) -> None:
        with self.db.write() as c:
            c.execute("INSERT INTO events(id,camera_id,track_id,kind,zone_id,t,payload) VALUES(?,?,?,?,?,?,?)",
                      (eid, cam, tid, kind, zone, t, json.dumps(payload)))

    def zone(self, zid: str, cam: str, kind: str = "line", points=((0.1, 0.7), (0.9, 0.7)), direction="any") -> None:
        with self.db.write() as c:
            c.execute("INSERT INTO zones(id,camera_id,kind,points,direction,created_at) VALUES(?,?,?,?,?,0)",
                      (zid, cam, kind, json.dumps([list(p) for p in points]), direction))

    def scene(self, cam: str, t: float, vec, tile: str = "full") -> None:
        self.store.open_table("scenes").add([{"vector": [float(x) for x in vec], "camera_id": cam, "t": t,
                                              "tile": tile, "frame_path": f"{cam}_{t}.jpg"}])
