"""Camera topology: how long people and vehicles take to get from one camera to another.

Learned from the most confident appearance matches (no manual calibration), stored in `camera_links`
as the mean and spread of the time gap between leaving one camera and appearing in the other, with a
flag for overlapping views. During linking the gap of a candidate pair is scored against this
distribution (contribution C5). A link is stored once per pair, with `cam_a <= cam_b`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evora.core.db import Database
from evora.perception.settings import IngestSettings

MIN_LINK_SAMPLES = 3


@dataclass(frozen=True)
class Link:
    cam_a: str
    cam_b: str
    mean_dt: float
    std_dt: float
    n: int
    overlap: bool


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def robust_centre_spread(gaps: np.ndarray, st: IngestSettings) -> tuple[float, float]:
    """Median and MAD-based spread (floored), so one wrong match among the confident ones does not distort the link."""
    centre = float(np.median(gaps))
    spread = 1.4826 * float(np.median(np.abs(gaps - centre)))
    return centre, max(spread, st.reid_std_floor_s)


def fit_links(samples: dict[tuple[str, str], list[tuple[float, bool]]], st: IngestSettings) -> dict[tuple[str, str], Link]:
    """One Link per camera pair from (gap seconds, overlapped?) samples of confident matches."""
    out: dict[tuple[str, str], Link] = {}
    for (a, b), obs in samples.items():
        if len(obs) < MIN_LINK_SAMPLES:
            continue
        gaps = np.array([g for g, _ in obs], dtype=np.float64)
        overlap_share = float(np.mean([o for _, o in obs]))
        centre, spread = robust_centre_spread(gaps, st)
        out[(a, b)] = Link(a, b, centre, spread, len(obs), overlap_share >= 0.5)
    return out


def topology_score(link: Link | None, gap_s: float, st: IngestSettings) -> float:
    """0..1 likelihood of this time gap for the camera pair; the prior when the pair has no learned link."""
    if link is None:
        return st.reid_topology_prior
    z = (gap_s - link.mean_dt) / max(link.std_dt, st.reid_std_floor_s)
    return float(math.exp(-0.5 * z * z))


def save_links(db: Database, links: Sequence[Link]) -> None:
    with db.write() as c:
        c.execute("DELETE FROM camera_links")
        c.executemany(
            "INSERT INTO camera_links(cam_a,cam_b,mean_dt,std_dt,n,overlap) VALUES(?,?,?,?,?,?)",
            [(lk.cam_a, lk.cam_b, lk.mean_dt, lk.std_dt, lk.n, int(lk.overlap)) for lk in links],
        )


def load_links(db: Database) -> dict[tuple[str, str], Link]:
    with db.read() as c:
        rows = c.execute("SELECT cam_a, cam_b, mean_dt, std_dt, n, overlap FROM camera_links").fetchall()
    return {
        (r["cam_a"], r["cam_b"]): Link(r["cam_a"], r["cam_b"], r["mean_dt"], r["std_dt"], r["n"], bool(r["overlap"]))
        for r in rows
    }
