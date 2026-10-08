"""Cross-camera association: appearance + attributes + travel-time topology -> global identities.

1. Candidate pairs: same class family, different cameras, plausible time gap (or overlapping time).
2. Bootstrap: confident appearance-only mutual matches teach the camera topology (`topology.fit_links`).
3. Final scoring: w_a * appearance cosine + w_t * attribute agreement + w_p * topology likelihood
   (the topology term is dropped when `reid_topology` is off).
4. Hungarian assignment per camera pair; accepted pairs are merged with union-find, refusing any merge
   that would put two simultaneous tracks of one camera into the same identity.
Every track ends with a `global_id` (singletons included) so paths and counts work uniformly.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binom

from evora.core.db import Database, open_db
from evora.core.vectors import open_store
from evora.core.workspace import Workspace
from evora.perception.settings import IngestSettings
from evora.reid.topology import Link, fit_links, pair_key, robust_centre_spread, save_links, topology_score

log = logging.getLogger("evora.reid.associate")

FAMILIES = {
    "person": "person", "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle", "bicycle": "bicycle",
}
BIG_COST = 1e6
MIN_POOL = 100   # pairs needed before the background similarity is trusted


@dataclass(frozen=True)
class T:
    id: str
    camera_id: str
    cls: str
    t0: float
    t1: float
    vec: np.ndarray
    attrs: dict


def attribute_agreement(a: dict, b: dict) -> float:
    """0..1 agreement of the stored attributes; unknown or infrared values count as neutral (0.5)."""
    if a.get("is_ir") or b.get("is_ir"):
        return 0.5
    keys = ["upper_color", "lower_color", "vehicle_type"]
    if not (a.get("upper_color") or b.get("upper_color")):
        keys.append("color")            # vehicles have a single colour; persons are compared by upper / lower
    parts = []
    for key in keys:
        x, y = a.get(key), b.get(key)
        parts.append(0.5 if x is None or y is None else (1.0 if x == y else 0.0))
    return float(np.mean(parts))


def time_gap(a: T, b: T) -> tuple[float, bool]:
    """(gap in seconds, overlapped?). Overlapping intervals have gap 0."""
    if a.t0 <= b.t1 and b.t0 <= a.t1:
        return 0.0, True
    return (b.t0 - a.t1 if b.t0 > a.t1 else a.t0 - b.t1), False


def load_tracks(db: Database, ws: Workspace, st: IngestSettings) -> list[T]:
    store = open_store(ws.vectors_dir)
    if "reid" not in set(store.list_tables().tables):
        return []
    vecs = {r["track_id"]: np.asarray(r["vector"], dtype=np.float32) for r in store.open_table("reid").to_arrow().to_pylist()}
    with db.read() as c:
        rows = c.execute("SELECT id, camera_id, cls, t_start, t_end, attrs FROM tracks").fetchall()
    return [
        T(r["id"], r["camera_id"], r["cls"], r["t_start"], r["t_end"], vecs[r["id"]], json.loads(r["attrs"] or "{}"))
        for r in rows if r["id"] in vecs and r["t_end"] - r["t_start"] >= st.reid_min_track_s
    ]


def _matrices(a: list[T], b: list[T], st: IngestSettings):
    """Appearance cosine, plausibility mask, gaps, overlaps and attribute agreement for two camera groups.

    Also returns `z`: each cosine in robust standard deviations above the median cosine of all same-family
    pairs of these two cameras, so that footage where everyone looks alike demands a closer match.
    """
    app = np.stack([t.vec for t in a]) @ np.stack([t.vec for t in b]).T
    fam = np.array([[FAMILIES.get(x.cls) is not None and FAMILIES.get(x.cls) == FAMILIES.get(y.cls) for y in b] for x in a])
    pool = app[fam] if fam.any() else app.ravel()
    if pool.size < MIN_POOL:
        z = np.full(app.shape, np.inf)      # too few pairs to know what 'everyone looks alike' means here
    else:
        med = float(np.median(pool))
        spread = max(1.4826 * float(np.median(np.abs(pool - med))), 1e-6)   # robust: true matches do not inflate it
        z = (app - med) / spread
    mask = np.zeros(app.shape, dtype=bool)
    gap = np.zeros(app.shape)
    over = np.zeros(app.shape, dtype=bool)
    attr = np.full(app.shape, 0.5)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if FAMILIES.get(x.cls) is None or FAMILIES.get(x.cls) != FAMILIES.get(y.cls):
                continue
            g, o = time_gap(x, y)
            if g <= st.reid_max_gap_s:
                mask[i, j], gap[i, j], over[i, j] = True, g, o
                attr[i, j] = attribute_agreement(x.attrs, y.attrs)
    return app, mask, gap, over, attr, z


def gaps_look_like_chance(matched: list[tuple[float, bool]], null: np.ndarray, st: IngestSettings) -> bool:
    """True when the matched time gaps are no more concentrated than the gaps of all candidate pairs.

    Counts how many matched gaps fall in a window around their mean and compares with the share of
    all candidate gaps that fall in the same window (one-sided binomial test).
    """
    gaps = np.array([g for g, _ in matched])
    centre, half = robust_centre_spread(gaps, st)
    inside = int(np.sum(np.abs(gaps - centre) <= half))
    base = float(np.mean(np.abs(null - centre) <= half))
    return float(binom.sf(inside - 1, len(gaps), min(max(base, 1e-6), 1.0))) > st.reid_topology_p


def _bootstrap(groups: dict[str, list[T]], st: IngestSettings) -> dict[tuple[str, str], Link]:
    samples: dict[tuple[str, str], list[tuple[float, bool]]] = defaultdict(list)
    for ca, cb in combinations(sorted(groups), 2):
        a, b = groups[ca], groups[cb]
        app, mask, gap, over, _, z = _matrices(a, b, st)
        score = np.where(mask, app, -1.0)
        for i in range(len(a)):
            order = np.argsort(-score[i])
            j = int(order[0])
            runner_up = score[i, order[1]] if len(order) > 1 else -1.0
            confident = score[i, j] >= st.reid_bootstrap_thr and z[i, j] >= st.reid_min_z
            if confident and score[i, j] - runner_up >= st.reid_margin and int(np.argmax(score[:, j])) == i:
                samples[pair_key(ca, cb)].append((float(gap[i, j]), bool(over[i, j])))
        key = pair_key(ca, cb)
        null = gap[mask]
        if key in samples and len(null) > 0 and len(samples[key]) >= 3 and gaps_look_like_chance(samples[key], null, st):
            log.info("topology %s-%s ignored: matched gaps look like chance", ca, cb)
            del samples[key]
    return fit_links(samples, st)


class _Union:
    def __init__(self, tracks: list[T]):
        self.parent = {t.id: t.id for t in tracks}
        self.members = {t.id: [t] for t in tracks}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        for x in self.members[ra]:
            for y in self.members[rb]:
                if x.camera_id == y.camera_id and time_gap(x, y)[1]:
                    return False   # two tracks seen at once by one camera are different objects
        self.parent[rb] = ra
        self.members[ra] += self.members.pop(rb)
        return True


def link_tracks(
    tracks: list[T], st: IngestSettings, links: dict[tuple[str, str], Link] | None = None,
) -> tuple[list[list[T]], dict[tuple[str, str], Link]]:
    """Groups of tracks judged to be the same identity, and the topology that was used."""
    groups: dict[str, list[T]] = defaultdict(list)
    for t in tracks:
        groups[t.camera_id].append(t)
    if links is None:
        links = _bootstrap(groups, st) if st.reid_topology else {}
    uf = _Union(tracks)
    w_app, w_attr, w_topo = st.reid_w_appearance, st.reid_w_attributes, st.reid_w_topology
    if not st.reid_topology:
        total = w_app + w_attr
        w_app, w_attr, w_topo = w_app / total, w_attr / total, 0.0
    candidates: list[tuple[float, str, str]] = []
    for ca, cb in combinations(sorted(groups), 2):
        a, b = groups[ca], groups[cb]
        app, mask, gap, _, attr, z = _matrices(a, b, st)
        link = links.get(pair_key(ca, cb))
        topo = np.zeros(app.shape)
        if w_topo:
            for i, j in zip(*np.nonzero(mask), strict=True):
                topo[i, j] = topology_score(link, float(gap[i, j]), st)
        score = w_app * app + w_attr * attr + w_topo * topo
        rows, cols = linear_sum_assignment(np.where(mask, -score, BIG_COST))
        for i, j in zip(rows, cols, strict=True):
            if mask[i, j] and score[i, j] >= st.reid_accept_thr and z[i, j] >= st.reid_min_z:
                candidates.append((float(score[i, j]), a[i].id, b[j].id))
    for _, x, y in sorted(candidates, reverse=True):   # strongest first, so a conflict drops the weaker link
        uf.union(x, y)
    clusters: dict[str, list[T]] = defaultdict(list)
    for t in tracks:
        clusters[uf.find(t.id)].append(t)
    return list(clusters.values()), links


def link_global_ids(workspace: Workspace | None = None, settings: IngestSettings | None = None) -> int:
    """Give every track of the workspace a global id; returns how many identities span two or more cameras."""
    from evora.perception.pipeline import resolve_workspace

    ws = workspace or resolve_workspace()
    st = settings or IngestSettings()
    db = open_db(ws.db_path)
    tracks = load_tracks(db, ws, st)
    clusters, links = link_tracks(tracks, st)
    linked = {t.id for m in clusters for t in m}
    with db.read() as c:   # tracks too short to link still get an identity of their own
        for r in c.execute("SELECT id, camera_id, cls, t_start, t_end FROM tracks").fetchall():
            if r["id"] not in linked:
                alone = T(r["id"], r["camera_id"], r["cls"], r["t_start"], r["t_end"], np.zeros(1, dtype=np.float32), {})
                clusters.append([alone])
    clusters.sort(key=lambda c: min(t.t0 for t in c))
    now = time.time()
    with db.write() as c:
        c.execute("UPDATE tracks SET global_id=NULL")
        c.execute("DELETE FROM global_ids")
        for n, members in enumerate(clusters, start=1):
            gid = f"g{n:06d}"
            cls = max({m.cls for m in members}, key=lambda k: sum(m.cls == k for m in members))
            c.execute("INSERT INTO global_ids(id, cls, label, created_at) VALUES(?,?,?,?)", (gid, cls, None, now))
            c.executemany("UPDATE tracks SET global_id=? WHERE id=?", [(gid, m.id) for m in members])
    save_links(db, list(links.values()))
    multi = sum(1 for m in clusters if len({t.camera_id for t in m}) > 1)
    log.info("%d tracks -> %d identities, %d across cameras, %d camera links", len(tracks), len(clusters), multi, len(links))
    return multi
