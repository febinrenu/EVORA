"""Cross-camera linking on synthetic identities: unit-vector appearances with noise, known travel times."""
import json

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("lancedb")

from evora.core import cameras as cams  # noqa: E402
from evora.core import workspace as wsmod  # noqa: E402
from evora.core.db import open_db  # noqa: E402
from evora.core.vectors import dims_from_meta, ensure_tables, open_store  # noqa: E402
from evora.perception.settings import IngestSettings  # noqa: E402
from evora.reid import associate, link_global_ids, path_for, similar_tracks  # noqa: E402
from evora.reid.features import track_feature  # noqa: E402
from evora.reid.topology import fit_links, topology_score  # noqa: E402

DIM = 32
RNG = np.random.default_rng(7)


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def identity():
    return unit(RNG.normal(size=DIM))


def noisy(v, amount):
    return unit(v + amount * unit(RNG.normal(size=DIM)))


class World:
    def __init__(self, tmp_path, n_cams=3):
        self.ws = wsmod.create("reid-test", tmp_path / "w")
        self.db = open_db(self.ws.db_path)
        self.cams = [cams.insert_camera(self.db, name=f"Cam {i + 1}", kind="file", source_uri=f"c{i}.mp4", t0=0.0,
                                        t0_source="manual").id for i in range(n_cams)]
        self.db.set_meta("embed_dim_reid", str(DIM))
        self.store = open_store(self.ws.vectors_dir)
        ensure_tables(self.store, dims_from_meta(self.db), only={"reid"})
        self.rows = []
        self.n = 0

    def track(self, cam_index, t0, t1, vec, cls="person", attrs=None, quality=0.5):
        self.n += 1
        cam = self.cams[cam_index]
        tid = f"{cam}:t{self.n:06d}"
        with self.db.write() as c:
            c.execute("INSERT INTO tracks(id,camera_id,cls,t_start,t_end,n_obs,attrs,quality) VALUES(?,?,?,?,?,?,?,?)",
                      (tid, cam, cls, t0, t1, 20, json.dumps(attrs or {}), quality))
        self.rows.append({"vector": vec.tolist(), "track_id": tid, "camera_id": cam, "cls": cls, "t_start": t0, "t_end": t1})
        return tid

    def commit(self):
        self.store.open_table("reid").add(self.rows)

    def gid(self, tid):
        with self.db.read() as c:
            return c.execute("SELECT global_id FROM tracks WHERE id=?", (tid,)).fetchone()[0]


def walkers(world, n=4, noise=0.25, stagger=25.0):
    """n people who each walk camera 1 -> 2 -> 3 with gaps of about 30 s and 40 s, starting `stagger` s apart."""
    out = []
    for i in range(n):
        v, t = identity(), i * stagger
        a = world.track(0, t, t + 10, noisy(v, noise), attrs={"upper_color": ["red", "blue", "green", "black"][i % 4]})
        b = world.track(1, t + 40, t + 50, noisy(v, noise), attrs={"upper_color": ["red", "blue", "green", "black"][i % 4]})
        c = world.track(2, t + 90, t + 100, noisy(v, noise), attrs={"upper_color": ["red", "blue", "green", "black"][i % 4]})
        out.append((a, b, c))
    return out


def test_people_walking_through_three_cameras_get_one_identity_each(tmp_path):
    w = World(tmp_path)
    people = walkers(w)
    w.commit()
    assert link_global_ids(w.ws) == 4
    gids = [{w.gid(t) for t in trio} for trio in people]
    assert all(len(g) == 1 for g in gids) and len({next(iter(g)) for g in gids}) == 4


def test_topology_is_learned_from_confident_matches(tmp_path):
    w = World(tmp_path)
    walkers(w)
    w.commit()
    link_global_ids(w.ws)
    with w.db.read() as c:
        links = {(r["cam_a"], r["cam_b"]): r for r in c.execute("SELECT * FROM camera_links")}
    first = links[(w.cams[0], w.cams[1])]
    assert first["mean_dt"] == pytest.approx(30.0, abs=1.0) and first["n"] == 4 and first["overlap"] == 0
    assert links[(w.cams[1], w.cams[2])]["mean_dt"] == pytest.approx(40.0, abs=1.0)


def test_topology_breaks_an_appearance_tie_and_the_switch_turns_it_off(tmp_path):
    def build(topology):
        w = World(tmp_path / str(topology))
        walkers(w, n=4)                                       # teaches the 30 s gap between camera 1 and 2
        v = identity()
        source = w.track(0, 300, 310, v)
        true_match = w.track(1, 340, 350, noisy(v, 0.45))    # 30 s later, a bit less similar
        decoy = w.track(1, 450, 460, v)                      # 140 s later, identical appearance
        w.commit()
        link_global_ids(w.ws, IngestSettings(reid_topology=topology))
        return w, source, true_match, decoy

    w, source, true_match, decoy = build(True)
    assert w.gid(source) == w.gid(true_match) != w.gid(decoy)
    w, source, true_match, decoy = build(False)
    assert w.gid(source) == w.gid(decoy)                      # appearance alone prefers the look-alike


def test_two_simultaneous_tracks_of_one_camera_never_share_an_identity(tmp_path):
    w = World(tmp_path)
    v = identity()
    a = w.track(0, 0, 10, v)
    b = w.track(0, 2, 12, v)          # seen at the same time as a
    c = w.track(1, 40, 50, v)
    d = w.track(2, 90, 100, v)
    w.commit()
    link_global_ids(w.ws, IngestSettings(reid_topology=False))
    assert w.gid(a) != w.gid(b)
    assert w.gid(c) in {w.gid(a), w.gid(b)} and w.gid(d) in {w.gid(a), w.gid(b)}


def test_only_the_same_class_family_links(tmp_path):
    w = World(tmp_path)
    v = identity()
    p = w.track(0, 0, 10, v, cls="person")
    car = w.track(1, 40, 50, v, cls="car")
    w.commit()
    assert link_global_ids(w.ws) == 0
    assert w.gid(p) != w.gid(car)


def test_colour_disagreement_keeps_similar_looking_people_apart(tmp_path):
    w = World(tmp_path)
    v = identity()
    a = w.track(0, 0, 10, v, attrs={"upper_color": "red", "lower_color": "black"})
    b = w.track(1, 40, 50, noisy(v, 0.6), attrs={"upper_color": "blue", "lower_color": "grey"})
    w.commit()
    link_global_ids(w.ws, IngestSettings(reid_topology=False, reid_accept_thr=0.75))
    assert w.gid(a) != w.gid(b)


def test_overlapping_views_link_simultaneous_tracks(tmp_path):
    w = World(tmp_path, n_cams=2)
    people = []
    for i in range(4):
        v = identity()
        people.append((w.track(0, i * 20, i * 20 + 15, noisy(v, 0.2)), w.track(1, i * 20 + 1, i * 20 + 16, noisy(v, 0.2))))
    w.commit()
    assert link_global_ids(w.ws) == 4
    with w.db.read() as c:
        link = c.execute("SELECT * FROM camera_links").fetchone()
    assert link["overlap"] == 1 and link["mean_dt"] == 0


def test_path_merges_consecutive_tracks_per_camera_and_orders_by_time(tmp_path):
    w = World(tmp_path)
    v = identity()
    t1 = w.track(0, 0, 4, v, quality=0.2)
    t2 = w.track(0, 5, 10, v, quality=0.9)
    t3 = w.track(1, 40, 50, v)
    w.commit()
    with w.db.write() as c:
        c.execute("INSERT INTO global_ids(id, cls, created_at) VALUES('gX','person',0)")
        c.execute("UPDATE tracks SET global_id='gX'")
    hops = path_for("gX", db=w.db)
    assert [h.camera_id for h in hops] == [w.cams[0], w.cams[1]]
    assert (hops[0].t_in, hops[0].t_out) == (0, 10) and hops[0].camera_name == "Cam 1"
    assert hops[0].evidence_id == t2.replace(":", "_") and hops[1].evidence_id == t3.replace(":", "_")
    assert t1 != t2
    assert path_for("nobody", db=w.db) == []


def test_similar_tracks_ranks_by_appearance_within_the_class(tmp_path):
    w = World(tmp_path)
    v = identity()
    q = w.track(0, 0, 10, v)
    near = w.track(1, 40, 50, noisy(v, 0.2))
    far = w.track(2, 90, 100, identity())
    other_class = w.track(1, 60, 70, v, cls="car")
    w.commit()
    ranked = similar_tracks(q, k=5, store=w.store)
    assert [t for t, _ in ranked] == [near, far]
    assert ranked[0][1] > ranked[1][1] and other_class not in {t for t, _ in ranked}
    assert similar_tracks("cam_01:t999999", store=w.store) == []
    with pytest.raises(ValueError):
        similar_tracks("x'; drop", store=w.store)


def test_helpers():
    feats = np.stack([unit(RNG.normal(size=8)) for _ in range(3)])
    assert np.linalg.norm(track_feature(feats)) == pytest.approx(1.0)
    st = IngestSettings()
    links = fit_links({("a", "b"): [(30.0, False), (32.0, False), (28.0, False)], ("a", "c"): [(5.0, True)]}, st)
    assert set(links) == {("a", "b")}                            # a pair with too few samples learns nothing
    assert topology_score(links[("a", "b")], 30.0, st) == pytest.approx(1.0)
    assert topology_score(links[("a", "b")], 140.0, st) < 0.01
    assert topology_score(None, 10.0, st) == st.reid_topology_prior
    assert associate.attribute_agreement({"upper_color": "red"}, {"upper_color": "red"}) > associate.attribute_agreement(
        {"upper_color": "red"}, {"upper_color": "blue"})
    assert associate.attribute_agreement({"is_ir": True, "upper_color": "red"}, {"upper_color": "blue"}) == 0.5
