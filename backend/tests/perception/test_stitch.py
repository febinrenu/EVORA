"""Within-camera stitching and the strict cross-camera gates, on synthetic tracks with known truth."""
import json

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("lancedb")

from evora.perception.settings import IngestSettings  # noqa: E402
from evora.reid import link_global_ids  # noqa: E402

from .test_reid import World, identity, noisy, unit  # noqa: E402


def add(world, cam, t0, t1, vec, foot0, foot1, *, cls="person", attrs=None):
    """A track with a start and an end foot point (normalised), written the way ingest writes them."""
    tid = world.track(cam, t0, t1, vec, cls=cls, attrs=attrs)
    with world.db.write() as c:
        for t, (x, y) in ((t0, foot0), (t1, foot1)):
            c.execute("INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,?,?,?,?,?)",
                      (tid, t, x - 0.02, y - 0.2, x + 0.02, y, 0.9))
    return tid


def link(world, **settings):
    world.commit()
    link_global_ids(world.ws, IngestSettings(**settings))


def test_fragments_of_one_walk_become_one_identity(tmp_path):
    w = World(tmp_path, n_cams=1)
    v = identity()
    a = add(w, 0, 0, 10, noisy(v, 0.2), (0.2, 0.8), (0.5, 0.8))
    b = add(w, 0, 12, 20, noisy(v, 0.2), (0.52, 0.8), (0.7, 0.8))       # starts where a ended, two seconds later
    c = add(w, 0, 21, 30, noisy(v, 0.2), (0.72, 0.8), (0.9, 0.8))
    link(w)
    assert w.gid(a) == w.gid(b) == w.gid(c)


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("far away", {"foot0": (0.95, 0.2)}),
        ("too long a pause", {"t0": 45, "t1": 55}),
        ("overlaps in time", {"t0": 8, "t1": 18}),
        ("contradicting colours", {"attrs": {"upper_color": "blue"}}),
        ("different kind of object", {"cls": "car"}),
    ],
)
def test_things_that_are_not_the_same_person_stay_apart(tmp_path, name, kwargs):
    w = World(tmp_path, n_cams=1)
    v = identity()
    a = add(w, 0, 0, 10, noisy(v, 0.2), (0.2, 0.8), (0.5, 0.8), attrs={"upper_color": "red"})
    opts = {"t0": 12, "t1": 20, "foot0": (0.52, 0.8), "attrs": {"upper_color": "red"}} | kwargs
    b = add(w, 0, opts["t0"], opts["t1"], noisy(v, 0.2), opts["foot0"], (0.7, 0.8),
            cls=opts.get("cls", "person"), attrs=opts["attrs"])
    link(w)
    assert w.gid(a) != w.gid(b), name


def test_a_different_looking_person_who_appears_nearby_is_not_stitched(tmp_path):
    w = World(tmp_path, n_cams=1)
    a = add(w, 0, 0, 10, identity(), (0.2, 0.8), (0.5, 0.8))
    b = add(w, 0, 11, 20, identity(), (0.51, 0.8), (0.7, 0.8))          # right place and time, unrelated appearance
    link(w)
    assert w.gid(a) != w.gid(b)


def test_stitching_can_be_switched_off(tmp_path):
    w = World(tmp_path, n_cams=1)
    v = identity()
    a = add(w, 0, 0, 10, noisy(v, 0.2), (0.2, 0.8), (0.5, 0.8))
    b = add(w, 0, 12, 20, noisy(v, 0.2), (0.52, 0.8), (0.7, 0.8))
    link(w, reid_stitch=False)
    assert w.gid(a) != w.gid(b)


def test_each_track_gets_at_most_one_successor(tmp_path):
    w = World(tmp_path, n_cams=1)
    v = identity()
    a = add(w, 0, 0, 10, noisy(v, 0.15), (0.5, 0.8), (0.5, 0.8))
    b = add(w, 0, 12, 20, noisy(v, 0.15), (0.5, 0.8), (0.6, 0.8))
    c = add(w, 0, 13, 22, noisy(v, 0.15), (0.5, 0.8), (0.4, 0.8))       # two candidates start where a ended
    link(w)
    assert (w.gid(a) == w.gid(b)) != (w.gid(a) == w.gid(c)) or w.gid(b) != w.gid(c)
    assert w.gid(b) != w.gid(c)                                         # b and c overlap in time: never one person


def look_alikes(w, cos_noise):
    v = identity()
    a = w.track(0, 0, 10, v, attrs={"upper_color": "black", "color_conf": 0.9})
    b = w.track(1, 40, 50, noisy(v, cos_noise), attrs={"upper_color": "black", "color_conf": 0.9})
    # enough other people for the camera pair's background similarity to be measurable
    for _ in range(14):
        w.track(0, 100 + _ * 15, 108 + _ * 15, identity(), attrs={"upper_color": "black", "color_conf": 0.9})
        w.track(1, 140 + _ * 15, 148 + _ * 15, identity(), attrs={"upper_color": "black", "color_conf": 0.9})
    return a, b, v


def test_a_weak_appearance_match_is_not_enough_even_when_everyone_wears_black(tmp_path):
    w = World(tmp_path)
    a, b, _ = look_alikes(w, 1.6)                                       # cosine around 0.5, same colour, plausible gap
    row = np.array(w.rows[1]["vector"]) @ np.array(w.rows[0]["vector"])
    assert row < 0.72
    link(w)
    assert w.gid(a) != w.gid(b)


def test_a_strong_match_is_still_linked(tmp_path):
    w = World(tmp_path)
    a, b, _ = look_alikes(w, 0.2)
    link(w)
    assert w.gid(a) == w.gid(b)


@pytest.mark.parametrize("conf, linked", [(0.9, False), (0.3, True)])
def test_confidently_different_colours_veto_a_link_but_unsure_ones_do_not(tmp_path, conf, linked):
    w = World(tmp_path)
    v = identity()
    a = w.track(0, 0, 10, noisy(v, 0.15), attrs={"upper_color": "red", "color_conf": conf})
    b = w.track(1, 40, 50, noisy(v, 0.15), attrs={"upper_color": "blue", "color_conf": conf})
    link(w)
    assert (w.gid(a) == w.gid(b)) is linked


def test_the_veto_ignores_infrared_footage(tmp_path):
    w = World(tmp_path)
    v = identity()
    a = w.track(0, 0, 10, noisy(v, 0.15), attrs={"upper_color": "red", "color_conf": 0.9, "is_ir": True})
    b = w.track(1, 40, 50, noisy(v, 0.15), attrs={"upper_color": "blue", "color_conf": 0.9})
    link(w)
    assert w.gid(a) == w.gid(b)


def test_every_track_still_gets_an_identity(tmp_path):
    w = World(tmp_path, n_cams=1)
    ids = [add(w, 0, i * 20, i * 20 + 5, identity(), (0.3, 0.8), (0.4, 0.8)) for i in range(4)]
    link(w)
    with w.db.read() as c:
        gids = [r["global_id"] for r in c.execute("SELECT global_id FROM tracks")]
    assert all(gids) and len(set(gids)) == len(ids)
    assert json.dumps(sorted(gids))
    assert unit(np.ones(3)).shape == (3,)


# ---- within-camera re-clustering: fragments of one person come together, look-alikes in a crowd do not
def _unit(*xs):
    import numpy as np

    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def _track(tid, t0, t1, vec, cam="cam_01"):
    from evora.reid.associate import T

    return T(tid, cam, "person", t0, t1, vec, {}, (0.5, 0.9), (0.5, 0.9))


def test_one_person_coming_back_after_a_long_absence_is_one_identity():
    import numpy as np

    from evora.perception.settings import IngestSettings
    from evora.reid.associate import recluster_camera

    rng = np.random.default_rng(1)
    people = [_unit(*rng.normal(size=16)) for _ in range(3)]
    tracks = []
    for k in range(4):          # each of three people shows up four times, 30 s apart; three people are always in view together
        for p, base in enumerate(people):
            noisy = base + 0.05 * rng.normal(size=16).astype(np.float32)
            tracks.append(_track(f"p{p}_{k}", 100.0 * k + 5 * p, 100.0 * k + 5 * p + 60.0, noisy / np.linalg.norm(noisy)))
    groups = recluster_camera(tracks, IngestSettings(reid_recluster_min_pairs=5))
    assert sorted(len(g) for g in groups) == [4, 4, 4]
    for g in groups:
        assert len({t.id.split("_")[0] for t in g}) == 1          # nobody was joined to somebody else


def test_look_alikes_seen_together_keep_the_bar_high():
    import numpy as np

    from evora.perception.settings import IngestSettings
    from evora.reid.associate import recluster_camera

    rng = np.random.default_rng(2)
    common = _unit(*rng.normal(size=16))      # everyone in dark jackets: a shared direction plus a little of their own
    tracks = []
    for i in range(30):
        own = _unit(*rng.normal(size=16))
        v = 0.93 * common + 0.37 * own
        tracks.append(_track(f"t{i}", 10.0 * (i // 10), 10.0 * (i // 10) + 8.0, v / np.linalg.norm(v)))
    # ten at a time, 3 waves: concurrent tracks are all look-alikes, so the cosine bar becomes their own level
    groups = recluster_camera(tracks, IngestSettings(reid_recluster_min_pairs=5, reid_recluster_floor=0.5))
    assert max((len(g) for g in groups), default=0) <= 3        # no big blob of look-alikes
    for g in groups:                                              # and nobody seen at the same time was joined
        assert all(min(a.t1, b.t1) - max(a.t0, b.t0) <= 0.5 for a in g for b in g if a is not b)


def test_too_few_concurrent_pairs_means_no_reclustering():
    from evora.perception.settings import IngestSettings
    from evora.reid.associate import recluster_camera

    tracks = [_track(f"t{i}", 100.0 * i, 100.0 * i + 5, _unit(1, 0.1 * i, 0)) for i in range(6)]   # never two at once
    assert recluster_camera(tracks, IngestSettings()) == []


def test_count_groups_merge_fragments_but_identities_do_not():
    import numpy as np

    from evora.perception.settings import IngestSettings
    from evora.reid.associate import group_tracks

    rng = np.random.default_rng(3)
    people = [_unit(*rng.normal(size=16)) for _ in range(3)]
    tracks = []
    for k in range(4):
        for p, base in enumerate(people):
            noisy = base + 0.05 * rng.normal(size=16).astype(np.float32)
            tracks.append(_track(f"p{p}_{k}", 100.0 * k + 5 * p, 100.0 * k + 5 * p + 60.0, noisy / np.linalg.norm(noisy)))
    identities = [[t] for t in tracks]          # as if linking left every track alone
    groups = group_tracks(tracks, identities, IngestSettings(reid_recluster_min_pairs=5))
    assert sorted(len(g) for g in groups) == [4, 4, 4]
    assert len(group_tracks(tracks, identities, IngestSettings(reid_recluster=False))) == 12
