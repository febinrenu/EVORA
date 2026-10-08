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
