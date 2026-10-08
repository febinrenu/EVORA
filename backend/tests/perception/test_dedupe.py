"""One object tracked twice at the same time becomes one identity; two objects side by side stay two."""
import pytest

pytest.importorskip("scipy")
pytest.importorskip("lancedb")

from evora.perception.settings import IngestSettings  # noqa: E402
from evora.reid import link_global_ids  # noqa: E402
from evora.reid.associate import duplicate_pairs  # noqa: E402

from .test_reid import World, identity, noisy  # noqa: E402


def boxed(world, cam, t0, t1, vec, box_at, *, cls="person", step=0.25):
    """A track with a box at every `step` seconds; `box_at(t)` gives (x1, y1, x2, y2)."""
    tid = world.track(cam, t0, t1, vec, cls=cls)
    with world.db.write() as c:
        t = t0
        while t <= t1 + 1e-9:
            c.execute("INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,?,?,?,?,0.9)", (tid, t, *box_at(t)))
            t += step
    return tid


def walking(x0, dx=0.02, w=0.06, h=0.25, y=0.5, t0=0.0):
    """A box that walks right at `dx` per second."""
    return lambda t: (x0 + dx * (t - t0), y, x0 + dx * (t - t0) + w, y + h)


def link(world, **settings):
    world.commit()
    link_global_ids(world.ws, IngestSettings(**settings))


def test_two_ids_on_one_body_become_one_identity(tmp_path):
    w = World(tmp_path, n_cams=1)
    v = identity()
    a = boxed(w, 0, 0, 10, noisy(v, 0.1), walking(0.1))
    b = boxed(w, 0, 3, 8, noisy(v, 0.1), lambda t: tuple(c + 0.003 for c in walking(0.1)(t)))  # the same body, again
    assert duplicate_pairs(w.db, IngestSettings()) == [(a, b)], "the longer track is kept"
    link(w)
    assert w.gid(a) == w.gid(b)


def test_a_cut_off_partial_box_inside_the_full_one_is_the_same_person(tmp_path):
    w = World(tmp_path, n_cams=1)
    v = identity()
    full = boxed(w, 0, 0, 10, noisy(v, 0.1), walking(0.2, h=0.3))
    upper = boxed(w, 0, 2, 9, noisy(v, 0.4),
                  lambda t: (walking(0.2, h=0.3)(t)[0], 0.5, walking(0.2, h=0.3)(t)[2], 0.5 + 0.15))  # top half only
    link(w)
    assert w.gid(full) == w.gid(upper)


def test_two_people_walking_side_by_side_stay_two(tmp_path):
    w = World(tmp_path, n_cams=1)
    a = boxed(w, 0, 0, 10, identity(), walking(0.1))
    b = boxed(w, 0, 0, 10, identity(), walking(0.18))  # next to each other, boxes touch but do not coincide
    assert duplicate_pairs(w.db, IngestSettings()) == []
    link(w)
    assert w.gid(a) != w.gid(b)


def test_a_small_person_far_behind_inside_a_big_box_is_not_merged(tmp_path):
    w = World(tmp_path, n_cams=1)
    near = boxed(w, 0, 0, 10, identity(), lambda t: (0.2, 0.2, 0.6, 0.95))         # close to the camera, large
    far = boxed(w, 0, 0, 10, identity(), lambda t: (0.35, 0.3, 0.38, 0.38))        # inside it on screen, tiny
    assert duplicate_pairs(w.db, IngestSettings()) == [], "inside, but overlapping too little to be the same body"
    link(w)
    assert w.gid(near) != w.gid(far)


def test_a_person_and_a_car_are_never_duplicates(tmp_path):
    w = World(tmp_path, n_cams=1)
    boxed(w, 0, 0, 10, identity(), walking(0.1), cls="person")
    boxed(w, 0, 0, 10, identity(), walking(0.1), cls="car")
    assert duplicate_pairs(w.db, IngestSettings()) == []


def test_too_few_shared_moments_decide_nothing(tmp_path):
    w = World(tmp_path, n_cams=1)
    v = identity()
    boxed(w, 0, 0, 10, noisy(v, 0.1), walking(0.1))
    boxed(w, 0, 9.6, 10, noisy(v, 0.1), walking(0.1, t0=0.0))  # overlaps for two points only
    assert duplicate_pairs(w.db, IngestSettings()) == []


def test_tracks_that_do_not_overlap_in_time_are_left_to_stitching(tmp_path):
    w = World(tmp_path, n_cams=1)
    v = identity()
    boxed(w, 0, 0, 5, noisy(v, 0.1), walking(0.1))
    boxed(w, 0, 6, 10, noisy(v, 0.1), walking(0.1))
    assert duplicate_pairs(w.db, IngestSettings()) == []


def test_the_switch_restores_two_identities(tmp_path):
    w = World(tmp_path, n_cams=1)
    v = identity()
    a = boxed(w, 0, 0, 10, noisy(v, 0.1), walking(0.1))
    b = boxed(w, 0, 3, 8, noisy(v, 0.1), walking(0.1))
    link(w, reid_dedupe=False)
    assert w.gid(a) != w.gid(b)


def test_a_duplicate_too_short_to_have_a_reid_vector_still_joins(tmp_path):
    w = World(tmp_path, n_cams=1)
    v = identity()
    a = boxed(w, 0, 0, 10, noisy(v, 0.1), walking(0.1))
    b = boxed(w, 0, 4, 5, noisy(v, 0.1), walking(0.1))
    w.rows = [r for r in w.rows if r["track_id"] != b]  # no ReID vector for the short one, like a real fragment
    link(w)
    assert w.gid(a) == w.gid(b)
