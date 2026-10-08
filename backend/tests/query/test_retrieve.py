import json

import numpy as np
import pytest
from contracts.models import QueryPlan, Target, TimeWindow

from evora.core.db import close_all, open_db
from evora.core.vectors import ensure_tables, open_store
from evora.query.fuse import Calibration
from evora.query.retrieve import RetrievalConfig, Retriever, SearchScope

DIM = 4
E = np.eye(DIM, dtype=np.float32)  # axis 0 red-car, 1 blue-car, 2 generic car, 3 sedan-only


class ToyEmbedder:
    def embed_text(self, text):
        t = text.lower()
        if "sedan" in t:
            return E[3]
        if "red" in t:
            return E[0]
        if "blue" in t:
            return E[1]
        return E[2]


@pytest.fixture
def ws(tmp_path):
    db = open_db(tmp_path / "evora.db")
    store = open_store(tmp_path / "vectors")
    ensure_tables(store, {"embed_dim_image": DIM, "embed_dim_text": 384})
    yield SimpleWorkspace(db, store)
    close_all()


class SimpleWorkspace:
    def __init__(self, db, store):
        self.db, self.store = db, store

    def camera(self, cid, name=None):
        with self.db.write() as c:
            c.execute("INSERT INTO cameras(id,name,kind,source_uri,t0,t0_source,created_at) "
                      "VALUES(?,?, 'file','x.mp4',0,'manual',0)", (cid, name or cid))

    def track(self, tid, cam, cls="car", t0=100.0, t1=110.0, attrs=None, gid=None, crops=(E[0],), best_t=None,
              points=None):
        with self.db.write() as c:
            c.execute("INSERT INTO tracks(id,camera_id,cls,t_start,t_end,n_obs,best_t,attrs,global_id) "
                      "VALUES(?,?,?,?,?,?,?,?,?)",
                      (tid, cam, cls, t0, t1, 10, best_t, json.dumps(attrs or {}), gid))
            for t in points or ():  # exact times the track was on screen, like the 4 Hz stored track points
                c.execute("INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,0.1,0.1,0.2,0.3,1)", (tid, t))
        rows = [{"vector": [float(x) for x in v], "track_id": tid, "camera_id": cam, "cls": cls,
                 "t": t0 + i, "quality": 0.5, "crop_path": f"{tid}_{i}.jpg"} for i, v in enumerate(crops)]
        if rows:
            self.store.open_table("crops").add(rows)

    def scene(self, cam, t, vec, tile="full"):
        self.store.open_table("scenes").add([{"vector": [float(x) for x in vec], "camera_id": cam, "t": t,
                                              "tile": tile, "frame_path": f"{cam}_{t}.jpg"}])

    def caption(self, tid, cam, text):
        self.store.open_table("captions").add([{"vector": [0.0] * 384, "text": text, "camera_id": cam,
                                                "t": 100.0, "track_id": tid}])

    def retriever(self, **cfg):
        defaults = {"calibration": Calibration(0.5, 0.2)}  # toy axes: cos 1.0 match, 0.0 mismatch
        return Retriever(self.db, self.store, ToyEmbedder(), RetrievalConfig(**{**defaults, **cfg}))


def plan(attrs=("red",), cls=("car",), noun="car", embed="a photo of a red car", cams=(), window=None):
    return QueryPlan(intent="list", targets=[Target(noun=noun, cls=list(cls), attributes=list(attrs), embed_text=embed)],
                     camera_ids=list(cams), time=window)


def ids(result):
    return [c.track.id for c in result.candidates]


@pytest.mark.asyncio
async def test_ranks_the_matching_track_first_and_filters_by_class(ws):
    ws.camera("cam_01")
    ws.track("t_red", "cam_01", attrs={"color": "red", "color_conf": 0.9}, crops=[E[0], E[0]])
    ws.track("t_blue", "cam_01", attrs={"color": "blue", "color_conf": 0.9}, crops=[E[1]])
    ws.track("t_person", "cam_01", cls="person", attrs={"color": "red"}, crops=[E[0]])
    res = await ws.retriever().search(plan())
    assert ids(res)[0] == "t_red" and "t_person" not in ids(res)
    top = res.candidates[0]
    assert top.score > 0.8 and top.why[0].startswith("siglip") and "colour red 0.90" in top.why
    assert {"crops", "attributes"} <= set(res.layers)


@pytest.mark.asyncio
async def test_attributes_separate_look_alike_tracks_and_can_be_switched_off(ws):
    ws.camera("cam_01")
    ws.track("t_red", "cam_01", attrs={"color": "red", "color_conf": 1.0}, crops=[E[2]], t0=100.0)
    ws.track("t_blue", "cam_01", attrs={"color": "blue", "color_conf": 1.0}, crops=[E[2]], t0=200.0, t1=210.0)
    on = await ws.retriever().search(plan(embed="a photo of a car"))
    assert ids(on) == ["t_red", "t_blue"] and on.candidates[0].score > on.candidates[1].score
    off = await ws.retriever(attributes=False).search(plan(embed="a photo of a car"))
    assert off.candidates[0].score == pytest.approx(off.candidates[1].score)  # nothing left to tell them apart
    assert any("Attribute matching is switched off" in n for n in off.notes)


@pytest.mark.asyncio
async def test_infrared_tracks_are_not_judged_on_colour(ws):
    ws.camera("cam_01")
    ws.track("t_ir", "cam_01", attrs={"color": "blue", "color_conf": 1.0, "is_ir": True}, crops=[E[0]])
    res = await ws.retriever().search(plan())
    assert ids(res) == ["t_ir"] and not any(w.startswith("colour") for w in res.candidates[0].why)
    assert "attributes" not in res.layers


@pytest.mark.asyncio
async def test_camera_and_time_filters(ws):
    ws.camera("cam_01")
    ws.camera("cam_02")
    ws.track("a", "cam_01", t0=100.0, t1=110.0)
    ws.track("b", "cam_02", t0=100.0, t1=110.0)
    ws.track("c", "cam_01", t0=500.0, t1=510.0)
    assert set(ids(await ws.retriever().search(plan()))) == {"a", "b", "c"}
    only_cam1 = await ws.retriever().search(plan(), SearchScope(frozenset({"cam_01"})))
    assert set(ids(only_cam1)) == {"a", "c"}
    windowed = await ws.retriever().search(plan(), SearchScope(window=TimeWindow(start=450.0, end=600.0)))
    assert ids(windowed) == ["c"]
    overlap = await ws.retriever().search(plan(), SearchScope(window=TimeWindow(start=105.0, end=106.0)))
    assert set(ids(overlap)) == {"a", "b"}  # a track overlapping the window counts


@pytest.mark.asyncio
async def test_max_mean_aggregation_prefers_the_steady_track(ws):
    ws.camera("cam_01")
    ws.track("lucky", "cam_01", crops=[E[0], E[3], E[3], E[3]], t0=100.0)
    ws.track("steady", "cam_01", crops=[E[0], E[0], E[0], E[0]], t0=200.0, t1=210.0)
    res = await ws.retriever().search(plan(attrs=()))
    assert ids(res)[0] == "steady"


@pytest.mark.asyncio
async def test_peak_time_is_the_best_matching_crop(ws):
    ws.camera("cam_01")
    ws.track("t", "cam_01", t0=100.0, t1=110.0, crops=[E[3], E[3], E[0], E[3]], best_t=100.0)
    res = await ws.retriever().search(plan(attrs=()))
    assert res.candidates[0].track.best_t == 102.0  # third crop is the red one


@pytest.mark.asyncio
async def test_captions_boost_matching_tracks(ws):
    ws.camera("cam_01")
    ws.track("plain", "cam_01", crops=[E[2]], t0=100.0)
    ws.track("captioned", "cam_01", crops=[E[2]], t0=200.0, t1=210.0)
    ws.caption("captioned", "cam_01", "a red car driving through a gate")
    ws.caption("plain", "cam_01", "an empty road")
    res = await ws.retriever().search(plan(attrs=(), embed="a photo of a red car"))
    assert ids(res)[0] == "captioned" and "caption match" in res.candidates[0].why
    assert "captions" in res.layers


@pytest.mark.asyncio
async def test_scene_tiles_overlapping_a_track_add_support(ws):
    ws.camera("cam_01")
    ws.track("seen", "cam_01", crops=[E[2]], t0=100.0, t1=110.0)
    ws.track("unseen", "cam_01", crops=[E[2]], t0=300.0, t1=310.0)
    ws.scene("cam_01", 105.0, E[0])
    res = await ws.retriever().search(plan(attrs=()))
    assert ids(res)[0] == "seen" and any(w.startswith("scene") for w in res.candidates[0].why)


@pytest.mark.asyncio
async def test_cameras_without_tracks_fall_back_to_scene_tiles(ws):
    ws.camera("ready", "Gate")
    ws.camera("indexing", "Lobby")
    ws.track("t1", "ready", crops=[E[0]])
    for t in (50.0, 51.0, 52.0, 200.0):
        ws.scene("indexing", t, E[0])
    ws.scene("indexing", 90.0, E[1])
    res = await ws.retriever().search(plan(attrs=()))
    scene_ids = [i for i in ids(res) if i.startswith("scene:")]
    # adjacent tiles merged into one window; the blue-looking tile is returned too, but ranked last
    assert scene_ids == ["scene:indexing:50", "scene:indexing:200", "scene:indexing:90"]
    scores = {c.track.id: c.score for c in res.candidates}
    assert scores["scene:indexing:50"] > 0.9 > 0.1 > scores["scene:indexing:90"]
    win = next(c for c in res.candidates if c.track.id == "scene:indexing:50").track
    assert (win.t_start, win.t_end, win.cls) == (50.0, 52.0, "scene")
    assert "t1" in ids(res)
    assert any("Still indexing Lobby" in n for n in res.notes)
    assert {"crops", "scenes"} <= set(res.layers)


@pytest.mark.asyncio
async def test_frame_unit_searches_whole_frames_only(ws):
    ws.camera("cam_01")
    ws.track("t1", "cam_01", crops=[E[0]])
    ws.scene("cam_01", 10.0, E[0], tile="full")
    ws.scene("cam_01", 40.0, E[0], tile="tl")  # tiles are not whole frames
    res = await ws.retriever(unit="frame").search(plan(attrs=()))
    assert ids(res) == ["scene:cam_01:10"]
    assert not any("Still indexing" in n for n in res.notes)


@pytest.mark.asyncio
async def test_lexicon_expansion_finds_what_the_plain_wording_misses(ws):
    ws.camera("cam_01")
    ws.track("sedan_only", "cam_01", crops=[E[3]])
    p = plan(attrs=(), embed="a photo of a car")
    plain = await ws.retriever().search(p)
    expanded = await ws.retriever(expansion="lexicon").search(p)
    best = lambda r: max((c.score for c in r.candidates if c.track.id == "sedan_only"), default=0.0)  # noqa: E731
    assert best(expanded) > best(plain) + 0.4


@pytest.mark.asyncio
async def test_degenerate_inputs_return_notes_not_errors(ws):
    res = await ws.retriever().search(plan())
    assert res.candidates == [] and any("No matching camera" in n for n in res.notes)
    ws.camera("cam_01")
    none = await ws.retriever().search(QueryPlan(intent="describe"))
    assert none.candidates == [] and any("no object" in n for n in none.notes)
    res = await ws.retriever().search(plan())
    assert res.candidates == [] and any("Nothing has been indexed" in n for n in res.notes)
    two = QueryPlan(intent="list", targets=[Target(noun="car", embed_text="a car"), Target(noun="bus", embed_text="a bus")])
    assert any("first object" in n for n in (await ws.retriever().search(two)).notes)


@pytest.mark.asyncio
async def test_pool_limit_and_odd_camera_ids(ws):
    ws.camera("o'brien")
    for i in range(6):
        ws.track(f"t{i}", "o'brien", t0=100.0 + i * 20, t1=110.0 + i * 20)
    res = await ws.retriever(pool_limit=4).search(plan(), SearchScope(frozenset({"o'brien"})))
    assert len(res.candidates) == 4


def test_config_from_the_yaml_section():
    cfg = RetrievalConfig.from_cfg({"retrieval": {"unit": "frame", "attributes": False, "expansion": "lexicon",
                                                  "weights": {"image": 0.9}, "calibration": {"midpoint": 0.3},
                                                  "pool_limit": 7}})
    assert (cfg.unit, cfg.attributes, cfg.expansion, cfg.pool_limit) == ("frame", False, "lexicon", 7)
    assert cfg.weights["image"] == 0.9 and cfg.weights["attributes"] == 0.25 and cfg.calibration.midpoint == 0.3
    default = RetrievalConfig.from_cfg(None)
    assert default.unit == "track" and default.attributes is True and default.expansion == "off"


# ------------------------------------------------------- presence inside the window
def spaced(a, b, step=0.25):
    return [a + k * step for k in range(int((b - a) / step) + 1)]


@pytest.mark.asyncio
async def test_a_track_whose_span_overlaps_the_window_but_was_never_on_screen_in_it_is_not_present(ws):
    ws.camera("cam_01")
    # a long visit seen at the start and the end only (say it left the view and came back)
    ws.track("away", "cam_01", t0=0.0, t1=200.0, points=spaced(0, 10) + spaced(190, 200), crops=[E[0]])
    ws.track("here", "cam_01", t0=50.0, t1=130.0, points=spaced(50, 130), crops=[E[0]])
    window = TimeWindow(start=60.0, end=120.0)
    res = await ws.retriever().search(plan(attrs=()), SearchScope(window=window))
    assert ids(res) == ["here"]


@pytest.mark.asyncio
async def test_the_evidence_moment_is_inside_the_window_even_if_the_best_crop_is_elsewhere(ws):
    ws.camera("cam_01")
    ws.track("long", "cam_01", t0=0.0, t1=200.0, points=spaced(0, 200), crops=[E[0]] + [E[2]] * 3, best_t=0.0)
    # the best matching crop is the first one, sampled at t=0; the question is about 100-120 s
    res = await ws.retriever().search(plan(attrs=()), SearchScope(window=TimeWindow(start=100.0, end=120.0)))
    peak = res.candidates[0].track.best_t
    assert 100.0 <= peak <= 120.0
    unconstrained = await ws.retriever().search(plan(attrs=()))
    assert unconstrained.candidates[0].track.best_t == 0.0  # without a window the best crop is used as before


@pytest.mark.asyncio
async def test_time_of_day_windows_use_presence_too(ws):
    ws.camera("cam_01")
    base = 3600.0 * 5  # 05:00 UTC
    ws.track("t1", "cam_01", t0=base, t1=base + 600, points=spaced(base, base + 10) + spaced(base + 590, base + 600),
             crops=[E[0]])
    present = TimeWindow(tod_after="05:00", tod_before="05:01")
    absent = TimeWindow(tod_after="05:03", tod_before="05:05")  # the span covers it, the track was not on screen
    assert ids(await ws.retriever().search(plan(attrs=()), SearchScope(window=present))) == ["t1"]
    assert ids(await ws.retriever().search(plan(attrs=()), SearchScope(window=absent))) == []


@pytest.mark.asyncio
async def test_tracks_without_stored_points_keep_the_span_based_decision(ws):
    ws.camera("cam_01")
    ws.track("nopoints", "cam_01", t0=0.0, t1=200.0, crops=[E[0]])
    res = await ws.retriever().search(plan(attrs=()), SearchScope(window=TimeWindow(start=60.0, end=120.0)))
    assert ids(res) == ["nopoints"]


@pytest.mark.asyncio
async def test_a_single_flicker_inside_the_window_is_not_presence(ws):
    ws.camera("cam_01")
    ws.track("flicker", "cam_01", t0=0.0, t1=200.0, points=spaced(0, 50) + [100.0] + spaced(150, 200), crops=[E[0]])
    res = await ws.retriever().search(plan(attrs=()), SearchScope(window=TimeWindow(start=90.0, end=110.0)))
    assert ids(res) == []
