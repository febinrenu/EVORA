"""The scene fallback: a camera where the detector found nothing in the asked window may still answer, weakly,
when that window's best scene tile stands far above the camera's usual similarity. Off by default."""
import numpy as np
import pytest
from contracts.models import Evidence, QueryPlan, Target, TimeWindow

from evora.query.fuse import Calibration
from evora.query.retrieve import RetrievalConfig, Retriever, SearchScope
from evora.query.router import _is_partial
from tests.query.ws_helpers import E, ToyEmbedder, Workspace

WINDOW = TimeWindow(start=1200.0, end=1260.0)


def car_plan(intent="exists", action="any"):
    return QueryPlan(intent=intent, targets=[Target(noun="car", cls=["car"], embed_text="a photo of a car")],
                     action=action, camera_ids=["cam_01"])


def background(seed: int, n: int) -> list[np.ndarray]:
    """Scene tiles that look a little, and variably, like the query (axis 2 is the generic car direction)."""
    rng = np.random.default_rng(seed)
    base = np.array([1.0, 1.0, 0.3, 1.0], dtype=np.float32)
    return [(v := base + rng.normal(0, 0.04, 4).astype(np.float32)) / np.linalg.norm(v) for _ in range(n)]


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.camera("cam_01", "Gate", t0=1000.0)
    w.track("t_old", "cam_01", t0=1010.0, t1=1020.0, crops=(E[2],))  # a car long before the window: a track camera
    for i, v in enumerate(background(7, 120)):
        w.scene("cam_01", 1000.0 + i * 2.5, v)
    yield w
    w.close()


async def search(ws, plan, window=WINDOW, **cfg):
    retriever = Retriever(ws.db, ws.store, ToyEmbedder(), RetrievalConfig(calibration=Calibration(0.5, 0.2), **cfg))
    return await retriever.search(plan, SearchScope(frozenset({"cam_01"}), window))


def standout(ws, t=1230.0):
    ws.scene("cam_01", t, E[2] * 0.98 + E[0] * 0.02)  # a tile that looks very much like the query


@pytest.mark.asyncio
async def test_a_clear_peak_in_the_window_gives_one_weak_candidate(ws):
    standout(ws)
    out = await search(ws, car_plan(), scene_fallback=True)
    assert len(out.candidates) == 1
    cand = out.candidates[0]
    assert cand.track.cls == "scene" and cand.track.best_t == 1230.0 and cand.score == pytest.approx(0.42)
    assert cand.why[0].startswith("scene fallback z ") and float(cand.why[0].split()[-1]) >= 3.5
    assert any("No car was detected on Gate" in n for n in out.notes)


@pytest.mark.asyncio
async def test_without_an_outlier_the_answer_stays_nothing_there(ws):
    out = await search(ws, car_plan(), scene_fallback=True)
    assert out.candidates == [], "an ordinary-looking window is not evidence: the negative is kept"


@pytest.mark.asyncio
async def test_it_is_off_by_default(ws):
    standout(ws)
    assert (await search(ws, car_plan())).candidates == []


@pytest.mark.asyncio
@pytest.mark.parametrize("intent,action", [("count", "any"), ("path", "any"), ("exists", "pass_through"), ("exists", "enter")])
async def test_never_for_counts_paths_or_movements_that_need_a_track(ws, intent, action):
    standout(ws)
    assert (await search(ws, car_plan(intent, action), scene_fallback=True)).candidates == []


@pytest.mark.asyncio
async def test_a_camera_with_a_real_track_in_the_window_does_not_fall_back(ws):
    standout(ws)
    ws.track("t_now", "cam_01", t0=1220.0, t1=1240.0, crops=(E[2],), bbox=(0.2, 0.2, 0.4, 0.6))
    out = await search(ws, car_plan(), scene_fallback=True)
    assert [c.track.id for c in out.candidates] == ["t_now"]


@pytest.mark.asyncio
async def test_no_time_window_means_no_fallback(ws):
    standout(ws)
    out = await search(ws, car_plan(), window=None, scene_fallback=True)
    assert not any(c.track.cls == "scene" for c in out.candidates), "the old track may answer; the scenes may not"


@pytest.mark.asyncio
async def test_too_few_tiles_to_know_the_usual_level(ws):
    standout(ws)
    assert (await search(ws, car_plan(), scene_fallback=True, scene_fallback_min_tiles=1000)).candidates == []


@pytest.mark.asyncio
async def test_a_higher_bar_turns_a_weak_outlier_away(ws):
    standout(ws)
    assert (await search(ws, car_plan(), scene_fallback=True, scene_fallback_z=1e6)).candidates == []


def test_config_reads_the_switches():
    cfg = RetrievalConfig.from_cfg({"retrieval": {"scene_fallback": True, "scene_fallback_z": 4.0}})
    assert cfg.scene_fallback is True and cfg.scene_fallback_z == 4.0
    assert RetrievalConfig.from_cfg({}).scene_fallback is False


def test_an_answer_from_the_fallback_is_always_partial():
    ev = Evidence(id="ev_1", camera_id="cam_01", camera_name="Gate", t_start=1229.0, t_end=1231.0, t_peak=1230.0,
                  offset_s=230.0, thumb_url="/t", clip_url="/c", score=0.42, why=["scene fallback z 5.1", "scene siglip 0.98"])
    assert _is_partial([], car_plan(), [ev]) is True
    assert _is_partial([], car_plan(), [ev.model_copy(update={"why": ["siglip 0.31"]})]) is False
