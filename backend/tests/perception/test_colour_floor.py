"""A colour the model is unsure about is left unknown, not guessed."""
import json

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")
pytest.importorskip("lancedb")

from contracts.models import TrackAttrs  # noqa: E402

from evora.perception import colourmodel as cm  # noqa: E402
from evora.perception import l2  # noqa: E402

from .test_pipeline import _settings  # noqa: E402,F401
from .test_pipeline import env as pipeline_env  # noqa: E402,F401


def fake_detail(conf_upper, conf_lower):
    def predict_detail(kind, crop, gains=None, mask=None, params=None):
        return {"upper": cm.ColourResult("red", conf_upper, "dark red", "dark"),
                "lower": cm.ColourResult("blue", conf_lower, "blue", None)}
    return predict_detail


@pytest.fixture
def images():
    return [np.zeros((60, 30, 3), dtype=np.uint8)] * 2


def run(images, monkeypatch, up, lo, floor):
    monkeypatch.setattr(cm, "predict_detail", fake_detail(up, lo))
    attrs, extras = TrackAttrs(), {}
    l2._v2_colours(attrs, extras, "person", images, None, None, floor)
    return attrs, extras


def test_confident_colours_are_kept_with_their_shade_names(images, monkeypatch):
    attrs, extras = run(images, monkeypatch, 0.9, 0.85, 0.6)
    assert (attrs.upper_color, attrs.lower_color, attrs.color) == ("red", "blue", "red")
    assert extras["upper_color_name"] == "dark red" and "colour_unsure" not in extras


def test_an_unsure_slot_is_left_unknown_while_the_sure_one_stays(images, monkeypatch):
    attrs, extras = run(images, monkeypatch, 0.9, 0.45, 0.6)
    assert attrs.upper_color == "red" and attrs.lower_color is None
    assert extras["colour_unsure"] == ["lower"] and extras["lower_color_conf"] == pytest.approx(0.45)


def test_nothing_is_claimed_when_everything_is_unsure(images, monkeypatch):
    attrs, extras = run(images, monkeypatch, 0.5, 0.5, 0.6)
    assert attrs.upper_color is None and attrs.color is None and attrs.color_conf is None and attrs.lower_color is None
    assert sorted(extras["colour_unsure"]) == ["lower", "upper"]


def test_a_floor_of_zero_disables_the_check(images, monkeypatch):
    attrs, _ = run(images, monkeypatch, 0.05, 0.05, 0.0)
    assert attrs.upper_color == "red" and attrs.lower_color == "blue"


def test_the_floor_reaches_the_stored_attributes(pipeline_env, monkeypatch):  # noqa: F811
    from evora.perception import pipeline

    monkeypatch.setattr(l2, "get_segmenter", lambda device="auto": None)
    monkeypatch.setattr(cm, "predict_detail", fake_detail(0.9, 0.3))
    ws, db, cam = pipeline_env
    pipeline.ingest(cam, "cpu", {"L0", "L1", "L2"}, lambda e: None, ws=ws,
                    settings=_settings(colour_engine="v2", colour_min_conf=0.6))
    with db.read() as c:
        stored = [json.loads(r["attrs"]) for r in c.execute("SELECT attrs FROM tracks")]
    assert stored and all(a.get("upper_color") == "red" and "lower_color" not in a for a in stored)
    assert all(a["colour_unsure"] == ["lower"] for a in stored)
