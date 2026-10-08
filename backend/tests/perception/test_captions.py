"""L3 captions with a scripted vision client and a fake text embedder: no model, GPU or network needed."""
import hashlib

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")
pytest.importorskip("lancedb")

from evora.core.vectors import open_store  # noqa: E402
from evora.perception import captions, pipeline, vision  # noqa: E402

from .test_pipeline import FakeEmbedder, FakeReid, ScriptedTracker, _settings  # noqa: E402,F401
from .test_pipeline import env as pipeline_env  # noqa: E402,F401


class FakeText:
    def embed(self, texts):
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, int(hashlib.md5(t.encode()).hexdigest(), 16) % 384] = 1.0
        return out


class ScriptedVision:
    def __init__(self, replies=None, default="a person in a red jacket and dark trousers"):
        self.replies, self.default, self.prompts, self.tokens = list(replies or []), default, [], []

    def describe(self, image_jpeg, prompt, *, max_tokens=64):
        self.prompts.append(prompt)
        self.tokens.append(max_tokens)
        return self.replies.pop(0) if self.replies else self.default


@pytest.fixture
def indexed(pipeline_env):  # noqa: F811
    ws, db, cam = pipeline_env
    pipeline.ingest(cam, "cpu", {"L0", "L1"}, lambda e: None, ws=ws, settings=_settings())
    return ws, db, cam


def run(indexed, vision_client, settings=None, embedder=None, events=None):
    ws, db, cam = indexed
    store = open_store(ws.vectors_dir)
    n = captions.run_l3(cam, ws, db, store, settings or _settings(), vision_client, embedder or FakeText(),
                        (events.append if events is not None else lambda e: None))
    return n, store.open_table("captions").to_arrow().to_pylist() if "captions" in store.list_tables().tables else []


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("a person in a red jacket", "a person in a red jacket"),
        ('"A man in a blue coat."', "A person in a blue coat."),
        ("<think>hmm the image shows</think>A woman wearing a black dress", "A person wearing a black dress"),
        ("  a   person\nin grey  ", "a person in grey"),
        ("a boy and a girl in green", "a person and a person in green"),
        ("a person in a black top carrying no bag", "a person in a black top"),
        ("a person in a red coat, holding nothing in their hands.", "a person in a red coat."),
        ("a person in a grey coat with no hat and dark trousers", "a person in a grey coat"),
        ("None", None), ("", None), (None, None), ("sorry, I cannot help", None), ("red", None),
    ],
)
def test_clean_caption(raw, expected):
    assert captions.clean_caption(raw) == expected


def test_long_captions_are_cut_at_a_word_boundary():
    out = captions.clean_caption("a person " + "in a very colourful jacket " * 20)
    assert out is not None and len(out) <= captions.MAX_CHARS + 1 and out.endswith(".")


def test_captions_are_written_with_contract_columns_and_vectors(indexed):
    n, rows = run(indexed, ScriptedVision(default="A man in a red jacket and dark trousers"))
    assert n == 2 and len(rows) == 2
    assert set(rows[0]) == {"vector", "text", "camera_id", "t", "track_id"}
    assert len(rows[0]["vector"]) == 384
    assert all(r["text"] == "A person in a red jacket and dark trousers" for r in rows)       # gendered noun rewritten
    ws, db, cam = indexed
    with db.read() as c:
        track_ids = {r["id"] for r in c.execute("SELECT id FROM tracks")}
    assert {r["track_id"] for r in rows} == track_ids and all(r["camera_id"] == cam.id for r in rows)


def test_prompts_depend_on_the_class_and_the_token_budget_allows_for_thinking_models(indexed):
    ws, db, cam = indexed
    with db.write() as c:
        c.execute("UPDATE tracks SET cls='car' WHERE id=(SELECT id FROM tracks ORDER BY id LIMIT 1)")
    v = ScriptedVision(default="a white SUV standing in the road")
    run(indexed, v)
    assert sum("vehicle" in p for p in v.prompts) == 1 and sum("person's clothing" in p for p in v.prompts) == 1
    assert all(t >= 256 for t in v.tokens)
    person_prompt = next(p for p in v.prompts if "clothing" in p)
    vehicle_prompt = next(p for p in v.prompts if "vehicle" in p)
    assert "Do not describe the face" in person_prompt and "identity" in person_prompt     # no faces, no identity
    assert "licence plate" in vehicle_prompt


def test_rerun_replaces_instead_of_duplicating(indexed):
    run(indexed, ScriptedVision())
    n, rows = run(indexed, ScriptedVision(default="a person in blue"))
    assert n == len(rows) == 2 and all(r["text"] == "a person in blue" for r in rows)


def test_track_cap_is_respected(indexed):
    n, rows = run(indexed, ScriptedVision(), settings=_settings(l3_max_tracks=1))
    assert n == len(rows) == 1


def test_one_bad_reply_does_not_stop_the_others(indexed):
    n, rows = run(indexed, ScriptedVision(replies=[None], default="a person in green trousers"))
    assert n == 1 and len(rows) == 1


def test_layer_is_not_finished_without_a_vision_model_or_when_every_call_fails(indexed):
    events = []
    assert run(indexed, None, events=events)[0] is None
    assert run(indexed, ScriptedVision(default=None), events=events)[0] is None
    assert not [e for e in events if e.progress >= 1.0]


def test_time_budget_stops_the_layer_but_keeps_what_exists(indexed):
    class Slow(ScriptedVision):
        def describe(self, *a, **k):
            import time

            time.sleep(0.4)
            return super().describe(*a, **k)

    n, rows = run(indexed, Slow(), settings=_settings(l3_budget_s=0.2))
    assert n == 1 and len(rows) == 1                      # the first track starts inside the budget, the second does not


def test_ingest_runs_l3_only_with_a_registered_client_and_reports_it_finished(indexed, monkeypatch):
    ws, db, cam = indexed
    monkeypatch.setattr(captions, "FastTextEmbedder", FakeText)
    events = []
    vision.register(ScriptedVision())
    try:
        pipeline.ingest(cam, "cpu", {"L3"}, events.append, ws=ws, settings=_settings())
    finally:
        vision.register(None)
    assert [e.layer for e in events if e.progress >= 1.0] == ["L3"]
    assert open_store(ws.vectors_dir).open_table("captions").count_rows() == 2
