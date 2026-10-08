import numpy as np
import pytest

from evora.memory import embedder as emb
from evora.memory.kb import FactNotFound, KBError, KnowledgeBase, normalize


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Main Gate", "main gate"), ("the main gate", "main gate"), ("At the Main Gate!", "main gate"),
        ("  in   my  car ", "my car"), ("a car", "car"), ("on the", ""), ("cam_02", "cam 02"), ("", ""), ("the", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_create_get_list_roundtrip(make_env):
    env = make_env()
    f = env.kb.create("place", "Main Gate", {"camera_id": "cam_01"}, "clarification", ["front gate"])
    assert env.kb.get(f.id) == f and f.aliases == ["front gate"] and f.use_count == 0
    assert [x.id for x in env.kb.list()] == [f.id]
    assert env.kb.list("object") == []


def test_create_validates(make_env):
    env = make_env()
    with pytest.raises(KBError):
        env.kb.create("animal", "x", {})
    with pytest.raises(KBError):
        env.kb.create("place", "the", {})


def test_add_alias_dedupes_on_normalised_form(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {})
    assert env.kb.add_alias(f.id, "Front Gate") is True
    assert env.kb.add_alias(f.id, "the front gate!") is False
    assert env.kb.add_alias(f.id, "THE MAIN GATE") is False
    assert env.kb.get(f.id).aliases == ["Front Gate"]


def test_touch_counts_use(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {})
    env.kb.touch(f.id)
    env.kb.touch(f.id)
    got = env.kb.get(f.id)
    assert got.use_count == 2 and got.last_used_at is not None


def test_update_reindexes_aliases(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {"camera_id": "cam_01"})
    env.kb.update(f.id, canonical="loading dock")
    assert env.kb.search("loading dock", "place")[0].similarity == pytest.approx(1.0, abs=1e-4)
    assert all(h.fact.id != f.id or h.phrase != "main gate" for h in env.kb.search("main gate", "place"))


def test_delete_removes_fact_vectors_and_unlinks_zone(make_env):
    env = make_env()
    cam = env.camera("Gate")
    f = env.kb.create("place", "main gate", {"camera_id": cam.id})
    with env.db.write() as c:
        c.execute("INSERT INTO zones(id,camera_id,kind,fact_id,created_at) VALUES('z1',?,'frame',?,1)", (cam.id, f.id))
    env.kb.delete(f.id)
    with pytest.raises(FactNotFound):
        env.kb.get(f.id)
    assert env.kb.search("main gate", "place") == []
    with env.db.read() as c:
        assert c.execute("SELECT fact_id FROM zones WHERE id='z1'").fetchone()[0] is None
    with pytest.raises(FactNotFound):
        env.kb.delete(f.id)


def test_supersede_keeps_name_and_aliases_and_hides_the_old_fact(make_env):
    env = make_env()
    old = env.kb.create("place", "main gate", {"camera_id": "cam_01"}, "clarification", ["front gate"])
    new = env.kb.supersede(old.id, {"camera_id": "cam_03"}, "correction")
    assert (new.canonical, new.aliases) == ("main gate", ["front gate"])
    assert (new.binding, new.source) == ({"camera_id": "cam_03"}, "correction")
    assert env.kb.get(old.id).superseded_by == new.id
    assert [f.id for f in env.kb.list()] == [new.id]
    assert [f.id for f in env.kb.list(include_superseded=True)] == [old.id, new.id]
    assert {h.fact.id for h in env.kb.search("main gate", "place")} == {new.id}
    with pytest.raises(KBError):
        env.kb.supersede(old.id, {}, "correction")


def test_changing_the_embedder_rebuilds_the_alias_index(make_env):
    env = make_env()
    f = env.kb.create("place", "main gate", {}, aliases=["front gate"])
    env.kb.search("main gate", "place")
    env.kb.embedder = emb.HashingEmbedder()  # a different vector space
    rebuilt = KnowledgeBase(env.db, env.store, env.kb.embedder)
    assert env.db.get_meta("embed_model_text") == "hash-ngram-384"
    hits = rebuilt.search("main gate", "place")
    assert hits and hits[0].fact.id == f.id and hits[0].similarity == pytest.approx(1.0, abs=1e-4)


def test_hashing_embedder_is_deterministic_and_typo_tolerant():
    e = emb.HashingEmbedder()
    a, b, c = e.embed(["main gate", "main gate", "mian gate"])
    d = e.embed(["loading dock"])[0]
    assert np.allclose(a, b) and float(np.linalg.norm(a)) == pytest.approx(1.0)
    assert float(a @ c) > float(a @ d) + 0.2


def test_default_embedder_falls_back_without_the_model(tmp_path):
    assert emb.default_embedder(tmp_path).name in {"hash-ngram-384", f"fastembed:{emb.MODEL_ID}"}
