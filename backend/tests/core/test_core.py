import threading

import pytest

from evora.core import db as dbmod
from evora.core import workspace as wsmod
from evora.core.vectors import DimensionError, dims_from_meta, ensure_tables, open_store


@pytest.fixture()
def root(tmp_path):
    yield tmp_path / "workspaces"
    dbmod.close_all()


def test_workspace_create_list_activate(root):
    ws = wsmod.create("Own Campus", root)
    assert ws.slug == "own-campus" and ws.vectors_dir.is_dir() and ws.media_dir.is_dir()
    wsmod.create("Mall", root)
    assert [w.slug for w in wsmod.list_all(root)] == ["mall", "own-campus"]
    assert wsmod.active(root) is None
    wsmod.activate("mall", root)
    assert wsmod.active(root).slug == "mall"


@pytest.mark.parametrize("bad", ["..", "  ", "!!!"])
def test_workspace_rejects_empty_names(root, bad):
    with pytest.raises(wsmod.WorkspaceError):
        wsmod.create(bad, root)


def test_traversal_name_is_confined(root):
    ws = wsmod.create("../escape", root)
    assert ws.root.resolve().parent == root.resolve()


def test_unknown_workspace(root):
    root.mkdir(parents=True)
    with pytest.raises(wsmod.WorkspaceError):
        wsmod.get("nope", root)


def test_db_migrates_and_is_idempotent(root):
    ws = wsmod.create("a", root)
    d = dbmod.open_db(ws.db_path)
    assert d.get_meta("schema_version") == "1.1"
    dbmod.close_all()
    d2 = dbmod.open_db(ws.db_path)
    assert d2.get_meta("schema_version") == "1.1"
    with d2.read() as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_one_writer_per_file(root):
    ws = wsmod.create("a", root)
    assert dbmod.open_db(ws.db_path) is dbmod.open_db(ws.db_path)


def test_write_rolls_back_on_error(root):
    d = dbmod.open_db(wsmod.create("a", root).db_path)
    with pytest.raises(RuntimeError):
        with d.write() as c:
            c.execute("INSERT INTO meta(key,value) VALUES('k','v')")
            raise RuntimeError("boom")
    assert d.get_meta("k") is None
    d.set_meta("k", "v2")
    assert d.get_meta("k") == "v2"


def test_concurrent_writers_serialise(root):
    d = dbmod.open_db(wsmod.create("a", root).db_path)
    errors = []

    def work(i):
        try:
            for j in range(20):
                d.set_meta(f"k{i}_{j}", "x")
        except Exception as e:  # noqa: BLE001 - collected and asserted below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    with d.read() as c:
        assert c.execute("SELECT count(*) FROM meta WHERE key LIKE 'k%'").fetchone()[0] == 120


def test_incompatible_schema_version_refused(root):
    ws = wsmod.create("a", root)
    d = dbmod.open_db(ws.db_path)
    d.set_meta("schema_version", "2")
    dbmod.close_all()
    with pytest.raises(RuntimeError):
        dbmod.open_db(ws.db_path)


def test_vector_tables_follow_meta_dims(root):
    ws = wsmod.create("a", root)
    d = dbmod.open_db(ws.db_path)
    store = open_store(ws.vectors_dir)
    # only the text-dim tables are creatable before image dims are known
    assert set(ensure_tables(store, dims_from_meta(d))) == {"captions", "aliases"}
    d.set_meta("embed_dim_image", "768")
    d.set_meta("embed_dim_reid", "512")
    assert set(ensure_tables(store, dims_from_meta(d))) == {"crops", "scenes", "reid"}
    assert ensure_tables(store, dims_from_meta(d)) == []
    assert store.open_table("crops").schema.field("vector").type.list_size == 768


def test_vector_dim_mismatch_refused(root):
    ws = wsmod.create("a", root)
    store = open_store(ws.vectors_dir)
    ensure_tables(store, {"embed_dim_image": 768})
    with pytest.raises(DimensionError):
        ensure_tables(store, {"embed_dim_image": 512})


def test_alias_roundtrip_and_nullable_caption_track(root):
    ws = wsmod.create("a", root)
    store = open_store(ws.vectors_dir)
    ensure_tables(store, {"embed_dim_text": 384})
    vec = [0.0] * 384
    vec[0] = 1.0
    store.open_table("aliases").add([{"vector": vec, "fact_id": "mf_1", "alias": "main gate"}])
    store.open_table("captions").add(
        [{"vector": vec, "text": "red car", "camera_id": "cam_01", "t": 1.0, "track_id": None}]
    )
    hit = store.open_table("aliases").search(vec).limit(1).to_list()[0]
    assert hit["fact_id"] == "mf_1"
