import pytest
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.evidence import audit
from tests.memory.conftest import FakeEmbedder


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "memroutes")
    return TestClient(create_app(workspaces_root=tmp_path / "ws", embedder=FakeEmbedder()))


def add(client, canonical="main gate", **extra):
    return client.post("/api/memory", json={"kind": "place", "canonical": canonical, "binding": {"camera_id": "cam_01"}, **extra})


def test_empty_ledger_then_add_and_list(client):
    assert client.get("/api/memory").json() == []
    fact = add(client, aliases=["front gate"]).json()
    assert fact["id"].startswith("mf_") and fact["aliases"] == ["front gate"] and fact["source"] == "statement"
    assert [f["id"] for f in client.get("/api/memory").json()] == [fact["id"]]
    assert client.get("/api/memory", params={"kind": "object"}).json() == []


def test_add_validation(client):
    assert client.post("/api/memory", json={"kind": "animal", "canonical": "x"}).status_code == 422
    assert client.post("/api/memory", json={"kind": "place", "canonical": ""}).status_code == 422
    assert client.post("/api/memory", json={"kind": "place", "canonical": "the"}).status_code == 422


def test_rename_and_edit_aliases_in_place(client):
    fid = add(client).json()["id"]
    body = client.patch(f"/api/memory/{fid}", json={"canonical": "front gate", "aliases": ["gate one"]}).json()
    assert body["id"] == fid and body["canonical"] == "front gate" and body["aliases"] == ["gate one"]


def test_changing_the_binding_is_a_correction_and_is_audited(client):
    old = add(client).json()
    new = client.patch(f"/api/memory/{old['id']}", json={"binding": {"camera_id": "cam_03"}}).json()
    assert new["id"] != old["id"] and new["source"] == "correction" and new["binding"] == {"camera_id": "cam_03"}
    assert [f["id"] for f in client.get("/api/memory").json()] == [new["id"]]
    assert {f["id"] for f in client.get("/api/memory", params={"include_superseded": True}).json()} == {old["id"], new["id"]}
    assert client.patch(f"/api/memory/{old['id']}", json={"canonical": "x"}).status_code == 409
    assert audit.entries(client.app.state.ctx.db, "memory_correct")[0]["detail"] == {"old": old["id"], "new": new["id"]}


def test_delete_returns_remaining_and_is_audited(client):
    a, b = add(client, "main gate").json(), add(client, "loading dock").json()
    remaining = client.delete(f"/api/memory/{a['id']}").json()
    assert [f["id"] for f in remaining] == [b["id"]]
    assert audit.entries(client.app.state.ctx.db, "memory_delete")[0]["detail"] == {"fact_id": a["id"]}


def test_unknown_fact_is_404(client):
    assert client.patch("/api/memory/mf_00000000", json={"canonical": "x"}).status_code == 404
    assert client.delete("/api/memory/mf_00000000").status_code == 404


def test_facts_survive_an_app_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "persist")
    first = TestClient(create_app(workspaces_root=tmp_path / "ws", embedder=FakeEmbedder()))
    fid = add(first, aliases=["front gate"]).json()["id"]
    from evora.core.db import close_all

    close_all()
    second = TestClient(create_app(workspaces_root=tmp_path / "ws", embedder=FakeEmbedder()))
    facts = second.get("/api/memory").json()
    assert [f["id"] for f in facts] == [fid] and facts[0]["aliases"] == ["front gate"]
