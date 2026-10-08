"""The runbooks may only mention commands, routes and files that exist, so they cannot drift from the code."""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]


def make_targets() -> set[str]:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Za-z][\w-]*):", text, flags=re.MULTILINE))


@pytest.fixture(scope="module")
def routes(tmp_path_factory):
    import os

    from fastapi.testclient import TestClient

    from evora.api.app import create_app
    from evora.core.db import close_all

    os.environ["evora_WORKSPACE"] = "docs"
    try:
        client = TestClient(create_app(workspaces_root=tmp_path_factory.mktemp("docs_ws"), mock=True))
        paths = set(client.get("/openapi.json").json()["paths"])
        client.close()
    finally:
        close_all()
        os.environ.pop("evora_WORKSPACE", None)
    return {re.sub(r"\{[^}]*\}", "{}", p) for p in paths} | {"/api/events", "/api/cameras/{}/live.mjpg"}


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_make_target_mentioned_exists(doc):
    mentioned = set(re.findall(r"`make ([a-z][\w-]*)", doc.read_text(encoding="utf-8")))
    assert mentioned <= make_targets(), f"{doc.name} mentions missing targets: {mentioned - make_targets()}"


ROUTE = re.compile(r"`(?:GET |POST |PATCH |DELETE )?(/api/[^\s`?]+)")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_api_route_mentioned_exists(doc, routes):
    mentioned = {re.sub(r"\{[^}]*\}", "{}", m.rstrip(".,;:)")) for m in ROUTE.findall(doc.read_text(encoding="utf-8"))}
    missing = {m for m in mentioned if m not in routes}
    assert not missing, f"{doc.name} mentions routes that do not exist: {sorted(missing)}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_file_path_mentioned_exists(doc):
    text = doc.read_text(encoding="utf-8")
    paths = set(re.findall(r"`((?:docs|eval|scripts|config|contracts)/[\w./-]+\.\w+)`", text))
    missing = {p for p in paths if not (ROOT / p).exists()}
    assert not missing, f"{doc.name} mentions files that do not exist: {sorted(missing)}"


def test_the_judge_sim_skeleton_is_valid_yaml_with_twenty_slots_and_the_agreed_quotas():
    items = yaml.safe_load((ROOT / "docs" / "judge_sim_skeleton.yaml").read_text(encoding="utf-8"))
    assert len(items) == 20 and len({i["id"] for i in items}) == 20
    purposes = [i["purpose"] for i in items]
    assert purposes.count("negative") >= 4 and purposes.count("first_time_place") == 4 and purposes.count("path") == 3
    assert all(i["split"] == "judge_sim" for i in items)


def test_the_real_judge_sim_queries_can_never_be_committed():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/data/" in ignore or "data/" in ignore.splitlines()
