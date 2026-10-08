"""P1.11: ask, answer one clarification, restart everything, ask again: nothing is asked twice."""
import pytest
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.core.db import close_all
from tests.api.fakes import FakeGateway, parse_sse
from tests.memory.conftest import FakeEmbedder

LINE = {"id": "c", "camera_id": "cam_01", "kind": "line", "points": [[0.12, 0.70], [0.88, 0.66]]}
PARAPHRASES = ("did a car pass the front gate?", "was there a car at the gate at the entrance?")


def start(tmp_path, monkeypatch, *, gateway=None):
    monkeypatch.setenv("evora_WORKSPACE", "e2e")
    return TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=gateway or FakeGateway(), embedder=FakeEmbedder()))


def ask(client, text):
    return parse_sse(client.post("/api/query", json={"text": text, "session_id": "s"}).text)


@pytest.fixture()
def footage(sample_mp4, sample_mp4_b):
    return sample_mp4, sample_mp4_b


def upload(client, footage):
    for path, name in zip(footage, ("north.mp4", "south.mp4"), strict=True):
        with open(path, "rb") as fh:
            client.post("/api/cameras", files=[("files", (name, fh))])


def test_clarify_once_survives_a_restart_and_paraphrases(tmp_path, monkeypatch, footage):
    first = start(tmp_path, monkeypatch)
    upload(first, footage)

    events = ask(first, "did a car pass the main gate?")
    assert [n for n, _ in events] == ["plan", "clarify"], "first time: exactly one question"
    qid = events[-1][1]["query_id"]
    resumed = parse_sse(first.post("/api/clarify", json={"query_id": qid, "camera_id": "cam_01", "zone": LINE}).text)
    assert [n for n, _ in resumed][-1] == "done" and "clarify" not in [n for n, _ in resumed]

    # the process goes away: every in-memory object is rebuilt from the workspace folder
    first.close()
    close_all()
    second = start(tmp_path, monkeypatch)

    clarifications = 0
    for text in ("did a car pass the main gate?", *PARAPHRASES):
        got = ask(second, text)
        clarifications += "clarify" in [n for n, _ in got]
        assert [n for n, _ in got][-1] == "done"
    assert clarifications == 0

    fact = second.get("/api/memory").json()[0]
    assert fact["binding"]["camera_id"] == "cam_01" and fact["binding"]["zone_id"].startswith("z_")
    assert set(fact["inferred_aliases"]) == {"front gate", "gate at entrance"}, "paraphrases were learned, not asked"
    zones = second.get("/api/zones", params={"camera_id": "cam_01"}).json()
    assert [(z["id"], z["kind"]) for z in zones] == [(fact["binding"]["zone_id"], "line")], "the drawn line survived the restart"


def test_without_alias_embeddings_a_paraphrase_asks_again(tmp_path, monkeypatch, footage):
    """The C4 ablation: exact aliases only, so 'front gate' is a new question."""
    import evora.core.config as config

    real = config.load_config

    def exact_only(profile=None):
        cfg = real(profile)
        cfg["memory"]["alias_embed"] = False
        return cfg

    import evora.api.app as appmod

    monkeypatch.setattr(appmod, "load_config", exact_only)
    client = start(tmp_path, monkeypatch)
    upload(client, footage)
    qid = ask(client, "did a car pass the main gate?")[-1][1]["query_id"]
    client.post("/api/clarify", json={"query_id": qid, "camera_id": "cam_01"})
    assert "clarify" not in [n for n, _ in ask(client, "did a car pass the main gate?")]
    assert "clarify" in [n for n, _ in ask(client, PARAPHRASES[0])]


def test_a_grey_band_paraphrase_needs_the_language_model_to_agree(tmp_path, monkeypatch, footage):
    for agrees, expect_clarify in ((True, False), (False, True)):
        client = start(tmp_path / str(agrees), monkeypatch, gateway=FakeGateway(same=agrees))
        upload(client, footage)
        qid = ask(client, "did a car pass the main gate?")[-1][1]["query_id"]
        client.post("/api/clarify", json={"query_id": qid, "camera_id": "cam_01"})
        got = [n for n, _ in ask(client, PARAPHRASES[1])]
        assert ("clarify" in got) is expect_clarify
        client.close()
        close_all()
