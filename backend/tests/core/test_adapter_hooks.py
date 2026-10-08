import sys
import types

import pytest
from contracts.models import Zone

from evora.core import cameras as cams
from evora.core import perception_adapter as adapter
from evora.core import workspace as wsmod
from evora.core.bus import Bus
from evora.core.db import open_db
from evora.core.zone_service import ZoneService

ZONE = Zone(id="z1", camera_id="cam_01", kind="line", points=[(0.1, 0.5), (0.9, 0.5)])


def install(monkeypatch, name, **functions):
    module = types.ModuleType(name)
    for k, v in functions.items():
        setattr(module, k, v)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(adapter, "MODULES", (name,))


def test_the_real_functions_are_found_where_the_other_members_put_them():
    assert adapter._find("recompute_events").__module__ == "evora.perception.events"
    assert adapter._find("link_global_ids").__module__ == "evora.reid.associate"
    assert adapter._find("detect_clock") is not None and adapter._find("blur_faces") is not None


def test_recompute_receives_the_apps_database_when_it_accepts_one(monkeypatch):
    seen = {}

    def real(camera_id, zones, *, db=None, settings=None):
        seen.update(camera_id=camera_id, zones=zones, db=db)
        return 7

    install(monkeypatch, "fake_events", recompute_events=real)
    assert adapter.recompute_events("cam_01", [ZONE], db="THE-DB") == 7
    assert seen == {"camera_id": "cam_01", "zones": [ZONE], "db": "THE-DB"}


def test_recompute_works_with_a_function_that_takes_no_database(monkeypatch):
    install(monkeypatch, "fake_events", recompute_events=lambda camera_id, zones: 3)
    assert adapter.recompute_events("cam_01", [ZONE], db="ignored") == 3


def test_recompute_failures_and_absence_mean_unavailable(monkeypatch):
    def boom(camera_id, zones):
        raise RuntimeError("no tracks")

    install(monkeypatch, "fake_events", recompute_events=boom)
    assert adapter.recompute_events("cam_01", [ZONE]) is None
    install(monkeypatch, "fake_empty")
    assert adapter.recompute_events("cam_01", [ZONE]) is None


def test_link_identities_gets_the_workspace(monkeypatch):
    seen = {}

    def real(workspace=None, settings=None):
        seen["workspace"] = workspace
        return 4

    install(monkeypatch, "fake_reid", link_global_ids=real)
    assert adapter.link_identities("THE-WS") == 4 and seen["workspace"] == "THE-WS"


def test_link_identities_old_style_failure_and_absence(monkeypatch):
    install(monkeypatch, "fake_reid", link_global_ids=lambda: 2)
    assert adapter.link_identities("ws") == 2

    def boom():
        raise RuntimeError("model missing")

    install(monkeypatch, "fake_reid", link_global_ids=boom)
    assert adapter.link_identities("ws") is None
    install(monkeypatch, "fake_empty")
    assert adapter.link_identities("ws") is None


def test_the_zone_service_hands_its_database_to_perception(tmp_path, monkeypatch):
    db = open_db(wsmod.create("hooks", tmp_path / "ws").db_path)
    cams.insert_camera(db, name="Gate", kind="file", source_uri="/x.mp4", t0=1.0, t0_source="manual")
    seen = {}
    monkeypatch.setattr(adapter, "recompute_events", lambda camera_id, zones, db=None: seen.update(db=db) or 5)
    assert ZoneService(db, Bus()).save(ZONE).events == 5
    assert seen["db"] is db


@pytest.mark.parametrize("name,expected", [("tau_hi", 0.92), ("tau_lo", 0.62), ("margin", 0.03)])
def test_memory_thresholds_follow_the_calibration(name, expected):
    from evora.core.config import load_config

    assert load_config()["memory"][name] == expected


def test_profiles_and_origins():
    from evora.core.config import load_config

    assert load_config("gpu")["jobs"]["workers"] == 2 and load_config("cpu")["jobs"]["workers"] == 0
    assert {"http://localhost:3000", "http://127.0.0.1:3000"} <= set(load_config()["server"]["cors_origins"])
