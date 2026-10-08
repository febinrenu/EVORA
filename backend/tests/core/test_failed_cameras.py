import pytest
from contracts.models import IngestJob
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.core import cameras as cams
from evora.core import doctor, launcher
from evora.core import perception_adapter as adapter
from tests.core.test_adapter_hooks import install
from tests.live.test_restream import wait_for

# ---- a missing perception stack says what to run ------------------------------------------------------------------------

def missing(name):
    def ingest(cam, profile, layers, on_progress):
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)
    return ingest


def test_a_missing_perception_package_names_the_setup_command(monkeypatch):
    install(monkeypatch, "fake_perception_missing", ingest=missing("torch"))
    cam = cams_info()
    with pytest.raises(adapter.PerceptionMissing) as exc:
        adapter.ingest(cam, "cpu", {"L0"}, lambda p: None)
    assert "start.bat setup" in str(exc.value) and "torch" in str(exc.value)


def test_any_other_missing_module_is_left_as_it_is(monkeypatch):
    install(monkeypatch, "fake_perception_other", ingest=missing("somethingelse"))
    with pytest.raises(ModuleNotFoundError) as exc:
        adapter.ingest(cams_info(), "cpu", {"L0"}, lambda p: None)
    assert not isinstance(exc.value, adapter.PerceptionMissing)


def cams_info():
    from contracts.models import CameraInfo

    return CameraInfo(id="cam_01", name="Gate", kind="file", source_uri="/x.mp4", t0=1.0, t0_source="manual")


def test_the_job_error_tells_the_operator_what_to_run(tmp_path, monkeypatch, sample_mp4):
    install(monkeypatch, "fake_perception_job", ingest=missing("ultralytics"))
    monkeypatch.setenv("evora_WORKSPACE", "missingstack")
    client = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))
    with open(sample_mp4, "rb") as fh:
        client.post("/api/cameras", files=[("files", ("gate.mp4", fh))])
    job = client.post("/api/ingest", json={"camera_ids": ["cam_01"]}).json()[0]
    runner = client.app.state.ctx.runner
    assert wait_for(lambda: runner.get(job["id"]).state == "error", 5)
    assert "start.bat setup" in runner.get(job["id"]).error


def test_the_start_banner_says_once_whether_footage_can_be_indexed():
    missing_row = [doctor.Check("perception", "Perception stack", doctor.WARN, "not installed: torch")]
    line = launcher.indexing_state(missing_row)
    assert "start.bat setup" in line and line.startswith("not possible yet")
    assert launcher.indexing_state([doctor.Check("perception", "Perception stack", doctor.OK, "torch 2.5")]) == "ready"
    assert launcher.indexing_state([]) is None, "checks skipped: no claim either way"
    text = launcher.banner("http://127.0.0.1:8700", "ws", False, 0, "running", True, True, [], line)
    assert "  indexing    not possible yet" in text
    assert "indexing" not in launcher.banner("u", "ws", False, 0, "x", True, True, [])


# ---- a camera that stopped with an error is not stuck there --------------------------------------------------------------

class Ingest:
    def __init__(self, fail=True):
        self.fail, self.calls = fail, 0

    def __call__(self, cam, profile, layers, on_progress):
        self.calls += 1
        if self.fail:
            raise RuntimeError("decoder crashed")
        for layer in sorted(layers):
            on_progress(IngestJob(id="", camera_id=cam.id, state="running", layer=layer, progress=1.0))


def app(tmp_path, monkeypatch, ingest):
    monkeypatch.setenv("evora_WORKSPACE", "retries")
    return TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object(), ingest_fn=ingest))


def failed_camera(tmp_path, monkeypatch, sample_mp4):
    ingest = Ingest(fail=True)
    client = app(tmp_path, monkeypatch, ingest)
    with open(sample_mp4, "rb") as fh:
        client.post("/api/cameras", files=[("files", ("gate.mp4", fh))])
    client.post("/api/ingest", json={"camera_ids": ["cam_01"]})
    ctx = client.app.state.ctx
    assert wait_for(lambda: cams.get_camera(ctx.db, "cam_01").status == "error", 5)
    ctx.runner.shutdown()
    return client, ingest


def test_uploading_a_failed_file_again_makes_it_ready_to_index(tmp_path, monkeypatch, sample_mp4):
    client, ingest = failed_camera(tmp_path, monkeypatch, sample_mp4)
    with open(sample_mp4, "rb") as fh:
        r = client.post("/api/cameras", files=[("files", ("gate.mp4", fh))])
    assert r.json()[0]["id"] == "cam_01" and r.json()[0]["status"] == "pending", "the interface offers 'Start indexing' again"
    assert cams.get_camera(client.app.state.ctx.db, "cam_01").status == "pending"


def test_a_failed_camera_is_indexed_again_when_the_app_starts(tmp_path, monkeypatch, sample_mp4):
    failed_camera(tmp_path, monkeypatch, sample_mp4)
    fixed = Ingest(fail=False)  # say setup has been run since
    client = app(tmp_path, monkeypatch, fixed)
    db = client.app.state.ctx.db
    assert wait_for(lambda: cams.get_camera(db, "cam_01").status == "ready", 5)
    assert fixed.calls == 1 and cams.get_camera(db, "cam_01").layers == ["L0", "L1", "L2"]


def test_the_startup_retry_can_be_switched_off(tmp_path, monkeypatch, sample_mp4):
    failed_camera(tmp_path, monkeypatch, sample_mp4)
    from evora.core import config

    real = config.load_config

    def no_retry(profile=None):
        cfg = real(profile)
        cfg["jobs"]["retry_failed_on_start"] = False
        return cfg

    monkeypatch.setattr("evora.api.context.load_config", no_retry, raising=False)
    monkeypatch.setattr(config, "load_config", no_retry)
    again = Ingest(fail=False)
    client = app(tmp_path, monkeypatch, again)
    assert cams.get_camera(client.app.state.ctx.db, "cam_01").status == "error" and again.calls == 0


def test_retry_failed_skips_live_cameras_and_healthy_ones(tmp_path, monkeypatch, sample_mp4):
    client, _ = failed_camera(tmp_path, monkeypatch, sample_mp4)
    db = client.app.state.ctx.db
    cams.insert_camera(db, name="Door", kind="rtsp", source_uri="rtsp://127.0.0.1/x", t0=1.0, t0_source="live")
    cams.set_status(db, "cam_02", "error")
    from evora.core.bus import Bus
    from evora.core.jobs import JobRunner

    runner = JobRunner(db, Bus(), workers=1, ingest_fn=Ingest(fail=False), default_layers=["L0"])
    jobs = runner.retry_failed()
    assert [j.camera_id for j in jobs] == ["cam_01"]
    runner.shutdown()
