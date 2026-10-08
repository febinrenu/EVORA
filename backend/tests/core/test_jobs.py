import asyncio
import json
import time

import pytest
from contracts.models import IngestJob

from evora.core import cameras as cams
from evora.core import workspace as wsmod
from evora.core.bus import Bus
from evora.core.db import open_db
from evora.core.jobs import JobRunner


@pytest.fixture()
def db(tmp_path):
    return open_db(wsmod.create("t", tmp_path / "ws").db_path)


def add_cam(db, name="c"):
    return cams.insert_camera(db, name=name, kind="file", source_uri=f"/x/{name}.mp4", t0=1.0, t0_source="manual")


def wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def fake_ingest(fail_for=()):
    def run(cam, profile, layers, on_progress):
        if cam.id in fail_for:
            raise RuntimeError("decoder crashed")
        for layer in sorted(layers):
            on_progress(IngestJob(id="", camera_id=cam.id, state="running", layer=layer, progress=0.5))
            on_progress(IngestJob(id="", camera_id=cam.id, state="running", layer=layer, progress=1.0))
    return run


def finishing(spy=None, only=None):
    """An ingest that reports each requested layer as finished (progress 1.0), optionally only some of them."""
    def run(cam, profile, layers, on_progress):
        if spy is not None:
            spy(set(layers))
        for layer in sorted(layers):
            if only is None or layer in only:
                on_progress(IngestJob(id="", camera_id=cam.id, state="running", layer=layer, progress=1.0))
    return run


def test_camera_ids_increment(db):
    assert [add_cam(db, n).id for n in "abc"] == ["cam_01", "cam_02", "cam_03"]


def test_job_completes_and_records_layers(db):
    cam = add_cam(db)
    runner = JobRunner(db, Bus(), workers=1, ingest_fn=fake_ingest())
    job = runner.submit([cam.id])[0]
    assert wait_for(lambda: runner.get(job.id).state == "done")
    done = cams.get_camera(db, cam.id)
    assert done.layers == ["L0", "L1", "L2", "L3"] and done.status == "ready"
    runner.shutdown()


def test_one_failing_camera_does_not_stop_the_other(db):
    a, b = add_cam(db, "a"), add_cam(db, "b")
    runner = JobRunner(db, Bus(), workers=2, ingest_fn=fake_ingest(fail_for={a.id}))
    ja, jb = runner.submit([a.id, b.id])
    assert wait_for(lambda: runner.get(ja.id).state == "error" and runner.get(jb.id).state == "done")
    assert "decoder crashed" in runner.get(ja.id).error
    assert cams.get_camera(db, a.id).status == "error" and cams.get_camera(db, b.id).status == "ready"
    runner.shutdown()


def test_finished_layers_are_skipped_on_resubmit(db):
    cam = add_cam(db)
    cams.add_layers(db, cam.id, ["L0", "L1"])
    seen: list[set[str]] = []

    runner = JobRunner(db, Bus(), workers=1, ingest_fn=finishing(seen.append))
    job = runner.submit([cam.id])[0]
    assert wait_for(lambda: runner.get(job.id).state == "done")
    assert seen == [{"L2", "L3"}]
    again = runner.submit([cam.id])[0]  # everything finished: no new work
    assert again.state == "done" and seen == [{"L2", "L3"}]
    runner.shutdown()


def test_duplicate_submit_while_active_returns_same_job(db):
    cam = add_cam(db)
    import threading

    gate = threading.Event()
    def gated(cam, profile, layers, on_progress):
        gate.wait(5)
        finishing()(cam, profile, layers, on_progress)

    runner = JobRunner(db, Bus(), workers=1, ingest_fn=gated)
    first = runner.submit([cam.id])[0]
    assert runner.submit([cam.id])[0].id == first.id
    gate.set()
    assert wait_for(lambda: runner.get(first.id).state == "done")
    runner.shutdown()


def test_recover_requeues_interrupted_job(db):
    cam = add_cam(db)
    cams.add_layers(db, cam.id, ["L0"])
    with db.write() as c:
        c.execute("INSERT INTO ingest_jobs(id,camera_id,state,updated_at) VALUES('job_old',?,'running',0)", (cam.id,))
    seen = []
    runner = JobRunner(db, Bus(), workers=1, ingest_fn=finishing(seen.append))
    assert runner.recover() == 1
    assert wait_for(lambda: bool(seen))
    assert seen == [{"L1", "L2", "L3"}]
    assert runner.get("job_old").state == "error"
    runner.shutdown()


def test_unknown_layer_and_camera(db):
    runner = JobRunner(db, Bus(), workers=1, ingest_fn=fake_ingest())
    cam = add_cam(db)
    with pytest.raises(ValueError):
        runner.submit([cam.id], ["L9"])
    with pytest.raises(cams.CameraNotFound):
        runner.submit(["cam_99"])
    runner.shutdown()


def test_progress_events_arrive_in_order(db):
    async def scenario():
        bus = Bus()
        sub = bus.subscribe()
        cam = add_cam(db)
        runner = JobRunner(db, bus, workers=1, ingest_fn=fake_ingest(), default_layers=["L0"])
        runner.submit([cam.id])
        states = []
        while True:
            msg = await asyncio.wait_for(sub.queue.get(), 5)
            data = json.loads(msg["data"])
            if data["kind"] == "ingest":
                states.append((data["job"]["state"], data["job"]["progress"]))
            if data["kind"] == "ingest" and data["job"]["state"] == "done":
                break
        runner.shutdown()
        return states

    states = asyncio.run(scenario())
    assert states[0] == ("queued", 0.0)
    assert states[-1] == ("done", 1.0)
    order = [s for s, _ in states]
    assert order.index("running") > order.index("queued")


def test_bus_heartbeat_when_idle():
    async def scenario():
        bus = Bus()
        stream = bus.stream(heartbeat_s=0.05)
        msg = await anext(stream)
        await stream.aclose()
        return json.loads(msg["data"])["kind"], len(bus._subs)

    assert asyncio.run(scenario()) == ("heartbeat", 0)


def test_only_layers_that_finished_are_recorded(db):
    cam = add_cam(db)
    seen: list[set[str]] = []
    runner = JobRunner(db, Bus(), workers=1, ingest_fn=finishing(seen.append, only={"L0", "L1"}))
    job = runner.submit([cam.id])[0]
    assert wait_for(lambda: runner.get(job.id).state == "done")
    done = cams.get_camera(db, cam.id)
    assert done.layers == ["L0", "L1"] and done.status == "ready"
    again = runner.submit([cam.id])[0]  # L2 and L3 were skipped, so a later ingest tries just those
    assert wait_for(lambda: runner.get(again.id).state == "done")
    assert seen == [{"L0", "L1", "L2", "L3"}, {"L2", "L3"}]
    runner.shutdown()


def test_ingest_that_completes_nothing_is_an_error(db):
    cam = add_cam(db)
    runner = JobRunner(db, Bus(), workers=1, ingest_fn=lambda c, p, ly, cb: None)
    job = runner.submit([cam.id])[0]
    assert wait_for(lambda: runner.get(job.id).state == "error")
    assert "without completing" in runner.get(job.id).error
    assert cams.get_camera(db, cam.id).status == "error"
    runner.shutdown()


def test_the_real_pipeline_is_told_which_workspace_to_use(db, monkeypatch):
    from evora.core import perception_adapter

    seen = {}

    def real(cam, profile, layers, on_progress, *, ws=None, settings=None):
        seen["ws"] = ws
        for layer in sorted(layers):
            on_progress(IngestJob(id="", camera_id=cam.id, state="running", layer=layer, progress=1.0))

    monkeypatch.setattr(perception_adapter, "_find", lambda name: real if name == "ingest" else None)
    cam = add_cam(db)
    runner = JobRunner(db, Bus(), workers=1, ws="THE-WORKSPACE")
    job = runner.submit([cam.id], ["L0"])[0]
    assert wait_for(lambda: runner.get(job.id).state == "done")
    assert seen["ws"] == "THE-WORKSPACE"
    runner.shutdown()


def test_a_perception_function_without_a_ws_parameter_is_still_called(db, monkeypatch):
    from evora.core import perception_adapter

    def old_style(cam, profile, layers, on_progress):
        on_progress(IngestJob(id="", camera_id=cam.id, state="running", layer="L0", progress=1.0))

    monkeypatch.setattr(perception_adapter, "_find", lambda name: old_style if name == "ingest" else None)
    cam = add_cam(db)
    runner = JobRunner(db, Bus(), workers=1, ws="x")
    job = runner.submit([cam.id], ["L0"])[0]
    assert wait_for(lambda: runner.get(job.id).state == "done")
    runner.shutdown()
