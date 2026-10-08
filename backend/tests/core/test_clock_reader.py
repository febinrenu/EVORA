import shutil
import subprocess
import threading
import time

import pytest
from contracts.models import IngestJob
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.core import cameras as cams
from evora.evidence import audit
from tests.live.test_restream import wait_for

OSD_T0 = 1791460000.0


@pytest.fixture(scope="session")
def undated_mp4(tmp_path_factory):
    """A clip with no clock in its name or metadata: only the on-screen reading could date it."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("undated") / "corridor.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2",
                    "-pix_fmt", "yuv420p", "-map_metadata", "-1", str(out)], check=True)
    return out


class Reader:
    """Stands in for the slow vision-model reading."""

    def __init__(self, result=(OSD_T0, "osd"), delay=0.4, fail=False):
        self.result, self.delay, self.fail, self.calls = result, delay, fail, []
        self.release = threading.Event()

    def __call__(self, path):
        self.calls.append(path)
        self.release.wait(self.delay)
        if self.fail:
            raise RuntimeError("vision model unavailable")
        return self.result


def make_client(tmp_path, monkeypatch, reader, seen=None):
    monkeypatch.setenv("evora_WORKSPACE", "clockread")

    def ingest(cam, profile, layers, on_progress):
        if seen is not None:
            seen.append(cam.t0)
        for layer in sorted(layers):
            on_progress(IngestJob(id="", camera_id=cam.id, state="running", layer=layer, progress=1.0))

    client = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object(), ingest_fn=ingest))
    ctx = client.app.state.ctx
    ctx.clock.read_fn = reader
    notes = []
    real = ctx.bus.publish
    monkeypatch.setattr(ctx.bus, "publish", lambda kind, data: (notes.append((kind, data)), real(kind, data))[0])
    client.notes = notes
    return client, ctx


def upload(client, path):
    with open(path, "rb") as fh:
        return client.post("/api/cameras", files=[("files", (path.name, fh))])


def test_an_upload_returns_at_once_and_the_clock_is_read_afterwards(tmp_path, monkeypatch, undated_mp4):
    reader = Reader(delay=1.0)
    client, ctx = make_client(tmp_path, monkeypatch, reader)
    started = time.monotonic()
    r = upload(client, undated_mp4)
    assert r.status_code == 200 and time.monotonic() - started < 1.0, "the upload did not wait for the vision model"
    assert r.json()[0]["t0_source"] == "manual" and ctx.clock.pending("cam_01")
    assert wait_for(lambda: cams.get_camera(ctx.db, "cam_01").t0_source == "osd", 5)
    assert cams.get_camera(ctx.db, "cam_01").t0 == OSD_T0
    states = [d["state"] for k, d in client.notes if k == "clock"]
    assert states == ["reading", "done"] and client.notes[-1][1]["t0"] == OSD_T0
    assert audit.entries(ctx.db, "clock_change")[0]["actor"] == "clock reader"


def test_indexing_waits_for_the_reading_and_starts_on_the_final_clock(tmp_path, monkeypatch, undated_mp4):
    seen: list[float] = []
    reader = Reader(delay=0.8)
    client, ctx = make_client(tmp_path, monkeypatch, reader, seen)
    upload(client, undated_mp4)
    client.post("/api/ingest", json={"camera_ids": ["cam_01"]})  # straight away, while the clock is still being read
    assert wait_for(lambda: seen, 5)
    assert seen == [OSD_T0], "the pipeline saw the on-screen clock, never the file time"


def test_a_file_with_a_known_clock_is_not_read_again(tmp_path, monkeypatch, sample_mp4):
    reader = Reader()
    client, ctx = make_client(tmp_path, monkeypatch, reader)
    r = upload(client, sample_mp4)
    assert r.json()[0]["t0_source"] == "metadata" and not ctx.clock.pending("cam_01")
    time.sleep(0.2)
    assert reader.calls == []


def test_a_failed_reading_keeps_the_file_time_and_never_blocks_indexing(tmp_path, monkeypatch, undated_mp4):
    seen: list[float] = []
    reader = Reader(delay=0.1, fail=True)
    client, ctx = make_client(tmp_path, monkeypatch, reader, seen)
    file_time = upload(client, undated_mp4).json()[0]["t0"]
    client.post("/api/ingest", json={"camera_ids": ["cam_01"]})
    assert wait_for(lambda: seen, 5) and seen == [file_time]
    assert cams.get_camera(ctx.db, "cam_01").t0_source == "manual"
    assert any(k == "clock" and d["state"] == "failed" for k, d in client.notes)


def test_a_reading_that_finds_nothing_better_changes_nothing(tmp_path, monkeypatch, undated_mp4):
    client, ctx = make_client(tmp_path, monkeypatch, Reader(result=(123.0, "manual"), delay=0.1))
    file_time = upload(client, undated_mp4).json()[0]["t0"]
    assert wait_for(lambda: not ctx.clock.pending("cam_01"), 5)
    assert cams.get_camera(ctx.db, "cam_01").t0 == file_time and audit.entries(ctx.db, "clock_change") == []


def test_a_clock_typed_by_the_operator_wins_over_the_reading(tmp_path, monkeypatch, undated_mp4):
    reader = Reader(delay=5.0)
    client, ctx = make_client(tmp_path, monkeypatch, reader)
    upload(client, undated_mp4)
    assert client.patch("/api/cameras/cam_01", json={"t0": 1700000000.0}).status_code == 200
    reader.release.set()
    assert wait_for(lambda: not ctx.clock.pending("cam_01"), 5)
    cam = cams.get_camera(ctx.db, "cam_01")
    assert (cam.t0, cam.t0_source) == (1700000000.0, "manual")


def test_shutdown_releases_anyone_waiting(tmp_path, monkeypatch, undated_mp4):
    reader = Reader(delay=30.0)
    client, ctx = make_client(tmp_path, monkeypatch, reader)
    upload(client, undated_mp4)
    ctx.clock.shutdown()
    assert ctx.clock.wait("cam_01", timeout=1.0) is True
    reader.release.set()  # let the stand-in reader thread end
