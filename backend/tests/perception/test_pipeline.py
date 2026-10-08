"""Pipeline integration with a scripted detector and a fake embedder: no GPU, weights or network needed."""
import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")
pytest.importorskip("lancedb")

from contracts.models import IngestJob  # noqa: E402

from evora.core import cameras as cams  # noqa: E402
from evora.core import workspace as wsmod  # noqa: E402
from evora.core.db import open_db  # noqa: E402
from evora.perception import pipeline  # noqa: E402
from evora.perception.settings import IngestSettings  # noqa: E402
from evora.perception.track import TrackedBox  # noqa: E402

DIM = 16


class FakeEmbedder:
    dim = DIM

    def embed_images(self, images):
        out = np.zeros((len(images), DIM), dtype=np.float32)
        for i, im in enumerate(images):
            out[i, int(np.mean(im)) % DIM] = 1.0
        return out


class ScriptedTracker:
    """A person walks left to right (tracker id 1) for the first second, then another one appears (id 2)."""

    def __init__(self, det, cfg):
        self.n = 0

    def update(self, bgr):
        h, w = bgr.shape[:2]
        self.n += 1
        key = 1 if self.n <= 10 else 2
        x = 20 + self.n * 8
        return [TrackedBox(key, "person", 0.9, (float(x), h * 0.3, float(x + 60), h * 0.9))]


@pytest.fixture
def env(tmp_path, monkeypatch, sample_mp4):
    monkeypatch.setattr(pipeline, "get_embedder", lambda cfg: FakeEmbedder())
    monkeypatch.setattr(pipeline, "load_detector", lambda cfg: object())
    monkeypatch.setattr(pipeline, "FrameTracker", ScriptedTracker)
    ws = wsmod.create("pipeline-test", tmp_path / "workspaces")
    db = open_db(ws.db_path)
    cam = cams.insert_camera(db, name="Gate", kind="file", source_uri=str(sample_mp4), t0=1_790_000_000.0,
                             t0_source="manual", duration_s=2.0)
    return ws, db, cam


def _settings(**kw) -> IngestSettings:
    return IngestSettings(min_track_obs=2, motion_gate=False, fixed_fps=10, scene_every_s=1.0, **kw)


def test_ingest_writes_tracks_points_crops_and_scenes(env):
    ws, db, cam = env
    events: list[IngestJob] = []
    pipeline.ingest(cam, "cpu", {"L0", "L1"}, events.append, ws=ws, settings=_settings())

    with db.read() as c:
        tracks = c.execute("SELECT * FROM tracks ORDER BY t_start").fetchall()
        n_points = c.execute("SELECT count(*) FROM track_points").fetchone()[0]
    assert [t["id"] for t in tracks] == [f"{cam.id}:t000001", f"{cam.id}:t000002"]
    assert all(t["cls"] == "person" and t["camera_id"] == cam.id for t in tracks)
    assert tracks[0]["t_start"] == pytest.approx(cam.t0)          # times are epoch: t0 + seconds into the file
    assert tracks[0]["direction"] == "left_to_right"
    assert n_points > 0 and tracks[0]["best_crop"].startswith(f"crops/{cam.id}/")
    assert (ws.media_dir / tracks[0]["best_crop"]).is_file()

    from evora.core.vectors import open_store

    store = open_store(ws.vectors_dir)
    crops = store.open_table("crops").to_arrow().to_pylist()
    scenes = store.open_table("scenes").to_arrow().to_pylist()
    assert crops and {r["track_id"] for r in crops} == {t["id"] for t in tracks}
    assert {r["tile"] for r in scenes} == {"full", "tl", "tr", "bl", "br"}
    assert db.get_meta("embed_dim_image") == str(DIM)
    assert all((ws.media_dir / r["crop_path"]).is_file() for r in crops)

    finished = {e.layer for e in events if e.progress >= 1.0}
    assert finished == {"L0", "L1"}
    assert all(e.camera_id == cam.id for e in events)
    assert max(e.progress for e in events if e.layer == "L0" and e.progress < 1.0) < 1.0


def test_reingest_replaces_rows_instead_of_duplicating(env):
    from evora.core.vectors import open_store

    ws, db, cam = env

    def counts() -> tuple[int, int, int]:
        with db.read() as c:
            tracks = c.execute("SELECT count(*) FROM tracks").fetchone()[0]
            points = c.execute("SELECT count(*) FROM track_points").fetchone()[0]
        return tracks, points, open_store(ws.vectors_dir).open_table("crops").count_rows()

    pipeline.ingest(cam, "cpu", {"L1"}, lambda e: None, ws=ws, settings=_settings())
    first = counts()
    pipeline.ingest(cam, "cpu", {"L1"}, lambda e: None, ws=ws, settings=_settings())
    assert counts() == first
    assert first[0] == 2


def test_unimplemented_layers_are_skipped_not_reported_finished(env, caplog):
    ws, _, cam = env
    events: list[IngestJob] = []
    with caplog.at_level("WARNING"):
        pipeline.ingest(cam, "cpu", {"L2", "L3"}, events.append, ws=ws, settings=_settings())
    assert events == []
    assert "not implemented" in caplog.text


def test_missing_source_file_raises(env, tmp_path):
    ws, _, cam = env
    gone = cam.model_copy(update={"source_uri": str(tmp_path / "gone.mp4")})
    with pytest.raises(Exception, match="source file missing"):
        pipeline.ingest(gone, "cpu", {"L1"}, lambda e: None, ws=ws, settings=_settings())
