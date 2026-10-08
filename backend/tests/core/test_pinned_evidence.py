import os
import time

import pytest
from contracts.models import Alert, Evidence
from fastapi.testclient import TestClient

from evora.alerts import store as alert_store
from evora.api.app import create_app
from evora.core import cameras as cams


def evidence(eid: str) -> Evidence:
    return Evidence(
        id=eid, camera_id="cam_01", camera_name="Gate", t_start=1.0, t_end=2.0, t_peak=1.5, offset_s=1.5,
        thumb_url=f"/api/media/thumb/{eid}.jpg", clip_url=f"/api/media/clip/{eid}.mp4", score=1.0,
    )


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "pins")
    client = TestClient(create_app(workspaces_root=tmp_path / "ws", gateway=object()))
    ctx = client.app.state.ctx
    cams.insert_camera(ctx.db, name="Gate", kind="file", source_uri="/x/gate.mp4", t0=1.0, t0_source="manual", duration_s=60.0)
    client.ctx = ctx
    return client


def raise_alert(ctx, eid: str) -> None:
    alert = Alert(id=f"al_{eid}", standing_query_id="sq_1", t=1.5, camera_id="cam_01", evidence=evidence(eid))
    alert_store.insert_alert(ctx.db, alert, None)


def cache_file(folder, name: str, age_s: float, size: int = 1000):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"x" * size)
    t = time.time() - age_s
    os.utime(path, (t, t))
    return path


def test_pinned_ids_come_from_the_alerts_and_survive_a_damaged_row(env):
    raise_alert(env.ctx, "ev_al_1")
    raise_alert(env.ctx, "ev_al_2")
    with env.ctx.db.write() as c:
        sql = "INSERT INTO alerts(id,sq_id,t,camera_id,track_id,evidence,acked) VALUES(?,'sq',1,'cam_01',NULL,?,0)"
        c.execute(sql, ("al_bad", "not json"))
        c.execute(sql, ("al_odd", "[1]"))
    assert alert_store.pinned_evidence_ids(env.ctx.db) == {"ev_al_1", "ev_al_2"}


def test_the_cache_trim_never_deletes_an_alerts_clip_or_thumbnail(env):
    ctx = env.ctx
    raise_alert(ctx, "ev_al_keep")
    thumbs, clips = ctx.ws.media_dir / "thumbs", ctx.ws.clips_dir
    pinned = [
        cache_file(clips, "ev_al_keep_raw.mp4", 900), cache_file(clips, "ev_al_keep_blur.mp4", 800),
        cache_file(thumbs, "ev_al_keep_blur.jpg", 700),
    ]
    loose = [cache_file(clips, "ev_old_raw.mp4", 600), cache_file(thumbs, "ev_old_raw.jpg", 500)]
    newest = cache_file(clips, "ev_new_raw.mp4", 1)
    ctx.media.cache_max_bytes = 1500
    removed = ctx.media.trim_cache(keep=newest)
    assert removed == 2 and not any(p.exists() for p in loose)
    assert all(p.exists() for p in pinned) and newest.exists()


def test_pinned_files_may_exceed_the_cap_and_everything_else_goes(env):
    ctx = env.ctx
    raise_alert(ctx, "ev_al_a")
    kept = cache_file(ctx.ws.clips_dir, "ev_al_a_raw.mp4", 100, size=5000)
    other = cache_file(ctx.ws.clips_dir, "ev_zzz_raw.mp4", 50, size=100)
    ctx.media.cache_max_bytes = 10
    ctx.media.trim_cache()
    assert kept.exists() and not other.exists()


def test_an_id_with_underscores_matches_exactly_not_by_prefix(env):
    ctx = env.ctx
    raise_alert(ctx, "ev_al_1")
    prefix_twin = cache_file(ctx.ws.clips_dir, "ev_al_12_raw.mp4", 100)
    ctx.media.cache_max_bytes = 1
    ctx.media.trim_cache()
    assert not prefix_twin.exists(), "ev_al_12 is not ev_al_1"


def test_a_failing_lookup_deletes_nothing_and_does_not_break_rendering(env):
    ctx = env.ctx
    victim = cache_file(ctx.ws.clips_dir, "ev_a_raw.mp4", 100)
    ctx.media.cache_max_bytes = 1

    def broken():
        raise RuntimeError("db locked")

    ctx.media._pinned = broken
    assert ctx.media.trim_cache() == 0 and victim.exists()


def test_a_service_without_a_pin_lookup_trims_as_before(env):
    ctx = env.ctx
    victim = cache_file(ctx.ws.clips_dir, "ev_al_x_raw.mp4", 100)
    ctx.media._pinned = None
    ctx.media.cache_max_bytes = 1
    ctx.media.trim_cache()
    assert not victim.exists()
