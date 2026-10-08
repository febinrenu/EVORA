import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from contracts.models import Evidence
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.core import cameras as cams
from evora.evidence import audit, store

T0 = 1791450000.0  # creation_time of the sample clip


@pytest.fixture(scope="session")
def gray_jpeg(tmp_path_factory) -> bytes:
    out = tmp_path_factory.mktemp("gray") / "gray.jpg"
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=gray:size=320x240", "-frames:v", "1", str(out)]
    subprocess.run(cmd, check=True)
    return out.read_bytes()


class Env:
    def __init__(self, client, ctx, cam_id, blur_fn):
        self.client, self.ctx, self.cam_id, self.blur_fn = client, ctx, cam_id, blur_fn

    def evidence(self, eid="ev_001", t_start=T0 + 0.5, t_end=T0 + 1.0, t_peak=T0 + 0.8, cam=None, bbox=(0.2, 0.2, 0.5, 0.6)):
        store.register(self.ctx.db, Evidence(
            id=eid, camera_id=cam or self.cam_id, camera_name="c", t_start=t_start, t_end=t_end, t_peak=t_peak,
            offset_s=t_peak - T0, bbox=bbox, thumb_url="/t", clip_url="/c", score=0.9,
        ))
        return eid


@pytest.fixture()
def env(tmp_path, monkeypatch, sample_mp4, gray_jpeg):
    monkeypatch.setenv("evora_WORKSPACE", "media")
    holder = {"fn": lambda jpeg: gray_jpeg}
    client = TestClient(create_app(workspaces_root=tmp_path / "ws", blur_provider=lambda: holder["fn"]))
    with open(sample_mp4, "rb") as fh:
        cam_id = client.post("/api/cameras", files=[("files", ("clip.mp4", fh))]).json()[0]["id"]
    return Env(client, client.app.state.ctx, cam_id, holder)


def dims(data: bytes) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
        "-of", "json", "-i", "pipe:0",
    ]
    out = subprocess.run(cmd, input=data, capture_output=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    return s["width"], s["height"]


def duration(data: bytes) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", "-i", "pipe:0"]
    out = subprocess.run(cmd, input=data, capture_output=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def test_unblurred_thumb_is_a_real_frame(env):
    env.client.post("/api/settings", json={"blur_faces": False})
    r = env.client.get(f"/api/media/thumb/{env.evidence()}.jpg")
    assert r.status_code == 200 and r.headers["x-evora-blur"] == "off"
    assert dims(r.content) == (320, 240)


def test_thumb_is_cached(env):
    eid = env.evidence()
    env.client.get(f"/api/media/thumb/{eid}.jpg")
    calls = env.ctx.media.ffmpeg_calls
    env.client.get(f"/api/media/thumb/{eid}.jpg")
    assert env.ctx.media.ffmpeg_calls == calls


def test_blur_is_applied_by_default(env, gray_jpeg):
    r = env.client.get(f"/api/media/thumb/{env.evidence()}.jpg")
    assert r.headers["x-evora-blur"] == "applied" and r.content == gray_jpeg


def test_blur_unavailable_is_reported_not_hidden(env):
    env.blur_fn["fn"] = None
    r = env.client.get(f"/api/media/thumb/{env.evidence()}.jpg")
    assert r.status_code == 200 and r.headers["x-evora-blur"] == "unavailable"


def test_toggling_blur_does_not_serve_stale_cache(env, gray_jpeg):
    eid = env.evidence()
    blurred = env.client.get(f"/api/media/thumb/{eid}.jpg").content
    env.client.post("/api/settings", json={"blur_faces": False})
    raw = env.client.get(f"/api/media/thumb/{eid}.jpg").content
    assert blurred == gray_jpeg and raw != gray_jpeg


def test_clip_is_clamped_to_footage_and_streamable(env):
    env.client.post("/api/settings", json={"blur_faces": False})
    r = env.client.get(f"/api/media/clip/{env.evidence()}.mp4")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    head = r.content[:4096]
    assert b"moov" in head and (b"mdat" not in head or head.index(b"moov") < head.index(b"mdat"))
    assert duration(r.content) == pytest.approx(2.0, abs=0.3)


def test_clip_supports_range_requests(env):
    env.client.post("/api/settings", json={"blur_faces": False})
    eid = env.evidence()
    full = env.client.get(f"/api/media/clip/{eid}.mp4").content
    part = env.client.get(f"/api/media/clip/{eid}.mp4", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and part.content == full[:100]
    assert part.headers["content-range"] == f"bytes 0-99/{len(full)}"


def test_blurred_clip_differs_from_raw(env):
    eid = env.evidence()
    blurred = env.client.get(f"/api/media/clip/{eid}.mp4")
    assert blurred.headers["x-evora-blur"] == "applied" and duration(blurred.content) > 1.0
    env.client.post("/api/settings", json={"blur_faces": False})
    raw = env.client.get(f"/api/media/clip/{eid}.mp4")
    assert raw.content != blurred.content


def test_concurrent_clip_requests_render_once(env):
    env.blur_fn["fn"] = None
    eid = env.evidence()
    before = env.ctx.media.ffmpeg_calls
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: env.client.get(f"/api/media/clip/{eid}.mp4").status_code, range(4)))
    assert results == [200] * 4 and env.ctx.media.ffmpeg_calls - before == 1


def test_unblur_needs_a_valid_token_and_is_audited(env, gray_jpeg):
    eid = env.evidence()
    assert env.client.post("/api/media/unblur", json={"reason": " "}).status_code == 422
    for url in (f"/api/media/thumb/{eid}.jpg", f"/api/media/thumb/{eid}.jpg?unblur=bogus"):
        assert env.client.get(url).content == gray_jpeg
    tok = env.client.post("/api/media/unblur", json={"reason": "incident review", "evidence_id": eid}).json()
    shown = env.client.get(f"/api/media/thumb/{eid}.jpg", params={"unblur": tok["token"]})
    assert shown.headers["x-evora-blur"] == "off" and shown.content != gray_jpeg
    assert audit.entries(env.ctx.db, "unblur_token")[0]["detail"]["reason"] == "incident review"
    assert len(audit.entries(env.ctx.db, "unblur_view")) == 1


def test_expired_token_is_blurred_again(env, gray_jpeg):
    eid = env.evidence()
    tok = env.client.post("/api/media/unblur", json={"reason": "x"}).json()["token"]
    env.ctx.unblur._clock = lambda: time.time() + 10_000
    assert env.client.get(f"/api/media/thumb/{eid}.jpg", params={"unblur": tok}).content == gray_jpeg


def test_turning_blur_off_is_audited(env):
    env.client.post("/api/settings", json={"blur_faces": False})
    assert audit.entries(env.ctx.db, "blur_setting")[0]["detail"] == {"blur_faces": False}


def test_error_cases(env):
    assert env.client.get("/api/media/thumb/ev_404.jpg").status_code == 404
    assert env.client.get("/api/media/thumb/a;b.jpg").status_code == 422
    assert env.client.get("/api/media/clip/x..y.mp4").status_code == 422
    rtsp = env.client.post("/api/cameras", json={"uri": "rtsp://10.0.0.2/live"}).json()[0]["id"]
    eid = env.evidence("ev_rtsp", cam=rtsp)
    assert env.client.get(f"/api/media/thumb/{eid}.jpg").status_code == 409
    late = env.evidence("ev_late", t_start=T0 + 500, t_end=T0 + 510, t_peak=T0 + 505)
    assert env.client.get(f"/api/media/clip/{late}.mp4").status_code == 422
    assert not list(env.ctx.ws.clips_dir.glob("*.tmp*"))


def test_frame_route_clamps_out_of_range_times(env):
    env.client.post("/api/settings", json={"blur_faces": False})
    for t in (T0 - 100, T0 + 1, T0 + 9999):
        r = env.client.get(f"/api/cameras/{env.cam_id}/frame", params={"t": t})
        assert r.status_code == 200 and dims(r.content) == (320, 240)
    assert env.client.get("/api/cameras/cam_99/frame", params={"t": 1}).status_code == 404
    assert cams.get_camera(env.ctx.db, env.cam_id).t0 == pytest.approx(T0, abs=1)


def _age(path, seconds_ago):
    import os

    t = time.time() - seconds_ago
    os.utime(path, (t, t))


def test_cache_removes_least_recently_used_first(env):
    env.client.post("/api/settings", json={"blur_faces": False})
    thumbs = env.ctx.ws.media_dir / "thumbs"
    ids = [env.evidence(f"ev_{n}") for n in "abcd"]
    env.client.get(f"/api/media/thumb/{ids[0]}.jpg")
    size = next(thumbs.glob("ev_a_*")).stat().st_size
    env.ctx.media.cache_max_bytes = 2 * size + 10
    _age(next(thumbs.glob("ev_a_*")), 300)
    env.client.get(f"/api/media/thumb/{ids[1]}.jpg")
    _age(next(thumbs.glob("ev_b_*")), 200)
    env.client.get(f"/api/media/thumb/{ids[2]}.jpg")  # over the cap: the oldest (a) goes
    assert sorted(p.name.split("_raw")[0] for p in thumbs.glob("*.jpg")) == ["ev_b", "ev_c"]
    _age(next(thumbs.glob("ev_c_*")), 100)  # explicit ages: two files touched within one clock tick would tie
    env.client.get(f"/api/media/thumb/{ids[1]}.jpg")  # a hit on b makes it recent
    env.client.get(f"/api/media/thumb/{ids[3]}.jpg")  # now c is the oldest
    assert sorted(p.name.split("_raw")[0] for p in thumbs.glob("*.jpg")) == ["ev_b", "ev_d"]


def test_cache_never_removes_the_file_just_rendered(env):
    env.client.post("/api/settings", json={"blur_faces": False})
    env.ctx.media.cache_max_bytes = 1
    r = env.client.get(f"/api/media/clip/{env.evidence()}.mp4")
    assert r.status_code == 200 and len(list(env.ctx.ws.clips_dir.glob("*.mp4"))) == 1


def test_prerender_warms_the_top_three_only(env):
    env.client.post("/api/settings", json={"blur_faces": False})
    ids = [env.evidence(f"ev_{n}") for n in "abcd"]
    futures = env.ctx.prerender.schedule(ids)
    assert len(futures) == 3 and [f.result(timeout=60) for f in futures] == [True] * 3
    calls = env.ctx.media.ffmpeg_calls
    for eid in ids[:3]:
        assert env.client.get(f"/api/media/thumb/{eid}.jpg").status_code == 200
        assert env.client.get(f"/api/media/clip/{eid}.mp4").status_code == 200
    assert env.ctx.media.ffmpeg_calls == calls, "served from the pre-rendered cache"


def test_prerender_failure_is_contained(env):
    futures = env.ctx.prerender.schedule(["ev_missing"])
    assert futures[0].result(timeout=30) is False


def _boom(_jpeg):
    raise RuntimeError("face model missing")


def test_a_blur_function_that_cannot_run_is_reported_not_a_crash(env):
    env.blur_fn["fn"] = _boom
    eid = env.evidence()
    thumb = env.client.get(f"/api/media/thumb/{eid}.jpg")
    assert thumb.status_code == 200 and thumb.headers["x-evora-blur"] == "unavailable"
    clip = env.client.get(f"/api/media/clip/{eid}.mp4")
    assert clip.status_code == 200 and clip.headers["x-evora-blur"] == "unavailable"
    frame = env.client.get(f"/api/cameras/{env.cam_id}/frame", params={"t": T0 + 0.5})
    assert frame.status_code == 200 and frame.headers["x-evora-blur"] == "unavailable"
    assert not list(env.ctx.ws.clips_dir.glob("*_blur.mp4"))


def test_an_unblurred_image_is_never_filed_as_the_blurred_one(env, gray_jpeg):
    eid = env.evidence()
    env.blur_fn["fn"] = _boom
    env.client.get(f"/api/media/thumb/{eid}.jpg")
    assert not list((env.ctx.ws.media_dir / "thumbs").glob("*_blur.jpg"))
    env.blur_fn["fn"] = lambda jpeg: gray_jpeg  # the model arrives later: blurred output must now be served
    again = env.client.get(f"/api/media/thumb/{eid}.jpg")
    assert again.headers["x-evora-blur"] == "applied" and again.content == gray_jpeg


# ---- evidence clips are capped in size and frame rate ----------------------------------------------------------------------

def probe_stream(content: bytes, tmp_path) -> tuple[int, int, float]:
    path = tmp_path / "probe.mp4"
    path.write_bytes(content)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    num, den = out[2].split("/")
    return int(out[0]), int(out[1]), float(num) / float(den)


def test_a_hd_clip_is_rendered_at_most_1280_wide_and_15_fps(env, tmp_path):
    hd = tmp_path / "hd.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=3", "-pix_fmt", "yuv420p",
         "-metadata", "creation_time=2026-10-08T09:00:00Z", str(hd)], check=True,
    )
    with open(hd, "rb") as fh:
        cid = env.client.post("/api/cameras", files=[("files", ("hd.mp4", fh))]).json()[0]["id"]
    eid = env.evidence("ev_hd", t_start=T0 + 1.0, t_end=T0 + 2.0, t_peak=T0 + 1.5, cam=cid)
    env.client.post("/api/settings", json={"blur_faces": False})
    r = env.client.get(f"/api/media/clip/{eid}.mp4")
    assert r.status_code == 200 and probe_stream(r.content, tmp_path) == (1280, 720, 15.0)
    env.client.post("/api/settings", json={"blur_faces": True})  # the test blur returns a fixed small image, so check the rate
    r = env.client.get(f"/api/media/clip/{eid}.mp4")
    assert r.status_code == 200 and probe_stream(r.content, tmp_path)[2] == 15.0, "the blurred clip keeps the capped rate"


def test_a_small_slow_clip_is_left_as_it_is(env, tmp_path):
    env.client.post("/api/settings", json={"blur_faces": False})
    r = env.client.get(f"/api/media/clip/{env.evidence()}.mp4")
    assert r.status_code == 200 and probe_stream(r.content, tmp_path) == (320, 240, 10.0), "never enlarged, no frames added"


def test_clip_rate_rules(env):
    from evora.core import cameras as cams

    media, cam = env.ctx.media, cams.get_camera(env.ctx.db, env.cam_id)
    assert media.clip_fps(cam.model_copy(update={"fps": 30.0})) == 15.0
    assert media.clip_fps(cam.model_copy(update={"fps": 12.5})) == 12.5
    assert media.clip_fps(cam.model_copy(update={"fps": None})) == 15.0, "a live camera's rate is unknown: use the cap"
    assert media._clip_filter(cam.model_copy(update={"fps": 10.0})) == ["-vf", "scale=w='min(1280,iw)':h=-2"]
    media.clip_max_fps, media.clip_max_width = 0, 0
    assert media._clip_filter(cam) == [] and media.clip_fps(cam.model_copy(update={"fps": 30.0})) == 30.0, "0 turns a cap off"
