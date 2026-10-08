import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from contracts.models import Evidence
from fastapi.testclient import TestClient

from evora.api.app import create_app
from evora.evidence import audit, pack, store

T0 = 1791450000.0  # creation_time of the sample clip
EXPECTED = {
    "clip.mp4", "frame_1_start.jpg", "frame_2_peak.jpg", "frame_3_end.jpg", "evidence.json", "manifest.json",
    "SHA256SUMS", "README.txt",
}


@pytest.fixture(scope="session")
def gray_jpeg(tmp_path_factory) -> bytes:
    out = tmp_path_factory.mktemp("gray") / "gray.jpg"
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=gray:size=320x240", "-frames:v", "1", str(out)]
    subprocess.run(cmd, check=True)
    return out.read_bytes()


class Env:
    def __init__(self, client, blur):
        self.client, self.blur, self.ctx = client, blur, client.app.state.ctx
        self.cam_id = "cam_01"

    def evidence(self, eid="ev_pack", cam=None, with_context=True):
        ev = Evidence(
            id=eid, camera_id=cam or self.cam_id, camera_name="gate", t_start=T0 + 0.5, t_end=T0 + 1.5, t_peak=T0 + 1.0,
            offset_s=1.0, track_id="cam_01:t1", bbox=(0.2, 0.2, 0.5, 0.6), thumb_url=f"/api/media/thumb/{eid}.jpg",
            clip_url=f"/api/media/clip/{eid}.mp4", score=0.87, why=["siglip 0.31", "crossed gate"],
        )
        store.register(self.ctx.db, ev)
        if with_context:
            answer = {"query_id": "q_1", "evidence": [ev.model_dump(mode="json")], "nearest_miss": None}
            plan = {"intent": "exists", "targets": [], "source": "fastpath"}
            with self.ctx.db.write() as c:
                c.execute(
                    "INSERT INTO query_log(id,text,plan,answer,timings,created_at) VALUES('q_1',?,?,?,?,5)",
                    ("did a car pass the gate?", json.dumps(plan), json.dumps(answer), json.dumps({"ttfa": 12.5})),
                )
        return eid

    def pack(self, eid="ev_pack", **params):
        return self.client.post(f"/api/evidence/{eid}/pack", params=params)

    def members(self, response) -> dict[str, bytes]:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            return {i.filename: zf.read(i) for i in zf.infolist()}


@pytest.fixture()
def env(tmp_path, monkeypatch, sample_mp4, gray_jpeg):
    monkeypatch.setenv("evora_WORKSPACE", "packs")
    blur = {"fn": lambda jpeg: gray_jpeg}
    app = create_app(workspaces_root=tmp_path / "ws", blur_provider=lambda: blur["fn"], mock=False, gateway=object())
    client = TestClient(app)
    with open(sample_mp4, "rb") as fh:
        assert client.post("/api/cameras", files=[("files", ("clip.mp4", fh))]).status_code == 200
    return Env(client, blur)


def test_pack_has_the_expected_files_and_headers(env):
    eid = env.evidence()
    r = env.pack(eid)
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert f"evora_evidence_{eid}.zip" in r.headers["content-disposition"]
    assert set(env.members(r)) == EXPECTED
    assert r.headers["x-evora-blur"] == "applied"
    assert r.headers["x-evora-pack-sha256"] == hashlib.sha256(r.content).hexdigest()


def test_evidence_json_carries_the_question_and_both_timestamps(env):
    r = env.pack(env.evidence())
    doc = json.loads(env.members(r)["evidence.json"])
    assert doc["produced_by"]["kind"] == "query" and doc["produced_by"]["question"] == "did a car pass the gate?"
    assert doc["produced_by"]["timings_ms"] == {"ttfa": 12.5} and doc["evidence"]["why"] == ["siglip 0.31", "crossed gate"]
    peak = doc["times"]["peak"]
    assert peak["epoch_s"] == T0 + 1.0 and peak["iso_utc"].startswith("2026-10-08T09:00:01") and "+00:00" in peak["iso_local"]
    assert doc["times"]["offset_in_file_s"] == pytest.approx(1.0) and doc["camera"]["clock_source"] == "metadata"
    assert doc["bbox_normalized"] == [0.2, 0.2, 0.5, 0.6]


def test_an_alerts_pack_names_the_watch_that_raised_it(env):
    eid = env.evidence("ev_alert", with_context=False)
    ev = store.find_evidence(env.ctx.db, eid) or Evidence(
        id=eid, camera_id="cam_01", camera_name="gate", t_start=T0, t_end=T0 + 2, t_peak=T0 + 1, offset_s=1.0,
        thumb_url="/t", clip_url="/c", score=1.0,
    )
    with env.ctx.db.write() as c:
        c.execute(
            "INSERT INTO standing_queries(id,text,rule,active,created_at) VALUES('sq_1','watch the gate',?,1,1)",
            (json.dumps({"events": ["cross_line"]}),),
        )
        c.execute("INSERT INTO alerts(id,sq_id,t,camera_id,track_id,evidence,acked) VALUES('al_1','sq_1',?,'cam_01','t',?,0)",
                  (T0 + 1, ev.model_dump_json()))
    doc = json.loads(env.members(env.pack(eid))["evidence.json"])
    assert doc["produced_by"]["kind"] == "alert"
    assert doc["produced_by"]["standing_query"]["text"] == "watch the gate"
    assert doc["produced_by"]["standing_query"]["rule"] == {"events": ["cross_line"]}


def test_no_absolute_path_leaks_into_the_pack(env, tmp_path):
    members = env.members(env.pack(env.evidence()))
    for name, data in members.items():
        assert "/" not in name and "\\" not in name
        if name.endswith((".json", ".txt")):
            text = data.decode()
            assert str(tmp_path) not in text and str(env.ctx.ws.uploads_dir) not in text
            assert "uploads" not in text.replace("/api/media", "")
    manifest = json.loads(members["manifest.json"])
    assert manifest["source"]["file_name"].endswith("clip.mp4") and "/" not in manifest["source"]["file_name"]


def test_a_fresh_pack_verifies(env):
    r = env.pack(env.evidence())
    assert pack.verify_pack(r.content) == []


def test_sha256sum_check_passes_and_fails_after_tampering(env, tmp_path):
    exe = shutil.which("sha256sum")
    if exe is None:
        pytest.skip("sha256sum is not installed")
    members = env.members(env.pack(env.evidence()))
    folder = tmp_path / "unzipped"
    folder.mkdir()
    for name, data in members.items():
        (folder / name).write_bytes(data)
    ok = subprocess.run([exe, "-c", "SHA256SUMS"], cwd=folder, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    (folder / "clip.mp4").write_bytes(members["clip.mp4"][:-1] + b"X")
    bad = subprocess.run([exe, "-c", "SHA256SUMS"], cwd=folder, capture_output=True, text=True)
    assert bad.returncode != 0


def rebuild(members: dict[str, bytes], **changes) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        items = {**members, **{k: v for k, v in changes.items() if v is not None}}
        for k in [k for k, v in changes.items() if v is None]:
            items.pop(k, None)
        for name, data in items.items():
            zf.writestr(name, data)
    return out.getvalue()


def test_every_kind_of_tampering_is_detected(env):
    members = env.members(env.pack(env.evidence()))
    flipped = members["clip.mp4"][:-1] + bytes([members["clip.mp4"][-1] ^ 1])
    assert any("clip.mp4 does not match" in p for p in pack.verify_pack(rebuild(members, **{"clip.mp4": flipped})))
    missing = pack.verify_pack(rebuild(members, **{"frame_2_peak.jpg": None}))
    assert any("frame_2_peak.jpg" in p and "missing" in p for p in missing)
    assert any("extra.txt" in p and "not listed" in p for p in pack.verify_pack(rebuild(members, **{"extra.txt": b"x"})))
    assert any("unsafe path" in p for p in pack.verify_pack(rebuild(members, **{"../evil.txt": b"x"})))
    assert any("SHA256SUMS is missing" in p for p in pack.verify_pack(rebuild(members, **{"SHA256SUMS": None})))
    edited = json.loads(members["manifest.json"])
    edited["files"]["clip.mp4"]["sha256"] = "0" * 64
    problems = pack.verify_pack(rebuild(members, **{"manifest.json": json.dumps(edited).encode()}))
    assert problems, "a manifest that disagrees with the files is caught"
    assert pack.verify_pack(b"not a zip")[0].startswith("not a readable zip")


def test_command_line_verifier(env, tmp_path):
    good = tmp_path / "good.zip"
    r = env.pack(env.evidence())
    good.write_bytes(r.content)
    run = [sys.executable, "-I", "-m", "evora.evidence.pack", "verify"]
    ok = subprocess.run([*run, str(good)], capture_output=True, text=True)
    assert ok.returncode == 0 and "OK" in ok.stdout
    bad = tmp_path / "bad.zip"
    bad.write_bytes(rebuild(env.members(r), **{"clip.mp4": b"tampered"}))
    fail = subprocess.run([*run, str(bad)], capture_output=True, text=True)
    assert fail.returncode == 1 and "does not match" in fail.stdout


def test_rebuilding_with_the_same_clock_gives_identical_bytes(env):
    eid = env.evidence()
    args = dict(blur_setting=True, unblur_token_valid=False)
    first = pack.build_pack(env.ctx.db, env.ctx.ws, env.ctx.media, eid, now=1000.0, **args)
    second = pack.build_pack(env.ctx.db, env.ctx.ws, env.ctx.media, eid, now=1000.0, **args)
    assert first.sha256 == second.sha256


def test_source_hashes_and_the_stored_hash_cache(env, sample_mp4, monkeypatch):
    upload_sha = hashlib.sha256(Path(sample_mp4).read_bytes()).hexdigest()
    eid = env.evidence()
    src = json.loads(env.members(env.pack(eid))["manifest.json"])["source"]
    assert src["source_sha256"] == upload_sha and src["stored_sha256"] == upload_sha and src["transcoded_from"] is None
    calls = []
    real = pack.sha256_file
    monkeypatch.setattr(pack, "sha256_file", lambda path: calls.append(path.name) or real(path))
    env.pack(eid)
    assert [n for n in calls if n.endswith("clip.mp4")] == [], "the footage is not hashed again"


def test_a_converted_upload_records_where_it_came_from(tmp_path, monkeypatch, mpeg2_avi, gray_jpeg):
    monkeypatch.setenv("evora_WORKSPACE", "conv")
    app = create_app(workspaces_root=tmp_path / "ws", blur_provider=lambda: (lambda j: gray_jpeg), gateway=object())
    client = TestClient(app)
    with open(mpeg2_avi, "rb") as fh:
        client.post("/api/cameras", files=[("files", ("legacy.avi", fh))])
    e = Env(client, {})
    cam = client.get("/api/cameras").json()[0]
    ev = Evidence(
        id="ev_conv", camera_id="cam_01", camera_name="x", t_start=cam["t0"] + 0.5, t_end=cam["t0"] + 1.5, t_peak=cam["t0"] + 1,
        offset_s=1.0, thumb_url="/t", clip_url="/c", score=1.0,
    )
    store.register(e.ctx.db, ev)
    src = json.loads(e.members(e.pack("ev_conv"))["manifest.json"])["source"]
    assert src["transcoded_from"]["codec"] == "mpeg2video" and src["transcoded_from"]["original_sha256"] == src["source_sha256"]
    assert src["stored_sha256"] != src["source_sha256"]


# ---- privacy policy -----------------------------------------------------------------------------------------

def frame_bytes(env, r):
    return env.members(r)["frame_2_peak.jpg"]


def test_by_default_the_pack_is_blurred(env, gray_jpeg):
    r = env.pack(env.evidence())
    assert frame_bytes(env, r) == gray_jpeg
    manifest = json.loads(env.members(r)["manifest.json"])
    assert manifest["faces_blurred"] is True and manifest["unblurred_because"] is None


def test_with_blur_switched_off_the_pack_says_so(env, gray_jpeg):
    env.client.post("/api/settings", json={"blur_faces": False})
    r = env.pack(env.evidence())
    assert r.headers["x-evora-blur"] == "off" and frame_bytes(env, r) != gray_jpeg
    manifest = json.loads(env.members(r)["manifest.json"])
    assert manifest["faces_blurred"] is False and manifest["unblurred_because"] == "face blur is switched off"


@pytest.mark.parametrize("broken", [None, "raises"])
def test_blur_that_cannot_run_blocks_the_export_and_leaves_nothing(env, broken):
    def boom(_):
        raise RuntimeError("model file missing")

    env.blur["fn"] = None if broken is None else boom
    eid = env.evidence()
    r = env.pack(eid)
    assert r.status_code == 409 and "blur is unavailable" in r.json()["detail"]
    assert not list((env.ctx.ws.root / "exports").glob("*.zip")) and audit.entries(env.ctx.db, "export") == []


def test_a_token_with_a_reason_allows_an_unblurred_export_and_records_it(env, gray_jpeg):
    env.blur["fn"] = None
    eid = env.evidence()
    token = env.client.post("/api/media/unblur", json={"reason": "police request 4411", "evidence_id": eid}).json()["token"]
    r = env.pack(eid, unblur=token)
    assert r.status_code == 200 and r.headers["x-evora-blur"] == "off"
    manifest = json.loads(env.members(r)["manifest.json"])
    assert manifest["faces_blurred"] is False and "police request 4411" in manifest["unblurred_because"]
    assert audit.entries(env.ctx.db, "export")[0]["detail"]["unblurred_because"] == manifest["unblurred_because"]


def test_a_bogus_or_expired_token_does_not_help(env):
    import time

    env.blur["fn"] = None
    eid = env.evidence()
    assert env.pack(eid, unblur="bogus").status_code == 409
    token = env.client.post("/api/media/unblur", json={"reason": "x"}).json()["token"]
    env.ctx.unblur._clock = lambda: time.time() + 10_000
    assert env.pack(eid, unblur=token).status_code == 409


def test_every_export_is_audited_with_hashes(env):
    eid = env.evidence()
    r = env.pack(eid)
    entry = audit.entries(env.ctx.db, "export")[0]["detail"]
    assert entry["evidence_id"] == eid and entry["pack_sha256"] == r.headers["x-evora-pack-sha256"]
    assert set(entry["files"]) == EXPECTED - {"SHA256SUMS", "manifest.json"} and entry["faces_blurred"] is True
    env.pack(eid)
    assert len(audit.entries(env.ctx.db, "export")) == 2


# ---- errors -------------------------------------------------------------------------------------------------

def test_error_cases(env):
    assert env.pack("ev_missing").status_code == 404
    assert env.pack("a;b").status_code == 422
    rtsp = env.client.post("/api/cameras", json={"uri": "rtsp://10.0.0.2/live"}).json()[0]["id"]
    assert env.pack(env.evidence("ev_rtsp", cam=rtsp)).status_code == 409
    late = Evidence(id="ev_late", camera_id="cam_01", camera_name="x", t_start=T0 + 900, t_end=T0 + 910, t_peak=T0 + 905,
                    offset_s=905.0, thumb_url="/t", clip_url="/c", score=1.0)
    store.register(env.ctx.db, late)
    assert env.pack("ev_late").status_code == 422
    assert audit.entries(env.ctx.db, "export") == []


def test_mock_mode_returns_the_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("evora_WORKSPACE", "mockpack")
    c = TestClient(create_app(workspaces_root=tmp_path / "ws", mock=True))
    assert c.post("/api/evidence/ev_001/pack").content[:2] == b"PK"
