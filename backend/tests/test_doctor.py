import json
from pathlib import Path

import pytest

from evora.core import doctor
from evora.core.doctor import FAIL, OK, SKIP, WARN, Env, mask, render, run_checks, to_json, verdict

KEY = "gsk_SECRETSECRETSECRET12345"


def healthy(tmp_path, **over) -> Env:
    """An environment where everything is fine; each test breaks one thing."""
    md = tmp_path / "models"
    for rel in ("ultralytics/yolo26n.pt", "boxmot/osnet_x0_25_msmt17.pt", "yunet/face_detection_yunet_2023mar.onnx",
                "fastembed/model.onnx", "hf/hub/models--google--siglip2-base-patch16-224/snapshots/abc/model.safetensors"):
        f = md / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    ui = tmp_path / "frontend" / "dist"
    ui.mkdir(parents=True, exist_ok=True)
    (ui / "index.html").write_text("<html></html>")
    (tmp_path / ".env").write_text("GROQ_KEYS=x")
    versions = {p: "1.0" for p in doctor.CORE_PACKAGES + doctor.PERCEPTION_PACKAGES}
    tags = json.dumps({"models": [{"name": "qwen3.5:4b"}, {"name": "qwen3-vl:2b"}]})

    def run(cmd, timeout=10.0):
        if "-encoders" in cmd:
            return 0, " V..... libx264"
        if "-muxers" in cmd:
            return 0, " E rtsp  RTSP output"
        return 0, "ffmpeg version 7.1 Copyright (c) the FFmpeg developers\nbuilt with gcc"

    kw = dict(
        root=tmp_path,
        cfg={"server": {"port": 8700, "ui_dir": "frontend/dist"}, "live": {"rtsp_port": 8554}, "workspace": {"root": "ws"}},
        environ={"HF_HOME": str(md / "hf")}, which=lambda n: f"/usr/bin/{n}", run=run,
        http_get=lambda url, t=1.0: (200, tags), version_of=lambda n: versions.get(n), disk_free_gb=lambda p: 500.0,
        port_state=lambda port: "free", compute=lambda: ("cuda", "RTX 4060"), blur_check=lambda: (True, "ok"),
        workspace_check=lambda p: (True, str(p)), tz_ok=lambda: True, mediamtx=lambda: Path("mediamtx.exe"),
        groq_probe=lambda: [{"key": mask(KEY), "ok": True, "note": "ok", "remaining_requests": "900", "models": 5}],
        python_version=(3, 12), models_dir=None,
    )
    kw.pop("models_dir")
    kw["environ"]["EVORA_MODELS_DIR"] = str(md)
    kw.update(over)
    return Env(**kw)


def by_id(checks):
    return {c.id: c for c in checks}


def test_a_healthy_machine_is_demo_ready(tmp_path):
    checks = run_checks(healthy(tmp_path))
    bad = [c for c in checks if c.status not in (OK, SKIP)]
    assert bad == [], bad
    assert verdict(checks) == "Demo-ready"


@pytest.mark.parametrize("version,status", [((3, 12), OK), ((3, 11), OK), ((3, 10), FAIL), ((3, 13), FAIL)])
def test_python_versions(tmp_path, version, status):
    assert by_id(run_checks(healthy(tmp_path, python_version=version)))["python"].status == status


def test_missing_core_package_is_a_failure_with_the_fix(tmp_path):
    env = healthy(tmp_path, version_of=lambda n: None if n == "scipy" else "1")
    c = by_id(run_checks(env))["packages"]
    assert c.status == FAIL and "scipy" in c.detail and c.fix == "make setup"


def test_missing_perception_stack_is_only_a_warning(tmp_path):
    env = healthy(tmp_path, version_of=lambda n: None if n in doctor.PERCEPTION_PACKAGES else "1")
    c = by_id(run_checks(env))["perception"]
    assert c.status == WARN and c.fix == "make setup-perception" and "torch" in c.detail


def test_ffmpeg_missing_and_incomplete(tmp_path):
    assert by_id(run_checks(healthy(tmp_path, which=lambda n: None)))["ffmpeg"].status == FAIL
    no_x264 = healthy(tmp_path, run=lambda cmd, t=10.0: (0, "nothing useful"))
    c = by_id(run_checks(no_x264))["ffmpeg"]
    assert c.status == FAIL and "libx264" in c.detail and "rtsp muxer" in c.detail


@pytest.mark.parametrize("kind,status", [("cuda", OK), ("mps", OK), ("cpu", WARN), ("none", WARN)])
def test_compute(tmp_path, kind, status):
    assert by_id(run_checks(healthy(tmp_path, compute=lambda: (kind, "x"))))["compute"].status == status


@pytest.mark.parametrize("free,status", [(500, OK), (20, OK), (10, WARN), (1, FAIL)])
def test_disk(tmp_path, free, status):
    assert by_id(run_checks(healthy(tmp_path, disk_free_gb=lambda p: free)))["disk"].status == status


def test_unwritable_workspace(tmp_path):
    c = by_id(run_checks(healthy(tmp_path, workspace_check=lambda p: (False, "PermissionError"))))["workspace"]
    assert c.status == FAIL and "PermissionError" in c.detail


@pytest.mark.parametrize(
    "path,check_id,status,only",
    [
        ("ultralytics/yolo26n.pt", "m_detector", FAIL, "yolo"),
        ("yunet/face_detection_yunet_2023mar.onnx", "m_face", FAIL, "yunet"),
        ("boxmot/osnet_x0_25_msmt17.pt", "m_reid", WARN, "boxmot"),
        ("fastembed/model.onnx", "m_bge", WARN, "fastembed"),
        ("hf/hub/models--google--siglip2-base-patch16-224/snapshots/abc/model.safetensors", "m_siglip", FAIL, "siglip2"),
    ],
)
def test_each_missing_model_names_its_fix(tmp_path, path, check_id, status, only):
    env = healthy(tmp_path)
    (env.models_dir / path).unlink()
    c = by_id(run_checks(env))[check_id]
    assert c.status == status and c.fix == f"python scripts/models_download.py --only {only}"


def test_models_read_the_names_from_the_download_script(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    lines = ["YOLO_WEIGHTS = ['detector-x.pt']", "SIGLIP2 = 'org/model'", "BGE = 'b'", "BOXMOT_WEIGHTS = {'reid.pt': ''}"]
    (scripts / "models_download.py").write_text(chr(10).join(lines) + chr(10))
    c = by_id(run_checks(healthy(tmp_path)))
    assert c["m_detector"].status == FAIL and "detector-x.pt" in c["m_detector"].detail
    assert "org/model" in c["m_siglip"].detail and "reid.pt" in c["m_reid"].detail


def test_ollama_states(tmp_path):
    down = healthy(tmp_path, http_get=lambda u, t=1.0: None)
    c = by_id(run_checks(down))["ollama"]
    assert c.status == WARN and "installed but not running" in c.detail
    gone = healthy(tmp_path, http_get=lambda u, t=1.0: None, which=lambda n: None if n == "ollama" else "/x")
    assert "not installed" in by_id(run_checks(gone))["ollama"].detail
    on_prem = healthy(tmp_path, http_get=lambda u, t=1.0: None, on_prem=True)
    assert by_id(run_checks(on_prem))["ollama"].status == FAIL, "on-prem cannot work without local models"
    empty = healthy(tmp_path, http_get=lambda u, t=1.0: (200, json.dumps({"models": []})))
    c = by_id(run_checks(empty))["ollama"]
    assert c.status == WARN and "ollama pull qwen3.5:4b" in c.fix and "qwen3-vl:2b" in c.fix
    remote = healthy(tmp_path, environ={"OLLAMA_HOST": "http://10.1.2.3:11434", "EVORA_MODELS_DIR": str(tmp_path / "models")})
    assert by_id(run_checks(remote))["ollama"].status == WARN


def test_groq_skips_and_results_without_leaking_keys(tmp_path):
    assert by_id(run_checks(healthy(tmp_path, on_prem=True)))["groq"].status == SKIP
    assert by_id(run_checks(healthy(tmp_path, quick=True)))["groq"].status == SKIP
    none = healthy(tmp_path, groq_probe=lambda: [])
    assert by_id(run_checks(none))["groq"].status == WARN
    ok = healthy(tmp_path)
    c = by_id(run_checks(ok))["groq"]
    assert c.status == OK and "900 requests left" in c.detail
    bad = [{"key": mask(KEY), "ok": True, "note": "ok"}, {"key": "gsk_...zz", "ok": False, "note": "HTTP 401"}]
    mixed = by_id(run_checks(healthy(tmp_path, groq_probe=lambda: bad)))["groq"]
    assert mixed.status == WARN and "HTTP 401" in mixed.detail
    everything = render(run_checks(ok)) + to_json(run_checks(ok))
    assert KEY not in everything and "SECRETSECRET" not in everything


def test_mask_never_returns_the_key():
    assert mask(KEY) == "gsk_...45" and mask("short") == "***" and KEY not in mask(KEY)


@pytest.mark.parametrize("state,api,rtsp", [("free", OK, OK), ("evora", OK, OK), ("busy", FAIL, WARN)])
def test_ports(tmp_path, state, api, rtsp):
    c = by_id(run_checks(healthy(tmp_path, port_state=lambda p: state)))
    assert c["port_8700"].status == api and c["port_8554"].status == rtsp


def test_mediamtx_missing(tmp_path):
    c = by_id(run_checks(healthy(tmp_path, mediamtx=lambda: None)))["mediamtx"]
    assert c.status == WARN and "winget install bluenviron.mediamtx" == c.fix


def test_blur_and_ui_and_env_and_tz(tmp_path):
    env = healthy(tmp_path, blur_check=lambda: (False, "face model missing: x"), tz_ok=lambda: False)
    (tmp_path / ".env").unlink()
    (tmp_path / "frontend" / "dist" / "index.html").unlink()
    c = by_id(run_checks(env))
    assert c["blur"].status == FAIL and "face model missing" in c["blur"].detail
    assert c["tz"].status == FAIL and c["envfile"].status == WARN and c["ui"].status == WARN


def test_a_crashing_check_does_not_stop_the_others(tmp_path):
    def boom(_):
        raise RuntimeError("disk exploded")

    c = by_id(run_checks(healthy(tmp_path, disk_free_gb=boom)))
    assert c["disk"].status == FAIL and "check crashed" in c["disk"].detail and "disk exploded" in c["disk"].detail
    assert c["ffmpeg"].status == OK and "m_face" in c


def test_skipping_a_group(tmp_path):
    checks = run_checks(healthy(tmp_path), skip={"models"})
    assert by_id(checks)["models"].status == SKIP and "m_face" not in by_id(checks)


def test_rendering_and_the_verdict(tmp_path):
    env = healthy(tmp_path, compute=lambda: ("cpu", "no GPU"), blur_check=lambda: (False, "model missing"))
    checks = run_checks(env)
    text = render(checks)
    assert "\033" not in text and "FAIL" in text and "WARN" in text
    assert "-> python scripts/models_download.py --only yunet" in text, "fixes are shown for problems"
    assert "-> " not in render([doctor.Check("a", "A", OK, "fine", "unused")]), "and only for problems"
    assert "\033[31m" in render(checks, color=True)
    assert verdict(checks) == "1 problem to fix, 1 warning"
    assert verdict([doctor.Check("a", "A", WARN)]) == "1 warning, nothing blocking"
    assert verdict([doctor.Check("a", "A", FAIL), doctor.Check("b", "B", FAIL)]) == "2 problems to fix"


def test_json_shape(tmp_path):
    data = json.loads(to_json(run_checks(healthy(tmp_path))))
    assert data["ok"] is True and data["verdict"] == "Demo-ready"
    assert {"id", "title", "status", "detail", "fix"} <= set(data["checks"][0])


def test_exit_code_is_one_only_for_a_failure(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "run_checks", lambda env, skip=frozenset(): [doctor.Check("a", "A", WARN, "meh")])
    assert doctor.main(["--quick"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(doctor, "run_checks", lambda env, skip=frozenset(): [doctor.Check("a", "A", FAIL, "broken")])
    assert doctor.main(["--quick", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
