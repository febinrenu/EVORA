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
        port_state=lambda port: "free", compute=lambda: ("cuda", "RTX 4060"), nvidia_gpu=lambda: None,
        blur_check=lambda: (True, "ok"),
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
    assert c.status == WARN and c.fix == "start.bat setup (Windows) or make setup-perception" and "torch" in c.detail


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


def test_the_web_interface_check_understands_a_next_app(tmp_path):
    f = tmp_path / "frontend"
    f.mkdir(parents=True, exist_ok=True)
    (f / "package.json").write_text("{}")
    base = healthy(tmp_path)
    no_node = by_id(run_checks(Env(**{**base.__dict__, "which": lambda n: None})))["ui"]
    assert no_node.status == WARN and "Node.js" in no_node.detail
    no_deps = by_id(run_checks(base))["ui"]
    assert no_deps.status == WARN and no_deps.fix == "cd frontend && npm ci"
    (f / "node_modules").mkdir()
    (f / "dist" / "index.html").unlink()  # the healthy() fixture starts with a build in place
    unbuilt = by_id(run_checks(base))["ui"]
    assert unbuilt.status == WARN and unbuilt.fix == "cd frontend && npm run build"
    (f / "dist" / "index.html").write_text("<html></html>")
    assert by_id(run_checks(base))["ui"].status == OK


# ---- --fix ---------------------------------------------------------------------------------------------------------------

class Runner:
    """Records the commands instead of running them; `fails` lists programs that exit non-zero."""

    def __init__(self, fails=()):
        self.calls, self.fails = [], set(fails)

    def __call__(self, argv, cwd, env):
        self.calls.append((argv, Path(cwd), dict(env)))
        return 1 if argv[0] in self.fails or argv[-1] in self.fails else 0


def broken(tmp_path, runner, **over):
    """No scipy, no face model, an Ollama without the vision model, and no frontend dependencies."""
    tags = json.dumps({"models": [{"name": "qwen3.5:4b"}]})
    env = healthy(
        tmp_path, version_of=lambda n: None if n == "scipy" else "1", http_get=lambda url, t=1.0: (200, tags),
        run_fix=runner, **over,
    )
    (tmp_path / "models" / "yunet" / "face_detection_yunet_2023mar.onnx").unlink()
    (tmp_path / "frontend" / "package.json").write_text("{}")
    return env


def test_every_fixable_row_lists_its_command(tmp_path):
    rows = by_id(run_checks(broken(tmp_path, Runner())))
    assert rows["packages"].fixes[0].argv == ["/usr/bin/uv", "--directory", "backend", "sync"]
    assert [f.argv for f in rows["ollama"].fixes] == [["ollama", "pull", "qwen3-vl:2b"]]
    assert rows["m_face"].fixes[0].argv[-2:] == ["--only", "yunet"]
    assert rows["ui"].fixes[0].cwd == "frontend" and rows["ui"].fixes[0].argv == ["npm", "ci"]
    assert by_id(run_checks(healthy(tmp_path)))["packages"].fixes == [], "nothing to fix, nothing listed"


def test_the_same_command_is_planned_once(tmp_path):
    env = healthy(tmp_path, version_of=lambda n: None if n == "scipy" else "1", tz_ok=lambda: False)
    fixes = doctor.planned_fixes(run_checks(env))
    assert len(fixes) == 1 and fixes[0].argv[-1] == "sync", "packages and time zone data share one sync"


def test_fix_with_yes_runs_everything_in_order_and_reports(tmp_path, capsys):
    runner = Runner()
    env = broken(tmp_path, runner)
    assert doctor.apply_fixes(env, run_checks(env), yes=True) is True
    assert [c[0][-1] for c in runner.calls] == ["sync", "yunet", "qwen3-vl:2b", "ci"], "in the order of the table"
    first_env = runner.calls[0][2]
    assert first_env["EVORA_MODELS_DIR"] == str(tmp_path / "models")
    assert runner.calls[-1][1] == tmp_path / "frontend" and runner.calls[0][1] == tmp_path
    out = capsys.readouterr().out
    assert out.count("done") == 4 and "failed" not in out


def test_one_failing_fix_does_not_stop_the_others(tmp_path, capsys):
    runner = Runner(fails={"ollama"})
    env = broken(tmp_path, runner)
    doctor.apply_fixes(env, run_checks(env), yes=True)
    assert len(runner.calls) == 4
    out = capsys.readouterr().out
    assert "failed (exit 1); the other fixes still run" in out and out.count("done") == 3


def test_without_yes_it_asks_and_a_no_runs_nothing(tmp_path, capsys):
    runner = Runner()
    env = broken(tmp_path, runner)
    asked = []
    assert doctor.apply_fixes(env, run_checks(env), confirm=lambda q: asked.append(q) or False) is False
    assert runner.calls == [] and asked == ["Run 4 commands?"]
    assert "Add --yes" in capsys.readouterr().out
    assert doctor.apply_fixes(env, run_checks(env), confirm=lambda q: True) is True and len(runner.calls) == 4


def test_a_script_without_yes_never_guesses(tmp_path, monkeypatch):
    runner = Runner()
    env = broken(tmp_path, runner)
    monkeypatch.setattr(doctor.sys.stdin, "isatty", lambda: False)
    assert doctor.apply_fixes(env, run_checks(env)) is False and runner.calls == []


def test_on_prem_refuses_every_fix_because_they_all_download(tmp_path, capsys):
    runner = Runner()
    env = broken(tmp_path, runner, on_prem=True)
    assert doctor.apply_fixes(env, run_checks(env), yes=True) is False and runner.calls == []
    out = capsys.readouterr().out
    assert "on-prem mode" in out and "none was run" in out and "npm ci" in out, "the commands are still shown"


def test_nothing_to_fix_says_so(tmp_path, capsys):
    runner = Runner()
    env = healthy(tmp_path, run_fix=runner)
    assert doctor.apply_fixes(env, run_checks(env), yes=True) is False and runner.calls == []
    assert "Nothing to fix automatically" in capsys.readouterr().out


def test_fixes_are_only_planned_for_rows_that_need_them(tmp_path):
    env = broken(tmp_path, Runner())
    checks = run_checks(env)
    for c in checks:
        if c.id == "packages":
            c.status = OK
    assert all(f.argv[-1] != "sync" for f in doctor.planned_fixes(checks))


def test_main_fixes_then_checks_again(tmp_path, monkeypatch, capsys):
    state = {"fixed": False}
    runner = Runner()

    def run_fix(argv, cwd, env):
        state["fixed"] = True
        return runner(argv, cwd, env)

    env = healthy(tmp_path, version_of=lambda n: "1" if state["fixed"] or n != "scipy" else None, run_fix=run_fix)
    monkeypatch.setattr(doctor, "Env", lambda **kw: env)
    code = doctor.main(["--fix", "--yes", "--quick"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.index("1 problem") < out.index("Checking again") < out.index("Demo-ready")


def test_json_output_lists_the_fixes_and_never_runs_them(tmp_path, monkeypatch, capsys):
    runner = Runner()
    env = broken(tmp_path, runner)
    monkeypatch.setattr(doctor, "Env", lambda **kw: env)
    doctor.main(["--json", "--fix", "--yes"])
    data = json.loads(capsys.readouterr().out)
    assert runner.calls == [] and any(c["fixes"] for c in data["checks"])


# ---- the recording buffer -------------------------------------------------------------------------------------------------

def test_the_recording_buffer_row_checks_the_disk_against_the_cap(tmp_path):
    row = by_id(run_checks(healthy(tmp_path)))["recording"]
    assert row.status == OK and "30 minutes" in row.detail and "2 GB each" in row.detail
    tight = by_id(run_checks(healthy(tmp_path, disk_free_gb=lambda p: 3.0)))["recording"]
    assert tight.status == WARN and "live.record_max_gb" in tight.fix


def test_no_recording_row_when_recording_is_off(tmp_path):
    env = healthy(tmp_path)
    env.cfg["live"]["record"] = False
    assert "recording" not in by_id(run_checks(env))


def test_a_running_ollama_without_the_one_model_setting_gets_a_hint(tmp_path):
    hint = by_id(run_checks(healthy(tmp_path)))["ollama"]
    assert hint.status == OK and "OLLAMA_MAX_LOADED_MODELS=1" in hint.detail
    env = healthy(tmp_path)
    env.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
    assert "OLLAMA_MAX_LOADED_MODELS" not in by_id(run_checks(env))["ollama"].detail


# ---- an NVIDIA card that PyTorch cannot use ---------------------------------------------------------------------------

def test_a_gpu_ignored_by_a_cpu_build_of_torch_says_how_to_fix_it(tmp_path):
    env = healthy(tmp_path, compute=lambda: ("cpu", "no GPU found"), nvidia_gpu=lambda: "NVIDIA GeForce RTX 4050 Laptop GPU")
    row = by_id(run_checks(env))["compute"]
    assert row.status == WARN and "RTX 4050" in row.detail and "CPU build" in row.detail
    assert "start.bat setup" in row.fix and row.fixes[0].argv[-5:] == ["sync", "--extra", "perception", "--extra", "embed"]


def test_without_a_gpu_the_plain_cpu_advice_stays(tmp_path):
    row = by_id(run_checks(healthy(tmp_path, compute=lambda: ("cpu", "no GPU found"), nvidia_gpu=lambda: None)))["compute"]
    assert row.status == WARN and "cpu profile" in row.detail and row.fixes == []


def test_no_torch_at_all_is_the_perception_rows_job(tmp_path):
    env = healthy(tmp_path, compute=lambda: ("none", "torch is not installed"), nvidia_gpu=lambda: "NVIDIA RTX 4060")
    assert "CPU build" not in by_id(run_checks(env))["compute"].detail
