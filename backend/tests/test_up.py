import json
import subprocess
from pathlib import Path

import pytest

from evora.core import doctor, launcher
from evora.core.doctor import FAIL, OK, WARN, Check, Env

KEY = "gsk_SECRETSECRETSECRET12345"


class Child:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


def env(running=True, has_ollama=True, **over):
    answers = {"up": running}
    kw = dict(
        root=Path("."), cfg={"server": {"port": 8700}}, environ={"GROQ_KEYS": f"{KEY},gsk_second_key_value"},
        which=lambda n: "/usr/bin/ollama" if has_ollama and n == "ollama" else None,
        http_get=lambda url, t=1.0: (200, json.dumps({"models": []})) if answers["up"] else None,
    )
    kw.update(over)
    e = Env(**kw)
    e.answers = answers
    return e


def test_a_running_ollama_is_left_alone():
    spawned = []
    state, child = launcher.ensure_ollama(env(), True, spawn=lambda *a, **k: spawned.append(a))
    assert (state, child) == ("running", None) and spawned == []


def test_ollama_that_is_not_installed():
    assert launcher.ensure_ollama(env(running=False, has_ollama=False), True)[0] == "not installed"


def test_ollama_is_not_started_unless_asked():
    spawned = []
    state, child = launcher.ensure_ollama(env(running=False), False, spawn=lambda *a, **k: spawned.append(a))
    assert state == "installed, not running" and child is None and spawned == []


def test_ollama_is_started_and_waited_for():
    e = env(running=False)
    spawned = []
    polls = {"n": 0}

    def sleep(_):
        polls["n"] += 1
        if polls["n"] == 3:
            e.answers["up"] = True

    def spawn(args, **kw):
        spawned.append(args)
        return Child()

    state, child = launcher.ensure_ollama(e, True, spawn=spawn, sleep=sleep)
    assert state == "started by evora" and spawned == [["/usr/bin/ollama", "serve"]] and isinstance(child, Child)


def test_ollama_that_never_answers_is_reported_and_still_returned_for_cleanup():
    e = env(running=False)
    state, child = launcher.ensure_ollama(e, True, spawn=lambda *a, **k: Child(), wait_s=1.0, sleep=lambda s: None)
    assert state == "started but not answering yet" and isinstance(child, Child)


def test_only_real_blockers_stop_the_start():
    checks = [
        Check("ffmpeg", "ffmpeg", FAIL, "missing"), Check("m_face", "Face model", FAIL, "missing"),
        Check("port_8700", "API port", FAIL, "busy"), Check("port_8554", "RTSP port", WARN, "busy"),
        Check("ollama", "Ollama", WARN, "down"), Check("python", "Python", OK, "3.12"),
    ]
    assert [c.id for c in launcher.blocking_problems(checks, 8700)] == ["ffmpeg", "port_8700"]
    assert launcher.blocking_problems(checks, 9999) == [checks[0]], "another port number is not the API port"


def test_the_key_count_never_exposes_a_key():
    e = env()
    assert launcher.count_keys(e) == 2
    text = launcher.banner("http://127.0.0.1:8700", "own-campus", False, launcher.count_keys(e), "running", True, False, [])
    assert KEY not in text and "groq keys   2" in text


def test_the_banner_says_what_is_missing():
    text = launcher.banner("http://127.0.0.1:8700", "w", True, 0, "not installed", False, False, ["Face model: run models"])
    assert "on-prem: nothing leaves this machine" in text and "MediaMTX not installed" in text
    assert "API only (no built UI yet)" in text and "- Face model: run models" in text


# ---- main() with the heavy parts replaced -------------------------------------------------------------------------

class FakeApp:
    class state:  # noqa: N801 - mimics app.state
        class ctx:  # noqa: N801
            class ws:  # noqa: N801
                slug = "own-campus"

            settings = {"onprem": False}

            class live:  # noqa: N801
                @staticmethod
                def _ensure_server():
                    return None

    routes: list = []


@pytest.fixture()
def wired(monkeypatch):
    calls = {"uvicorn": [], "checks": []}
    monkeypatch.setattr("evora.api.app.create_app", lambda: FakeApp)
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls["uvicorn"].append(kw))
    monkeypatch.setattr(launcher, "load_config", lambda: {"server": {"host": "127.0.0.1", "port": 8700}, "up": {}})
    healthy_rows = [Check("python", "Python", OK, "3.12")]
    monkeypatch.setattr(doctor, "run_checks", lambda e, skip=frozenset(): calls["checks"] or healthy_rows)
    monkeypatch.setattr(doctor.Env, "port_state", lambda self, port: "free", raising=False)
    monkeypatch.setattr("evora.core.config.load_env_file", lambda *a, **k: 0)
    calls["set_checks"] = lambda checks: calls["checks"].extend(checks)
    return calls


@pytest.fixture(autouse=True)
def _quiet_env(monkeypatch):
    monkeypatch.delenv("evora_ONPREM", raising=False)


def make_env_factory(monkeypatch, port_state="free"):
    real = doctor.Env

    def factory(**kw):
        e = real(**kw)
        e.port_state = lambda port: port_state
        e.which = lambda n: None
        e.http_get = lambda url, t=1.0: None
        return e

    monkeypatch.setattr(doctor, "Env", factory)


def test_the_default_host_is_loopback_and_starts_the_server(wired, monkeypatch, capsys):
    make_env_factory(monkeypatch)
    assert launcher.main([]) == 0
    assert wired["uvicorn"] == [{"host": "127.0.0.1", "port": 8700, "log_level": "warning"}]
    out = capsys.readouterr().out
    assert "WARNING" not in out and "evora is starting at http://127.0.0.1:8700" in out


def test_listening_on_the_network_prints_a_warning(wired, monkeypatch, capsys):
    make_env_factory(monkeypatch)
    assert launcher.main(["--host", "0.0.0.0"]) == 0
    assert "WARNING: listening on 0.0.0.0" in capsys.readouterr().out


def test_a_blocking_problem_stops_the_start_with_the_fix(wired, monkeypatch, capsys):
    make_env_factory(monkeypatch)
    wired["set_checks"]([Check("ffmpeg", "ffmpeg", FAIL, "not found", "winget install Gyan.FFmpeg")])
    assert launcher.main([]) == 1
    out = capsys.readouterr().out
    assert "cannot start" in out and "winget install Gyan.FFmpeg" in out and wired["uvicorn"] == []


def test_a_second_start_is_recognised(wired, monkeypatch, capsys):
    make_env_factory(monkeypatch, port_state="evora")
    assert launcher.main([]) == 0
    assert "already running" in capsys.readouterr().out and wired["uvicorn"] == []


def test_the_ollama_child_is_stopped_when_the_server_exits(wired, monkeypatch):
    make_env_factory(monkeypatch)
    child = Child()
    monkeypatch.setattr(launcher, "ensure_ollama", lambda env, want, **k: ("started by evora", child))
    launcher.main([])
    assert child.terminated is True


def test_on_prem_asks_for_ollama(wired, monkeypatch):
    make_env_factory(monkeypatch)
    seen = {}
    monkeypatch.setattr(launcher, "ensure_ollama", lambda env, want, **k: seen.setdefault("want", want) and ("running", None))
    monkeypatch.setenv("evora_ONPREM", "1")
    launcher.main([])
    assert seen["want"] is True


def test_subprocess_is_not_used_to_start_anything_else(wired, monkeypatch):
    make_env_factory(monkeypatch)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected process start"))
    assert launcher.main([]) == 0
