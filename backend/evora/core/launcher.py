"""`make up`: check the machine, start what is needed, serve the API and the built UI."""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
import webbrowser
from collections.abc import Callable
from typing import Any

from evora.core import doctor
from evora.core.config import REPO_ROOT, load_config

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# Problems that make starting pointless; everything else is a warning the operator can read and live with.
BLOCKING = {"python", "packages", "ffmpeg", "workspace", "tz"}


def blocking_problems(checks: list[doctor.Check], api_port: int) -> list[doctor.Check]:
    return [c for c in checks if c.status == doctor.FAIL and (c.id in BLOCKING or c.id == f"port_{api_port}")]


def ollama_running(env: doctor.Env) -> bool:
    host = (env.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    reply = env.http_get(f"{host}/api/tags", 1.0)
    return reply is not None and reply[0] == 200


def ensure_ollama(
    env: doctor.Env, want: bool, spawn: Callable[..., Any] = subprocess.Popen, wait_s: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, Any | None]:
    """Returns (state text, the child process if we started one). Never touches an Ollama that was already running."""
    if ollama_running(env):
        return "running", None
    exe = env.which("ollama")
    if exe is None:
        return "not installed", None
    if not want:
        return "installed, not running", None
    child = spawn([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    waited = 0.0
    while waited < wait_s:
        if ollama_running(env):
            return "started by evora", child
        sleep(0.5)
        waited += 0.5
    return "started but not answering yet", child


def banner(url: str, ws: str, on_prem: bool, keys: int, ollama: str, mediamtx: bool, ui: bool, problems: list[str]) -> str:
    lines = [
        "", f"  evora is starting at {url}", "",
        f"  workspace   {ws}",
        f"  privacy     {'on-prem: nothing leaves this machine' if on_prem else 'cloud allowed'}",
        f"  groq keys   {keys}",
        f"  ollama      {ollama}",
        f"  live replay {'ready (MediaMTX found)' if mediamtx else 'unavailable (MediaMTX not installed)'}",
        f"  interface   {'built UI served here' if ui else 'API only (no built UI yet)'}",
    ]
    if problems:
        lines += ["", "  to improve:"] + [f"    - {p}" for p in problems]
    lines.append("")
    return "\n".join(lines)


def count_keys(env: doctor.Env) -> int:
    value = env.environ.get("GROQ_KEYS", "")
    keys = [k for k in value.split(",") if k.strip()]
    keys += [v for k, v in env.environ.items() if k.startswith("GROQ_KEY") and k != "GROQ_KEYS" and v.strip()]
    return len(set(keys))


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Start evora: checks, local services, API and UI.")
    p.add_argument("--host", default=cfg["server"]["host"])
    p.add_argument("--port", type=int, default=int(cfg["server"]["port"]))
    p.add_argument("--live", action="store_true", help="start the local RTSP server for replay-as-live now")
    p.add_argument("--open", action="store_true", help="open the browser")
    p.add_argument("--no-checks", action="store_true")
    args = p.parse_args(argv)

    from evora.core.config import load_env_file

    load_env_file()
    on_prem = os.environ.get("evora_ONPREM") == "1"
    cfg["server"]["port"] = args.port
    env = doctor.Env(root=REPO_ROOT, cfg=cfg, environ=dict(os.environ), on_prem=on_prem, quick=True)

    if env.port_state(args.port) == "evora":
        print(f"evora is already running at http://{args.host}:{args.port}")
        return 0
    checks = [] if args.no_checks else doctor.run_checks(env)
    stoppers = blocking_problems(checks, args.port)
    if stoppers:
        print("evora cannot start yet:\n")
        for c in stoppers:
            print(f"  - {c.title}: {c.detail}" + (f"\n    fix: {c.fix}" if c.fix else ""))
        return 1

    if args.host not in LOOPBACK:
        print(f"WARNING: listening on {args.host}. Anyone on this network can reach the API; the privacy guard only "
              "controls what leaves this machine, not who connects.\n")

    start_ollama = on_prem or bool(cfg.get("up", {}).get("start_ollama", False))
    ollama_state, ollama_child = ensure_ollama(env, start_ollama)

    from evora.api.app import create_app

    app = create_app()
    ctx = app.state.ctx
    ui = any(getattr(r, "name", "") == "ui" for r in app.routes)
    problems = [f"{c.title}: {c.fix or c.detail}" for c in checks if c.status in (doctor.WARN, doctor.FAIL)]
    if args.live:
        try:
            ctx.live._ensure_server()
        except Exception as exc:  # noqa: BLE001 - the live demo is optional; say why and carry on
            problems.append(f"live replay: {getattr(exc, 'message', str(exc))}")
    url = f"http://{args.host}:{args.port}"
    print(banner(url, ctx.ws.slug, bool(ctx.settings["onprem"]), count_keys(env), ollama_state,
                 env.mediamtx() is not None, ui, problems), flush=True)
    if args.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        if ollama_child is not None:
            ollama_child.terminate()
    return 0

