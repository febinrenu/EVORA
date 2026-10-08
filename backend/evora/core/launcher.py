"""`make up`: check the machine, start what is needed, serve the API and the built UI."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
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
    # one resident model at a time: the vision model next to SigLIP2, YOLO and OSNet leaves a laptop almost no free memory
    child_env = {**env.environ, "OLLAMA_MAX_LOADED_MODELS": env.environ.get("OLLAMA_MAX_LOADED_MODELS") or "1"}
    child = spawn(
        [exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, env=child_env,
    )
    waited = 0.0
    while waited < wait_s:
        if ollama_running(env):
            return "started by evora", child
        sleep(0.5)
        waited += 0.5
    return "started but not answering yet", child


UI_PORT = 3000


def ui_dir(env: doctor.Env) -> Path:
    return env.root / "frontend"


def ui_needs_build(frontend: Path) -> bool:
    """True when there is no production build (frontend/dist/index.html) or a source file is newer than it."""
    marker = frontend / "dist" / "index.html"  # the UI is a static export; it never writes .next/BUILD_ID
    if not marker.is_file():
        return True
    built = marker.stat().st_mtime
    watched = [frontend / "package.json", frontend / "next.config.ts"]
    for folder in ("src", "public"):
        if (frontend / folder).is_dir():
            watched += [p for p in (frontend / folder).rglob("*") if p.is_file()]
    return any(p.is_file() and p.stat().st_mtime > built for p in watched)


def stop_tree(child: Any) -> None:
    """Stop a child and everything it started (npm starts node, so terminating npm alone leaves the server running)."""
    pid = getattr(child, "pid", None)
    if sys.platform == "win32" and pid:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False)
    else:
        child.terminate()


def start_ui(
    env: doctor.Env, api_url: str, port: int = UI_PORT, spawn: Callable[..., Any] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep, wait_s: float = 40.0,
) -> tuple[str, Any | None]:
    """Build (when stale) and start the Next.js UI on loopback. Returns (what to tell the operator, child or None)."""
    frontend = ui_dir(env)
    if not (frontend / "package.json").is_file():
        return "no frontend/ folder yet: API only", None
    npm, node = env.which("npm"), env.which("node")
    if not npm or not node:
        return "Node.js is not installed: API only (install Node 20+ to serve the UI)", None
    if not (frontend / "node_modules").is_dir():
        return "dependencies are missing: run `cd frontend && npm ci` while online", None
    child_env = {**env.environ, "NEXT_PUBLIC_EVORA_API": api_url}
    if ui_needs_build(frontend):
        code, out = env.run_in(frontend, [npm, "run", "build"], child_env, 900.0)
        if code != 0:
            tail = " | ".join(out.strip().splitlines()[-3:])
            return f"the UI build failed ({tail}): API only", None
    child = spawn(
        [npm, "run", "start", "--", "--port", str(port), "--hostname", "127.0.0.1"], cwd=str(frontend), env=child_env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
    )
    waited = 0.0
    while waited < wait_s:
        if env.port_state(port) != "free":
            return f"http://127.0.0.1:{port}", child
        sleep(0.5)
        waited += 0.5
    return f"started but not answering on port {port} yet", child


def banner(url: str, ws: str, on_prem: bool, keys: int, ollama: str, mediamtx: bool, ui: bool | str, problems: list[str]) -> str:
    interface = ui if isinstance(ui, str) else ("built UI served here" if ui else "API only (no built UI yet)")
    lines = [
        "", f"  evora is starting at {url}", "",
        f"  workspace   {ws}",
        f"  privacy     {'on-prem: nothing leaves this machine' if on_prem else 'cloud allowed'}",
        f"  groq keys   {keys}",
        f"  ollama      {ollama}",
        f"  live replay {'ready (MediaMTX found)' if mediamtx else 'unavailable (MediaMTX not installed)'}",
        f"  interface   {interface}",
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
    p.add_argument("--no-ui", action="store_true", help="do not build or start the web interface")
    p.add_argument("--ui-port", type=int, default=UI_PORT)
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
    served_here = any(getattr(r, "name", "") == "ui" for r in app.routes)
    problems = [f"{c.title}: {c.fix or c.detail}" for c in checks if c.status in (doctor.WARN, doctor.FAIL)]
    if args.live:
        try:
            ctx.live._ensure_server()
        except Exception as exc:  # noqa: BLE001 - the live demo is optional; say why and carry on
            problems.append(f"live replay: {getattr(exc, 'message', str(exc))}")
    url = f"http://{args.host}:{args.port}"
    ui_child = None
    if served_here:
        interface: bool | str = f"built UI served at {url}"
    elif args.no_ui:
        interface = "API only (--no-ui)"
    else:
        print("Preparing the web interface (the first build can take a minute)...", flush=True)
        interface, ui_child = start_ui(env, url, args.ui_port)
    print(banner(url, ctx.ws.slug, bool(ctx.settings["onprem"]), count_keys(env), ollama_state,
                 env.mediamtx() is not None, interface, problems), flush=True)
    if args.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        if ui_child is not None:
            stop_tree(ui_child)
        if ollama_child is not None:
            ollama_child.terminate()
    return 0

