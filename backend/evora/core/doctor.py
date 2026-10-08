"""`make doctor`: one table that says whether this machine is ready to demo, and the command that fixes each gap.

The environment is injected (`Env`) so every check is testable without the real machine. A check that crashes is
reported as a failed row; the doctor itself never crashes and never prints a key.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from evora.core.media_service import probe_jpeg
from evora.live.restream import find_mediamtx

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
CORE_PACKAGES = ["fastapi", "pydantic", "lancedb", "numpy", "opencv-python-headless", "av", "scipy", "tzdata", "httpx"]
PERCEPTION_PACKAGES = ["torch", "ultralytics", "transformers", "boxmot"]
OLLAMA_NEEDED = {"planner": ["qwen3.5:4b"], "vision": ["qwen3-vl:2b", "qwen3-vl:4b"]}



@dataclass
class Check:
    id: str
    title: str
    status: str
    detail: str = ""
    fix: str = ""


def mask(key: str) -> str:
    return f"{key[:4]}...{key[-2:]}" if len(key) > 8 else "***"


# ---- environment (real implementations) --------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}"
    return out.returncode, (out.stdout + out.stderr)


def _run_in(cwd: Path, cmd: list[str], env: Mapping[str, str], timeout: float) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, cwd=cwd, env=dict(env), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}"
    return out.returncode, (out.stdout + out.stderr)


def _http_get(url: str, timeout: float = 1.0) -> tuple[int, str] | None:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - loopback URLs only (checked by the callers)
            return resp.status, resp.read(200_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except (OSError, ValueError):
        return None


def _port_state(port: int) -> str:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            pass
    except OSError:
        return "free"
    reply = _http_get(f"http://127.0.0.1:{port}/api/health", 1.0)
    if reply and reply[0] == 200 and '"version"' in reply[1]:
        return "evora"
    return "busy"


def _version_of(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _compute() -> tuple[str, str]:
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch missing or broken both mean "no accelerator we can use"
        return "none", "torch is not installed"
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", "Apple GPU"
    return "cpu", "no GPU found"


def _blur_check() -> tuple[bool, str]:
    try:
        from evora.perception import blur_faces
    except Exception as exc:  # noqa: BLE001 - an import failure is the answer here
        return False, f"face blur could not be imported ({type(exc).__name__})"
    try:
        blur_faces(probe_jpeg())
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200] or type(exc).__name__
    return True, "blur ran on a test image"


def _workspace_check(root: Path) -> tuple[bool, str]:
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix=".doctor-") as tmp:
            db = sqlite3.connect(Path(tmp) / "t.sqlite")
            db.execute("CREATE TABLE t(x)")
            db.execute("INSERT INTO t VALUES (1)")
            db.commit()
            ok = db.execute("SELECT x FROM t").fetchone() == (1,)
            db.close()
        return ok, str(root)
    except (OSError, sqlite3.Error) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _tz_ok() -> bool:
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo("Asia/Kolkata")
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_script(root: Path, name: str) -> Any | None:
    path = root / "scripts" / f"{name}.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"_doctor_{name}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - a broken helper script must not break the doctor
        return None
    return module


def _groq_probe(root: Path, environ: Mapping[str, str], model: str) -> list[dict[str, Any]]:
    """One row per key: valid or not, number of models and remaining requests. Reuses M3's check_groq."""
    mod = _load_script(root, "check_groq")
    if mod is None:
        return [{"key": "?", "ok": False, "note": "scripts/check_groq.py is missing"}]
    import httpx

    keys = mod.resolve_keys(None, dict(environ), mod.load_env_file(root / ".env"))
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=15.0) as client:
        for key in keys:
            models, note = mod.list_models(client, key)
            row: dict[str, Any] = {"key": mask(key), "ok": models is not None, "note": note, "models": len(models or [])}
            if models is not None and model in models:
                headers = mod.probe_model(client, key, model)
                row["remaining_requests"] = headers.get("x-ratelimit-remaining-requests")
                row["status"] = headers.get("status")
            elif models is not None:
                row["note"] = f"{model} is not available to this key"
                row["ok"] = False
            rows.append(row)
    return rows


@dataclass
class Env:
    root: Path
    cfg: dict[str, Any]
    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    on_prem: bool = False
    quick: bool = False
    which: Callable[[str], str | None] = shutil.which
    run: Callable[[list[str], float], tuple[int, str]] = _run
    run_in: Callable[[Path, list[str], Mapping[str, str], float], tuple[int, str]] = _run_in
    http_get: Callable[[str, float], tuple[int, str] | None] = _http_get
    version_of: Callable[[str], str | None] = _version_of
    disk_free_gb: Callable[[Path], float] = lambda p: shutil.disk_usage(p).free / 1e9
    port_state: Callable[[int], str] = _port_state
    compute: Callable[[], tuple[str, str]] = _compute
    blur_check: Callable[[], tuple[bool, str]] = _blur_check
    workspace_check: Callable[[Path], tuple[bool, str]] = _workspace_check
    tz_ok: Callable[[], bool] = _tz_ok
    groq_probe: Callable[[], list[dict[str, Any]]] | None = None
    mediamtx: Callable[[], Path | None] = lambda: find_mediamtx()
    python_version: tuple[int, int] = field(default_factory=lambda: sys.version_info[:2])

    @property
    def models_dir(self) -> Path:
        return Path(self.environ.get("EVORA_MODELS_DIR") or self.root / "models")

    @property
    def workspaces_dir(self) -> Path:
        return self.root / self.cfg.get("workspace", {}).get("root", "workspaces")

    def groq_rows(self) -> list[dict[str, Any]]:
        if self.groq_probe is not None:
            return self.groq_probe()
        planner = self.cfg.get("llm", {}).get("models", {}).get("planner", "openai/gpt-oss-20b")
        return _groq_probe(self.root, self.environ, planner)


# ---- checks ---------------------------------------------------------------------------------------------------

def check_python(env: Env) -> list[Check]:
    out: list[Check] = []
    major_minor = env.python_version
    ok = (3, 11) <= major_minor < (3, 13)
    out.append(Check("python", "Python", OK if ok else FAIL, f"{major_minor[0]}.{major_minor[1]}",
                     "" if ok else "Use Python 3.11 or 3.12 (uv installs it: `make setup`)."))
    missing = [p for p in CORE_PACKAGES if env.version_of(p) is None]
    shown = ", ".join(f"{p} {env.version_of(p)}" for p in CORE_PACKAGES[:4]) + ", ..."
    out.append(Check("packages", "Core packages", FAIL if missing else OK,
                     "missing: " + ", ".join(missing) if missing else shown, "make setup" if missing else ""))
    absent = [p for p in PERCEPTION_PACKAGES if env.version_of(p) is None]
    out.append(Check("perception", "Perception stack", WARN if absent else OK,
                     ("not installed: " + ", ".join(absent) + ". New footage cannot be indexed yet.") if absent
                     else ", ".join(f"{p} {env.version_of(p)}" for p in PERCEPTION_PACKAGES),
                     "make setup-perception" if absent else ""))
    return out


def check_ffmpeg(env: Env) -> list[Check]:
    ffmpeg, ffprobe = env.which("ffmpeg"), env.which("ffprobe")
    if not ffmpeg or not ffprobe:
        return [Check("ffmpeg", "ffmpeg and ffprobe", FAIL, "not found on PATH",
                      "Install ffmpeg (Windows: `winget install Gyan.FFmpeg`, macOS: `brew install ffmpeg`).")]
    _, version = env.run([ffmpeg, "-version"], 8.0)
    _, encoders = env.run([ffmpeg, "-hide_banner", "-encoders"], 8.0)
    _, muxers = env.run([ffmpeg, "-hide_banner", "-muxers"], 8.0)
    gaps = []
    if "libx264" not in encoders:
        gaps.append("libx264")
    if "rtsp" not in muxers:
        gaps.append("rtsp muxer")
    first = version.splitlines()[0] if version else "version unknown"
    if gaps:
        detail = f"{first}; missing {', '.join(gaps)}"
        return [Check("ffmpeg", "ffmpeg and ffprobe", FAIL, detail, "Install a full ffmpeg build.")]
    return [Check("ffmpeg", "ffmpeg and ffprobe", OK, first.replace("ffmpeg version ", "ffmpeg ").split(" Copyright")[0][:70])]


def check_compute(env: Env) -> list[Check]:
    kind, name = env.compute()
    if kind in ("cuda", "mps"):
        return [Check("compute", "Compute", OK, f"{kind}: {name}")]
    return [Check("compute", "Compute", WARN, f"CPU only ({name}). Indexing runs slower; use the cpu profile.",
                  "Set evora_PROFILE=cpu; expect L0 to finish first and L1 progressively.")]


def check_disk(env: Env) -> list[Check]:
    root = env.workspaces_dir
    probe = root if root.exists() else root.parent
    free = env.disk_free_gb(probe)
    want = float(env.cfg.get("doctor", {}).get("min_free_gb", 20))
    status = OK if free >= want else FAIL if free < 2 else WARN
    return [Check("disk", "Free disk", status, f"{free:.0f} GB free next to {root.name}/ (want {want:g})",
                  "" if status == OK else "Free some space before ingesting long footage.")]


def check_workspace(env: Env) -> list[Check]:
    ok, detail = env.workspace_check(env.workspaces_dir)
    return [Check("workspace", "Workspace writable", OK if ok else FAIL, detail if ok else f"cannot write: {detail}",
                  "" if ok else "Check permissions on the workspaces folder.")]


def _models_spec(env: Env) -> dict[str, Any]:
    mod = _load_script(env.root, "models_download")
    return {
        "yolo": list(getattr(mod, "YOLO_WEIGHTS", ["yolo26n.pt"])),
        "siglip": getattr(mod, "SIGLIP2", "google/siglip2-base-patch16-224"),
        "bge": getattr(mod, "BGE", "BAAI/bge-small-en-v1.5"),
        "boxmot": list(getattr(mod, "BOXMOT_WEIGHTS", {"osnet_x0_25_msmt17.pt": ""})),
    }


def _hf_present(env: Env, repo: str) -> bool:
    home = Path(env.environ.get("HF_HOME") or env.models_dir / "hf")
    snaps = home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
    return snaps.is_dir() and any(any(p.iterdir()) for p in snaps.iterdir() if p.is_dir())


def check_models(env: Env) -> list[Check]:
    spec = _models_spec(env)
    md = env.models_dir
    fix = "python scripts/models_download.py --only {}"
    rows: list[Check] = []
    yolo_missing = [n for n in spec["yolo"] if not (md / "ultralytics" / n).is_file()]
    primary = spec["yolo"][0]
    rows.append(Check("m_detector", "Detector weights", FAIL if primary in yolo_missing else (WARN if yolo_missing else OK),
                      "missing: " + ", ".join(yolo_missing) if yolo_missing else f"{len(spec['yolo'])} files",
                      fix.format("yolo") if yolo_missing else ""))
    sig = _hf_present(env, spec["siglip"])
    siglip_detail = spec["siglip"] if sig else f"{spec['siglip']} is not cached"
    rows.append(Check("m_siglip", "SigLIP2 (image and text)", OK if sig else FAIL, siglip_detail,
                      "" if sig else fix.format("siglip2")))
    bge_ok = (md / "fastembed").is_dir() and any((md / "fastembed").iterdir())
    rows.append(Check("m_bge", "Text embedder for aliases", OK if bge_ok else WARN,
                      "fastembed bge-small cached" if bge_ok else "not cached; aliases fall back to a weaker hashing embedder",
                      "" if bge_ok else fix.format("fastembed")))
    reid_missing = [n for n in spec["boxmot"] if not (md / "boxmot" / n).is_file()]
    rows.append(Check("m_reid", "Re-identification weights", WARN if reid_missing else OK,
                      "missing: " + ", ".join(reid_missing) if reid_missing else "present",
                      fix.format("boxmot") if reid_missing else ""))
    face = md / "yunet" / "face_detection_yunet_2023mar.onnx"
    rows.append(Check("m_face", "Face model for blur", OK if face.is_file() else FAIL,
                      "present" if face.is_file() else "missing: faces cannot be blurred, so evidence cannot be exported",
                      "" if face.is_file() else fix.format("yunet")))
    return rows


def check_ollama(env: Env) -> list[Check]:
    host = (env.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    needed_status = FAIL if env.on_prem else WARN
    if not host.startswith(("http://127.", "http://localhost", "http://[::1]")):
        return [Check("ollama", "Ollama", WARN, f"{host} is not on this machine; on-prem mode only allows loopback")]
    reply = env.http_get(f"{host}/api/tags", 1.5)
    if reply is None or reply[0] != 200:
        exe = env.which("ollama")
        why = "installed but not running" if exe else "not installed"
        return [Check("ollama", "Ollama (local language and vision models)", needed_status, why,
                      "Run `ollama serve`" if exe else "Install Ollama from ollama.com, then `make models`.")]
    try:
        have = {m["name"] for m in json.loads(reply[1]).get("models", [])}
    except (ValueError, KeyError, TypeError):
        have = set()
    missing = [role for role, tags in OLLAMA_NEEDED.items() if not any(t in have for t in tags)]
    if missing:
        wanted = [OLLAMA_NEEDED[r][0] for r in missing]
        return [Check("ollama", "Ollama (local language and vision models)", needed_status,
                      f"running; missing {', '.join(missing)} model", "ollama pull " + " && ollama pull ".join(wanted))]
    return [Check("ollama", "Ollama (local language and vision models)", OK, f"running; {len(have)} models")]


def check_groq(env: Env) -> list[Check]:
    if env.on_prem:
        return [Check("groq", "Groq keys", SKIP, "on-prem mode: the cloud is not used")]
    if env.quick:
        return [Check("groq", "Groq keys", SKIP, "skipped in quick mode")]
    rows = env.groq_rows()
    if not rows:
        return [Check("groq", "Groq keys", WARN, "no keys in GROQ_KEYS or .env: planning uses the fast path and the local model",
                      "Add your key to .env (GROQ_KEYS=gsk_...). Each member uses their own account.")]
    bad = [r for r in rows if not r.get("ok")]
    parts = [f"{r['key']}: " + ("ok" + (f", {r['remaining_requests']} requests left" if r.get("remaining_requests") else "")
                                 if r.get("ok") else r.get("note", "rejected")) for r in rows]
    if bad and len(bad) == len(rows):
        return [Check("groq", "Groq keys", WARN, "; ".join(parts), "Check the keys with `python scripts/check_groq.py`.")]
    fix = "Remove or replace the rejected key." if bad else ""
    return [Check("groq", "Groq keys", WARN if bad else OK, "; ".join(parts), fix)]


def check_live(env: Env) -> list[Check]:
    rows: list[Check] = []
    exe = env.mediamtx()
    rows.append(Check("mediamtx", "MediaMTX (replay-as-live)", OK if exe else WARN,
                      exe.name if exe else "not installed: the live demo is unavailable",
                      "" if exe else "winget install bluenviron.mediamtx"))
    api_port = int(env.cfg.get("server", {}).get("port", 8700))
    rtsp_port = int(env.cfg.get("live", {}).get("rtsp_port", 8554))
    for label, port, busy_status in (("API port", api_port, FAIL), ("RTSP port", rtsp_port, WARN)):
        state = env.port_state(port)
        rows.append(Check(f"port_{port}", label, OK if state in ("free", "evora") else busy_status,
                          {"free": f"{port} is free", "evora": f"{port} is already served by evora",
                           "busy": f"{port} is in use by another program"}[state],
                          "" if state != "busy" else "Stop the other program or change the port in config/default.yaml."))
    return rows


def check_blur(env: Env) -> list[Check]:
    ok, detail = env.blur_check()
    return [Check("blur", "Face blur works", OK if ok else FAIL, detail,
                  "" if ok else "python scripts/models_download.py --only yunet")]


def check_privacy(env: Env) -> list[Check]:
    rows = [Check("onprem", "Privacy mode", OK, "on-prem: nothing leaves this machine" if env.on_prem
                  else "cloud allowed; switch on-prem in the UI or with evora_ONPREM=1")]
    has_env = (env.root / ".env").is_file()
    rows.append(Check("envfile", ".env file", OK if has_env else WARN,
                      ".env present" if has_env else "no .env (keys and options come from the shell only)",
                      "" if has_env else "copy .env.example to .env"))
    return rows


def check_tz(env: Env) -> list[Check]:
    ok = env.tz_ok()
    return [Check("tz", "Time zone data (Asia/Kolkata)", OK if ok else FAIL, "available" if ok else "tzdata is missing",
                  "" if ok else "make setup")]


def check_ui(env: Env) -> list[Check]:
    frontend = env.root / "frontend"
    if (frontend / "package.json").is_file():
        have_node = bool(env.which("node") and env.which("npm"))
        built = (frontend / ".next" / "BUILD_ID").is_file()
        if not have_node:
            return [Check("ui", "Web interface", WARN, "Node.js is not installed: `make up` serves the API only",
                          "Install Node 20+ (winget install OpenJS.NodeJS.LTS).")]
        if not (frontend / "node_modules").is_dir():
            return [Check("ui", "Web interface", WARN, "dependencies are not installed", "cd frontend && npm ci")]
        return [Check("ui", "Web interface", OK if built else WARN,
                      "Next.js build is ready" if built else "not built yet: `make up` builds it on first start",
                      "" if built else "cd frontend && npm run build")]
    ui = env.root / env.cfg.get("server", {}).get("ui_dir", "frontend/dist") / "index.html"
    return [Check("ui", "Web interface", OK if ui.is_file() else WARN,
                  "static build is ready" if ui.is_file() else "no frontend yet: `make up` serves the API only",
                  "" if ui.is_file() else "build the frontend")]


CHECKS: list[tuple[str, Callable[[Env], list[Check]], bool]] = [  # (group, function, needs the network)
    ("python", check_python, False), ("ffmpeg", check_ffmpeg, False), ("compute", check_compute, False),
    ("disk", check_disk, False), ("workspace", check_workspace, False), ("models", check_models, False),
    ("ollama", check_ollama, False), ("groq", check_groq, True), ("live", check_live, False),
    ("blur", check_blur, False), ("privacy", check_privacy, False), ("tz", check_tz, False), ("ui", check_ui, False),
]


def run_checks(env: Env, skip: set[str] | frozenset[str] = frozenset()) -> list[Check]:
    results: list[Check] = []
    for group, fn, _network in CHECKS:
        if group in skip:
            results.append(Check(group, group, SKIP, "skipped by request"))
            continue
        try:
            results.extend(fn(env))
        except Exception as exc:  # noqa: BLE001 - one broken check must never take the doctor down
            results.append(Check(group, group, FAIL, f"check crashed: {type(exc).__name__}: {str(exc)[:120]}"))
    return results


# ---- output --------------------------------------------------------------------------------------------------

MARK = {OK: "ok", WARN: "WARN", FAIL: "FAIL", SKIP: "skip"}
COLOR = {OK: "32", WARN: "33", FAIL: "31", SKIP: "90"}


def verdict(checks: list[Check]) -> str:
    fails = sum(c.status == FAIL for c in checks)
    warns = sum(c.status == WARN for c in checks)
    if fails:
        text = f"{fails} problem{'s' if fails != 1 else ''} to fix"
        return text + (f", {warns} warning{'s' if warns != 1 else ''}" if warns else "")
    if warns:
        return f"{warns} warning{'s' if warns != 1 else ''}, nothing blocking"
    return "Demo-ready"


def render(checks: list[Check], color: bool = False) -> str:
    width = max(len(c.title) for c in checks)
    lines = []
    for c in checks:
        mark = MARK[c.status]
        tag = f"\033[{COLOR[c.status]}m{mark:<4}\033[0m" if color else f"{mark:<4}"
        lines.append(f"{tag}  {c.title:<{width}}  {c.detail}")
        if c.fix and c.status in (WARN, FAIL):
            lines.append(f"{'':<4}  {'':<{width}}  -> {c.fix}")
    lines.append("")
    lines.append(verdict(checks))
    return "\n".join(lines)


def to_json(checks: list[Check]) -> str:
    return json.dumps({"verdict": verdict(checks), "ok": not any(c.status == FAIL for c in checks),
                       "checks": [asdict(c) for c in checks]}, indent=2)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from evora.core.config import REPO_ROOT, load_config

    p = argparse.ArgumentParser(description="Check that this machine is ready to demo.")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--quick", action="store_true", help="skip checks that need the network")
    p.add_argument("--on-prem", action="store_true", help="check for on-prem mode (local models required, cloud skipped)")
    p.add_argument("--skip", default="", help="comma separated groups to skip: " + ",".join(g for g, _, _ in CHECKS))
    args = p.parse_args(argv)
    on_prem = args.on_prem or os.environ.get("evora_ONPREM") == "1"
    env = Env(root=REPO_ROOT, cfg=load_config(), on_prem=on_prem, quick=args.quick)
    checks = run_checks(env, {s.strip() for s in args.skip.split(",") if s.strip()})
    print(to_json(checks) if args.json else render(checks, color=sys.stdout.isatty()))
    return 1 if any(c.status == FAIL for c in checks) else 0
