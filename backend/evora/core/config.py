"""Configuration loader: config/default.yaml, overridden by a profile and by environment."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
_PATH_KEYS = {"HF_HOME"}


def load_env_file(path: Path | None = None, environ: dict[str, str] | None = None) -> int:
    """Load KEY=VALUE lines from `.env` into the environment without overriding existing variables.

    Returns how many variables were set. Values are never logged.
    """
    target = os.environ if environ is None else environ
    env_path = path or REPO_ROOT / ".env"
    if not env_path.is_file():
        return 0
    count = 0
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1] and len(value) >= 2:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key in _PATH_KEYS and value and not Path(value).is_absolute():
            value = str(REPO_ROOT / value)  # ./models/hf means the repo's folder, not wherever the server was started
        if key and key not in target:
            target[key] = value
            count += 1
    return count


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(profile: str | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))
    name = profile or os.environ.get("evora_PROFILE", "cpu")
    prof = CONFIG_DIR / "profiles" / f"{name}.yaml"
    if prof.exists():
        cfg = _merge(cfg, yaml.safe_load(prof.read_text(encoding="utf-8")) or {})
    if os.environ.get("evora_ONPREM") == "1":
        cfg["llm"]["onprem"] = True
    return cfg
