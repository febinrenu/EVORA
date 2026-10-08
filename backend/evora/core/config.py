"""Configuration loader: config/default.yaml, overridden by a profile and by environment."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


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
