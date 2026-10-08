"""User-changeable settings (on-prem, face blur, reference clock), validated and kept in the workspace `meta` table."""
from __future__ import annotations

import json
from typing import Any

from evora.core.db import Database

KEYS = ("onprem", "blur_faces", "reference_now")


class SettingsError(ValueError):
    pass


def validate(changes: dict[str, Any]) -> dict[str, Any]:
    """Keep known keys, reject wrong types. Unknown keys are ignored (the contract allows extra client fields)."""
    out: dict[str, Any] = {}
    for key in KEYS:
        if key not in changes:
            continue
        value = changes[key]
        if key in ("onprem", "blur_faces"):
            if not isinstance(value, bool):
                raise SettingsError(f"{key} must be true or false")
        elif value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
            raise SettingsError("reference_now must be epoch seconds or null")
        out[key] = value
    return out


def load(db: Database, defaults: dict[str, Any], force_onprem: bool = False) -> dict[str, Any]:
    """Defaults, overridden by what was saved. `force_onprem` (evora_ONPREM=1) always starts the app on-prem."""
    settings = dict(defaults)
    for key in KEYS:
        raw = db.get_meta(f"setting.{key}")
        if raw is not None:
            try:
                settings[key] = json.loads(raw)
            except json.JSONDecodeError:
                continue
    if force_onprem:
        settings["onprem"] = True
    return settings


def save(db: Database, changes: dict[str, Any]) -> None:
    for key, value in changes.items():
        db.set_meta(f"setting.{key}", json.dumps(value))
