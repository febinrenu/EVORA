"""Fixture loader: the skeleton API serves these until the real services land."""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "contracts" / "fixtures"


@cache
def load(name: str):
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
