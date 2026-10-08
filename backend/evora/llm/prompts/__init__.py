"""Prompt assembly for the query planner.

The system message is built once and never varies between calls (rules, schema,
examples), so the provider's prefix cache can serve it. Everything that changes per
request, the camera list and the question, goes in the user message after it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from evora._contracts_stub import QueryPlan

_DIR = Path(__file__).parent
EXAMPLE_CAMERAS = [
    {"id": "cam_01", "name": "Gate"},
    {"id": "cam_02", "name": "Lobby"},
    {"id": "cam_03", "name": "Back Door"},
    {"id": "cam_04", "name": "Parking"},
]


def _compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def load_examples() -> list[dict]:
    return json.loads((_DIR / "planner_examples.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def planner_system_prompt() -> str:
    rules = (_DIR / "planner_system.txt").read_text(encoding="utf-8").strip()
    schema = _compact(QueryPlan.model_json_schema())
    shots = "\n".join(
        f"QUESTION: {_compact(ex['question'])}\nPLAN: {_compact(ex['plan'])}" for ex in load_examples()
    )
    return (
        f"{rules}\n\nSCHEMA: {schema}\n\n"
        f"EXAMPLES (in all of them CONTEXT is cameras = {_compact(EXAMPLE_CAMERAS)}):\n{shots}"
    )


def build_planner_messages(cameras: list[dict[str, str]], question: str) -> list[dict]:
    """`cameras` is a list of {"id": ..., "name": ...}."""
    slim = [{"id": c["id"], "name": c["name"]} for c in cameras]
    return [
        {"role": "system", "content": planner_system_prompt()},
        {"role": "user", "content": f"CONTEXT: cameras = {_compact(slim)}\nQUESTION: {_compact(question)}"},
    ]
