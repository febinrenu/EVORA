"""Record and replay of structured model calls.

Eval reruns and ablations ask the same questions many times. With a replay file the
second run costs zero Groq calls and returns the same plans, which also makes the
numbers comparable between runs.

Modes: "record" always calls live and saves; "replay" never calls live (a miss is an
error, so a run is provably free); "auto" replays hits and records misses.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

log = logging.getLogger("evora.llm.replay")

Mode = Literal["record", "replay", "auto"]


class ReplayStore:
    def __init__(self, path: Path, mode: Mode) -> None:
        if mode not in ("record", "replay", "auto"):
            raise ValueError(f"unknown replay mode {mode!r}")
        self.path = path
        self.mode = mode
        self._entries: dict[str, dict[str, str]] = {}
        if mode != "record" and path.is_file():
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                try:
                    row = json.loads(line)
                    self._entries[row["key"]] = {"text": row["text"], "backend": row["backend"]}
                except (json.JSONDecodeError, KeyError):
                    log.warning("%s line %d is not a replay entry; skipped", path, n)

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def key(task: str, messages: list[dict], schema_name: str) -> str:
        blob = json.dumps([task, schema_name, messages], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def reads(self) -> bool:
        return self.mode in ("replay", "auto")

    @property
    def writes(self) -> bool:
        return self.mode in ("record", "auto")

    def get(self, key: str) -> dict[str, str] | None:
        return self._entries.get(key) if self.reads else None

    def put(self, key: str, task: str, text: str, backend: str) -> None:
        if not self.writes:
            return
        self._entries[key] = {"text": text, "backend": backend}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "task": task, "backend": backend, "text": text}) + "\n")
        except OSError as exc:
            log.warning("could not write %s: %s", self.path, exc)
