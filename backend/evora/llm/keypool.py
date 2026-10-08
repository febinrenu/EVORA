"""Per-(key, model) rate-limit bookkeeping for the Groq key pool.

Limits apply per organization, so each member contributes a key from their own
account. The pool tracks remaining tokens from response headers, backs off a key
that returned 429, and opens a circuit after repeated failures so the gateway can
route to the local model instead.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

DEFAULT_TOKEN_BUDGET = 8000.0
CIRCUIT_FAILURES = 3
CIRCUIT_OPEN_S = 60.0

_NUMBERED = re.compile(r"^GROQ_KEY[-_]?(\d+)$")
_DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)(ms|h|m|s))")


def parse_duration(text: str | None) -> float | None:
    """Parse Groq style durations such as '7.66s', '2m59.56s', '120ms', '1h2m3s'."""
    if not text:
        return None
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    parts = _DURATION.findall(text)
    if not parts or "".join(v + u for v, u in parts) != text:
        return None
    return sum(float(v) * units[u] for v, u in parts)


def mask_key(key: str) -> str:
    return f"...{key[-4:]}" if len(key) > 4 else "..."


@dataclass
class _State:
    remaining_tokens: float | None = None
    refill_at: float = 0.0
    blocked_until: float = 0.0
    failures: int = 0
    circuit_until: float = 0.0


class KeyPool:
    def __init__(self, keys: list[str], clock: Callable[[], float] = time.monotonic) -> None:
        self._keys = [k.strip() for k in keys if k.strip()]
        self._clock = clock
        self._state: dict[tuple[int, str], _State] = {}

    @classmethod
    def from_env(cls, env: Mapping[str, str], clock: Callable[[], float] = time.monotonic) -> KeyPool:
        """GROQ_KEYS (comma list) plus GROQ_KEY-1, GROQ_KEY_2 ... in numeric order, duplicates dropped."""
        keys = [k.strip() for k in (env.get("GROQ_KEYS") or "").split(",") if k.strip()]
        numbered = sorted((int(m.group(1)), name) for name in env if (m := _NUMBERED.match(name)))
        keys += [env[name].strip() for _, name in numbered if env[name].strip()]
        return cls(list(dict.fromkeys(keys)), clock)

    @classmethod
    def from_env_value(cls, value: str | None, clock: Callable[[], float] = time.monotonic) -> KeyPool:
        return cls((value or "").split(","), clock)

    def __len__(self) -> int:
        return len(self._keys)

    def key(self, idx: int) -> str:
        return self._keys[idx]

    def label(self, idx: int) -> str:
        return mask_key(self._keys[idx])

    def _st(self, idx: int, model: str) -> _State:
        return self._state.setdefault((idx, model), _State())

    def headroom(self, idx: int, model: str) -> float:
        st = self._st(idx, model)
        if st.remaining_tokens is None:
            return DEFAULT_TOKEN_BUDGET
        if self._clock() >= st.refill_at:
            return DEFAULT_TOKEN_BUDGET
        return st.remaining_tokens

    def available(self, idx: int, model: str) -> bool:
        st = self._st(idx, model)
        now = self._clock()
        return now >= st.blocked_until and now >= st.circuit_until

    def pick(self, model: str, need_tokens: int = 0) -> int | None:
        """Key index with the most headroom that can take `need_tokens`, else None."""
        best: tuple[float, int] | None = None
        for idx in range(len(self._keys)):
            if not self.available(idx, model):
                continue
            room = self.headroom(idx, model)
            if room < need_tokens:
                continue
            if best is None or room > best[0]:
                best = (room, idx)
        return None if best is None else best[1]

    def update_from_headers(self, idx: int, model: str, headers: Mapping[str, str]) -> None:
        low = {k.lower(): v for k, v in headers.items()}
        st = self._st(idx, model)
        remaining = low.get("x-ratelimit-remaining-tokens")
        if remaining is not None:
            try:
                st.remaining_tokens = float(remaining)
            except ValueError:
                pass
        reset = parse_duration(low.get("x-ratelimit-reset-tokens"))
        if reset is not None:
            st.refill_at = self._clock() + reset

    def record_success(self, idx: int, model: str) -> None:
        self._st(idx, model).failures = 0

    def record_rate_limited(self, idx: int, model: str, retry_after: float | None) -> None:
        st = self._st(idx, model)
        st.blocked_until = self._clock() + (retry_after if retry_after is not None else 5.0)

    def record_failure(self, idx: int, model: str) -> None:
        st = self._st(idx, model)
        st.failures += 1
        if st.failures >= CIRCUIT_FAILURES:
            st.circuit_until = self._clock() + CIRCUIT_OPEN_S
            st.failures = 0

    def all_unavailable(self, model: str) -> bool:
        return all(not self.available(i, model) for i in range(len(self._keys)))
