"""The local vision model, as perception sees it: `describe(image_jpeg, prompt) -> str | None`.

Perception never opens a network connection itself (repo rule 9). The platform registers a client at start-up that
forwards to the LLM gateway's local-only vision call, running on the application's own event loop:

    from evora.perception import vision
    vision.register(vision.LoopVisionClient(gateway.vision_text, loop))

Until a client is registered, `get_vision_client()` returns None and the features that need one (burned-in clock
reading, L3 captions) are skipped with a clear log line; nothing else changes.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Protocol

log = logging.getLogger("evora.perception.vision")


class VisionClient(Protocol):
    def describe(self, image_jpeg: bytes, prompt: str, *, max_tokens: int = 64) -> str | None:
        """A short text answer about one image, or None when the model is unavailable or fails."""


_lock = threading.Lock()
_client: VisionClient | None = None


def register(client: VisionClient | None) -> None:
    """Install (or, with None, remove) the process-wide vision client."""
    global _client
    with _lock:
        _client = client


def get_vision_client() -> VisionClient | None:
    with _lock:
        return _client


class LoopVisionClient:
    """Calls an async gateway function from an ingest thread by running it on the loop that owns the gateway.

    `call(image_jpeg, prompt, local_only=True, max_tokens=n)` must return the model's text or None.
    The images are pictures of people, so the request is always made local-only.
    """

    def __init__(self, call: Callable[..., Awaitable[str | None]], loop: asyncio.AbstractEventLoop, timeout_s: float = 60.0):
        self._call, self._loop, self._timeout = call, loop, timeout_s

    def describe(self, image_jpeg: bytes, prompt: str, *, max_tokens: int = 64) -> str | None:
        future = asyncio.run_coroutine_threadsafe(
            self._call(image_jpeg, prompt, local_only=True, max_tokens=max_tokens), self._loop)
        try:
            return future.result(timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001 - a vision failure must never break ingestion
            future.cancel()
            log.warning("vision call failed: %s", exc)
            return None
