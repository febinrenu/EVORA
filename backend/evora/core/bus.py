"""In-process event bus. Publishers may be any thread; subscribers are asyncio consumers (SSE routes)."""
from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from typing import Any


class Subscription:
    def __init__(self, loop: asyncio.AbstractEventLoop, maxsize: int):
        self.loop = loop
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)

    def _offer(self, msg: dict[str, Any]) -> None:
        if self.queue.full():  # slow consumer: drop the oldest message rather than block the publisher
            self.queue.get_nowait()
        self.queue.put_nowait(msg)


class Bus:
    def __init__(self, queue_size: int = 256):
        self._subs: set[Subscription] = set()
        self._lock = threading.Lock()
        self._queue_size = queue_size

    def subscribe(self) -> Subscription:
        sub = Subscription(asyncio.get_running_loop(), self._queue_size)
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subs.discard(sub)

    def publish(self, kind: str, data: dict[str, Any]) -> None:
        """Publish a `note` event whose payload carries its own `kind` (ingest, camera, alert, ...)."""
        msg = {"event": "note", "data": json.dumps({"kind": kind, **data})}
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            try:
                sub.loop.call_soon_threadsafe(sub._offer, msg)
            except RuntimeError:  # the subscriber's loop is closed
                self.unsubscribe(sub)

    async def stream(self, heartbeat_s: float = 15.0) -> AsyncIterator[dict[str, Any]]:
        """SSE message iterator: bus events as they arrive, a heartbeat note when idle."""
        sub = self.subscribe()
        n = 0
        try:
            while True:
                try:
                    yield await asyncio.wait_for(sub.queue.get(), timeout=heartbeat_s)
                except TimeoutError:
                    yield {"event": "note", "data": json.dumps({"kind": "heartbeat", "n": n})}
                    n += 1
        finally:
            self.unsubscribe(sub)
