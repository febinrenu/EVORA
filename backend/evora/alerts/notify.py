"""Phone push through ntfy. The network call itself lives in the gateway (rule 9); this only decides whether to send."""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger("evora.alerts.notify")

Submit = Callable[[Awaitable[Any]], None]


class Notifier:
    def __init__(
        self, gateway: Any, onprem: Callable[[], bool], submit: Submit | None = None,
        topic: Callable[[], str | None] = lambda: os.environ.get("NTFY_TOPIC") or None,
    ) -> None:
        self._gateway, self._onprem, self._topic = gateway, onprem, topic
        self._submit = submit or self._submit_on_loop
        self.loop: asyncio.AbstractEventLoop | None = None  # set when the server starts
        self._warned: set[str] = set()

    def _hint(self, key: str, message: str) -> None:
        if key not in self._warned:  # one line per reason, never one per alert
            self._warned.add(key)
            log.info(message)

    def push(self, title: str, message: str) -> bool:
        """Send a text-only notification. Returns True when a send was started."""
        topic = self._topic()
        if not topic:
            return False
        if self._onprem():
            self._hint("onprem", "ntfy push skipped: on-prem mode keeps everything on this machine")
            return False
        notify = getattr(self._gateway, "notify", None)
        if notify is None:
            self._hint("nogateway", "ntfy push skipped: the gateway has no notify method yet (requested from M3)")
            return False
        self._submit(notify(topic, title, message))
        return True

    def _submit_on_loop(self, coro: Awaitable[Any]) -> None:
        if self.loop is None or self.loop.is_closed():
            self._hint("noloop", "ntfy push skipped: the server loop is not running")
            close = getattr(coro, "close", None)
            if close:
                close()
            return
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)  # type: ignore[arg-type]
        future.add_done_callback(self._log_failure)

    @staticmethod
    def _log_failure(future) -> None:  # noqa: ANN001
        exc = future.exception()
        if exc is not None:
            log.warning("ntfy push failed: %s", exc.__class__.__name__)  # the topic and the message stay out of logs
