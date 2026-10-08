"""SSE helpers shared by the query, clarify and events routes."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable

from contracts.models import StreamEvent
from sse_starlette.sse import EventSourceResponse


def encode(ev: StreamEvent | dict) -> dict:
    """Shape one StreamEvent as an sse-starlette message: event name = type, data = JSON."""
    model = ev if isinstance(ev, StreamEvent) else StreamEvent.model_validate(ev)
    return {"event": model.type, "data": json.dumps(model.data)}


async def _replay(events: Iterable[dict], delay_s: float) -> AsyncIterator[dict]:
    for ev in events:
        yield encode(ev)
        if delay_s:
            await asyncio.sleep(delay_s)


def stream_events(events: Iterable[dict], delay_s: float = 0.0) -> EventSourceResponse:
    return EventSourceResponse(_replay(list(events), delay_s))


async def heartbeat(interval_s: float, limit: int | None = None) -> AsyncIterator[dict]:
    n = 0
    while limit is None or n < limit:
        yield {"event": "note", "data": json.dumps({"heartbeat": n})}
        n += 1
        await asyncio.sleep(interval_s)
