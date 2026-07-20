"""Server-Sent Events bridge from the in-process event bus to the browser."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.app.integration.activity_log import ActivityLogEntry

router = APIRouter(prefix="/stream", tags=["stream"])

_HEARTBEAT_SECONDS = 15.0


def _format(entry: ActivityLogEntry) -> bytes:
    return f"data: {json.dumps(entry.as_dict(), default=str)}\n\n".encode()


@router.get("/events")
async def stream_events(request: Request, backlog: int = 100) -> StreamingResponse:
    activity_log = request.app.state.pipeline_activity_log

    async def generator() -> AsyncIterator[bytes]:
        key, queue = activity_log.subscribe()
        try:
            yield b": connected\n\n"
            for entry in activity_log.snapshot(backlog):
                yield _format(entry)
            while True:
                if await request.is_disconnected():
                    break
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_SECONDS)
                except TimeoutError:
                    yield b": heartbeat\n\n"
                    continue
                yield _format(entry)
        finally:
            activity_log.unsubscribe(key)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
