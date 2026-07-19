from __future__ import annotations

import asyncio

from .service import FullSystemIntegrationService


class IntegrationWorker:
    """Single-purpose bounded outbox publisher; deployment coordination is platform-owned."""

    def __init__(self, service: FullSystemIntegrationService) -> None:
        self.service = service
        self._stop = asyncio.Event()

    async def run(self) -> None:
        while not self._stop.is_set():
            processed = await self.service.process_outbox_once()
            if processed == 0:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.service.config.limits.outbox_poll_seconds)
                except TimeoutError:
                    pass

    def stop(self) -> None:
        self._stop.set()
