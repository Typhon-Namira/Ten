from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import logging

from .service import FullSystemIntegrationService

logger = logging.getLogger(__name__)


class IntegrationWorker:
    """Single-purpose bounded outbox publisher; deployment coordination is platform-owned."""

    def __init__(self, service: FullSystemIntegrationService) -> None:
        self.service = service
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.last_heartbeat_at: datetime | None = None
        self.last_success_at: datetime | None = None
        self.last_error: str | None = None
        self.consecutive_failures = 0

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self.run(), name="ten-integration-worker")

    async def run(self) -> None:
        while not self._stop.is_set():
            self.last_heartbeat_at = datetime.now(UTC)
            logger.info("worker.heartbeat", extra={"worker": "integration"})
            try:
                processed = await self.service.process_outbox_once()
                self.last_success_at = datetime.now(UTC)
                self.last_error = None
                self.consecutive_failures = 0
            except Exception as exc:
                processed = 0
                self.last_error = type(exc).__name__
                self.consecutive_failures += 1
                logger.exception("integration.worker.failed", extra={"error_type": self.last_error})
            if processed == 0:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.service.config.limits.outbox_poll_seconds)
                except TimeoutError:
                    pass

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    def status(self, enabled: bool) -> dict[str, object]:
        return {
            "enabled": enabled,
            "running": self._task is not None and not self._task.done(),
            "last_heartbeat_at": self.last_heartbeat_at,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "consecutive_failures": self.consecutive_failures,
        }
