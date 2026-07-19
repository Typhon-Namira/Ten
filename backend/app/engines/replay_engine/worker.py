from __future__ import annotations

import asyncio
from uuid import uuid4

from .config import ReplayConfig
from .models import ReplayStatus
from .service import ReplayService


class ReplayWorker:
    """Database-lease worker; no process-local queue is authoritative."""

    def __init__(self, service: ReplayService, config: ReplayConfig, worker_id: str | None = None) -> None:
        self.service = service
        self.config = config
        self.worker_id = worker_id or f"replay-worker-{uuid4()}"
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def run_once(self) -> int:
        sessions = await self.service.list(0, self.config.worker.max_concurrency, ReplayStatus.RUNNING)
        processed = 0
        for session in sessions:
            if self._stop.is_set():
                break
            try:
                await self.service.run(session.replay_id, self.worker_id)
                processed += 1
            except Exception:
                continue
        return processed

    async def serve(self) -> None:
        while not self._stop.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.config.worker.poll_interval_seconds)
            except TimeoutError:
                pass

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self.serve(), name="ten-replay-worker")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
