from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
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
        # Set only if the task itself terminated with an exception `run()`'s own try/except could
        # not absorb — see `MarketDataWorker.last_fatal_error` for the full rationale. `run()`
        # below is already resilient to any per-cycle `Exception`, so this is primarily a backstop
        # against future changes reintroducing an unguarded path, and against a genuine
        # `BaseException` the loop deliberately does not swallow.
        self.last_fatal_error: str | None = None
        self._next_waiting_recovery_at: datetime | None = None

    async def _recover_waiting_simulations(self) -> None:
        now = datetime.now(UTC)
        if (
            self._next_waiting_recovery_at is not None
            and now < self._next_waiting_recovery_at
        ):
            return
        self._next_waiting_recovery_at = now + timedelta(seconds=10)
        simulation = getattr(self.service, "market_simulation", None)
        if simulation is None:
            return
        for configured in self.service.config.instruments:
            attempt = await simulation.repository.latest_attempt(
                configured.instrument_id
            )
            if (
                attempt is None
                or attempt.status.value != "WAITING_FOR_AI_ANALYSIS"
            ):
                continue
            reasoning = getattr(self.service, "ai_reasoning", None)
            claim = (
                await reasoning.repository.claim_for_cutoff(
                    configured.instrument_id, attempt.market_cutoff
                )
                if reasoning is not None
                else None
            )
            if claim is not None and claim.status == "FAILED_SCHEMA":
                await simulation.record_blocked_cutoff(
                    instrument=configured.instrument_id,
                    market_cutoff=attempt.market_cutoff,
                    server_time=now,
                    reason="AI_SCHEMA_VALIDATION_FAILED",
                    market_state_id=attempt.market_state_id,
                    quantitative_forecast_id=attempt.quantitative_forecast_id,
                    correlation_id=attempt.correlation_id,
                )
                continue
            if (
                claim is not None
                and claim.status == "WAITING_PROVIDER"
                and claim.next_retry_at is not None
                and now < claim.next_retry_at
            ):
                logger.info(
                    "ai_reasoning.recovery.deferred",
                    extra={
                        "instrument": configured.instrument_id,
                        "market_cutoff": attempt.market_cutoff.isoformat(),
                        "claim_id": str(claim.claim_id),
                        "next_retry_at": claim.next_retry_at.isoformat(),
                        "remaining_seconds": (
                            claim.next_retry_at - now
                        ).total_seconds(),
                    },
                )
                continue
            logger.info(
                "market_simulation.waiting.recovery.started",
                extra={
                    "instrument": configured.instrument_id,
                    "market_cutoff": attempt.market_cutoff.isoformat(),
                    "scenario_attempt_id": str(attempt.attempt_id),
                    "retry_count": attempt.retry_count,
                },
            )
            await self.service.recover_authoritative_ai_analysis(
                configured.instrument_id
            )
            await simulation.recover_latest(configured.instrument_id, now=now)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self.run(), name="ten-integration-worker")
            self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.last_fatal_error = f"{type(exc).__name__}: {exc}"
            logger.critical("integration.worker.task_died", exc_info=exc, extra={"worker": "integration", "error_type": type(exc).__name__})

    async def run(self) -> None:
        simulation = getattr(self.service, "market_simulation", None)
        if hasattr(self.service, "recover_authoritative_ai_analysis"):
            for configured in self.service.config.instruments:
                try:
                    await self.service.recover_authoritative_ai_analysis(
                        configured.instrument_id
                    )
                except Exception as exc:
                    logger.exception(
                        "ai_reasoning.recovery.failed",
                        extra={
                            "instrument": configured.instrument_id,
                            "error_type": type(exc).__name__,
                        },
                    )
        if simulation is not None and hasattr(simulation, "recover_latest"):
            for configured in self.service.config.instruments:
                try:
                    await simulation.recover_latest(configured.instrument_id)
                except Exception as exc:
                    logger.exception(
                        "market_simulation.recovery.failed",
                        extra={
                            "instrument": configured.instrument_id,
                            "error_type": type(exc).__name__,
                        },
                    )
        while not self._stop.is_set():
            self.last_heartbeat_at = datetime.now(UTC)
            logger.info("worker.heartbeat", extra={"worker": "integration"})
            try:
                processed = await self.service.process_outbox_once()
                await self._recover_waiting_simulations()
                if self.service.last_batch_failures:
                    self.last_error = "IntegrationBatchItemFailed"
                    self.consecutive_failures += self.service.last_batch_failures
                else:
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
        running = self._task is not None and not self._task.done()
        backlog = self.service.repository.metrics().get("outbox_backlog", 0)
        progressing = running and not self.last_error and (not backlog or self.last_success_at is not None)
        return {
            "enabled": enabled,
            "running": running,
            # True only when this worker is configured on but its task is not actually running —
            # the exact "looks enabled, silently did nothing" state that used to report healthy.
            "crashed": enabled and not running,
            "progressing": progressing,
            "stalled": enabled and running and not progressing,
            "last_heartbeat_at": self.last_heartbeat_at,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "last_fatal_error": self.last_fatal_error,
            "consecutive_failures": self.consecutive_failures,
        }
