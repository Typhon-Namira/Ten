from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import logging
from collections.abc import Awaitable, Callable

from .models import Candle, Timeframe
from .service import MarketDataService

logger = logging.getLogger(__name__)


class MarketDataWorker:
    """Bounded bootstrap and polling worker hosted by the production ASGI lifespan."""

    def __init__(
        self,
        service: MarketDataService,
        *,
        enabled: bool,
        symbols: tuple[str, ...],
        timeframes: tuple[Timeframe, ...],
        bootstrap_enabled: bool,
        bootstrap_candles: int,
        poll_seconds: float,
        historical_analysis: Callable[[Candle], Awaitable[None]] | None = None,
    ) -> None:
        self.service = service
        self.enabled = enabled
        self.symbols = symbols
        self.timeframes = timeframes
        self.bootstrap_enabled = bootstrap_enabled
        self.bootstrap_candles = bootstrap_candles
        self.poll_seconds = poll_seconds
        self.historical_analysis = historical_analysis
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.last_heartbeat_at: datetime | None = None
        self.last_success_at: datetime | None = None
        self.last_error: str | None = None
        self.consecutive_failures = 0
        self.processing_state = "disabled" if not enabled else "idle"
        self.loaded_candles = 0
        # Distinct from `last_error` (a single poll/bootstrap failure the loop already recovered
        # from) — this is set only if the task itself terminated with an exception the loop could
        # not absorb, e.g. a bug outside the per-iteration try/except below. `status()`/`system.py`
        # must be able to tell "the loop is alive and merely had a bad poll" apart from "the loop
        # is dead and nothing will ever run again" — conflating the two is what let a dead worker
        # report as healthy.
        self.last_fatal_error: str | None = None

    def start(self) -> None:
        if self.enabled and (self._task is None or self._task.done()):
            self._stop.clear()
            self._task = asyncio.create_task(self.run(), name="ten-market-data-worker")
            self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        """Belt-and-suspenders visibility on top of `run()`'s own try/except below: if the task
        ever ends with an exception anyway (a bug in future code, or a `BaseException` the loop
        deliberately doesn't swallow, e.g. cancellation), this guarantees it is logged loudly
        instead of silently vanishing as an unretrieved `asyncio.Task` exception — the exact
        failure mode `asyncio.create_task(...)` invites when nothing ever awaits or inspects the
        task it returns."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.last_fatal_error = f"{type(exc).__name__}: {exc}"
            logger.critical("market_data.worker.task_died", exc_info=exc, extra={"worker": "market_data", "error_type": type(exc).__name__})

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def run(self) -> None:
        if self.bootstrap_enabled:
            try:
                await self._bootstrap()
            except Exception as exc:
                # `_bootstrap()`'s own per-(symbol, timeframe) loop already isolates individual
                # failures; this only fires if something outside that boundary breaks. Bootstrap
                # failing must never prevent the recurring poll loop below from ever starting.
                logger.exception("market_data.bootstrap.crashed", extra={"worker": "market_data", "error_type": type(exc).__name__})
        while not self._stop.is_set():
            try:
                self.last_heartbeat_at = datetime.now(UTC)
                schedule = self.service.sessions.status_at(self.last_heartbeat_at)
                logger.info("worker.heartbeat", extra={"worker": "market_data", "processing_state": self.processing_state})
                if not schedule.market_open:
                    self.processing_state = "market_closed"
                    market_status = getattr(schedule, "market_status", "closed")
                    logger.info("market_data.poll.skipped_market_closed", extra={"market_status": getattr(market_status, "value", market_status)})
                else:
                    self.processing_state = "polling"
                    for symbol in self.symbols:
                        for timeframe in self.timeframes:
                            await self._poll(symbol, timeframe)
            except Exception as exc:
                # `_poll()` below already catches its own exceptions (a single provider failure
                # must not stop other (symbol, timeframe) pairs from being polled) — reaching this
                # branch means something OUTSIDE that per-poll boundary broke instead, e.g.
                # `sessions.status_at()`. Before this fix, that exception propagated straight out of
                # `run()`, silently ending the `asyncio.Task` this loop runs on forever: nothing
                # awaits or inspects that task (see `start()`), so the failure was invisible, the
                # worker never polled again, `NewCandle` never fired again, and
                # `status()["enabled"]` kept reporting `True` since it only reflects static config,
                # not whether the loop is still alive. This closes one concrete, demonstrated way a
                # dead worker could silently masquerade as healthy — see `_on_task_done` and
                # `system.py`'s `operational_state` computation for the other two layers of the
                # same fix.
                self.consecutive_failures += 1
                self.last_error = type(exc).__name__
                logger.exception("market_data.worker.iteration_failed", extra={"worker": "market_data", "error_type": self.last_error})
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass

    async def _bootstrap(self) -> None:
        self.processing_state = "bootstrapping"
        logger.info(
            "market_data.bootstrap.started",
            extra={"symbols": self.symbols, "timeframes": tuple(item.value for item in self.timeframes), "required_candles": self.bootstrap_candles},
        )
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    candles = await self.service.history(symbol, timeframe, limit=self.bootstrap_candles, refresh=True)
                    self.loaded_candles += len(candles)
                    self.last_success_at = datetime.now(UTC)
                    self.last_error = None
                    self.consecutive_failures = 0
                    logger.info(
                        "market_data.bootstrap.completed",
                        extra={"symbol": symbol, "timeframe": timeframe.value, "candle_count": len(candles), "latest_candle_at": candles[-1].timestamp.isoformat() if candles else None},
                    )
                    if candles and self.historical_analysis is not None:
                        await self.historical_analysis(candles[-1])
                except Exception as exc:
                    self._failed(exc, "market_data.bootstrap.failed", symbol, timeframe)
        self.processing_state = "idle"

    async def _poll(self, symbol: str, timeframe: Timeframe) -> None:
        try:
            candle = await self.service.latest(symbol, timeframe, refresh=True)
            self.last_success_at = datetime.now(UTC)
            self.last_error = None
            self.consecutive_failures = 0
            logger.info(
                "market_data.poll.success",
                extra={"symbol": symbol, "timeframe": timeframe.value, "candle_timestamp": candle.timestamp.isoformat() if candle else None},
            )
        except Exception as exc:
            self._failed(exc, "market_data.poll.failed", symbol, timeframe)

    def _failed(self, exc: Exception, event: str, symbol: str, timeframe: Timeframe) -> None:
        self.consecutive_failures += 1
        self.last_error = type(exc).__name__
        logger.warning(event, extra={"symbol": symbol, "timeframe": timeframe.value, "error_type": self.last_error})

    def status(self) -> dict[str, object]:
        running = self._task is not None and not self._task.done()
        return {
            "enabled": self.enabled,
            "running": running,
            # True only when this worker is configured on but its task is not actually running —
            # the exact "looks enabled, silently did nothing" state that used to report healthy.
            "crashed": self.enabled and not running,
            "last_heartbeat_at": self.last_heartbeat_at,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "last_fatal_error": self.last_fatal_error,
            "consecutive_failures": self.consecutive_failures,
            "processing_state": self.processing_state,
            "loaded_candles": self.loaded_candles,
        }
