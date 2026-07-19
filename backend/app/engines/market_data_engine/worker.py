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

    def start(self) -> None:
        if self.enabled and (self._task is None or self._task.done()):
            self._stop.clear()
            self._task = asyncio.create_task(self.run(), name="ten-market-data-worker")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def run(self) -> None:
        if self.bootstrap_enabled:
            await self._bootstrap()
        while not self._stop.is_set():
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
            "last_heartbeat_at": self.last_heartbeat_at,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "consecutive_failures": self.consecutive_failures,
            "processing_state": self.processing_state,
            "loaded_candles": self.loaded_candles,
        }
