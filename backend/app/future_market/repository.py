"""Bounded persistence for TEN 2.0 forecasts."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Protocol

from .models import FutureMarketForecast, MAX_FORECASTS_PER_INSTRUMENT


class FutureMarketRepository(Protocol):
    async def save(self, forecast: FutureMarketForecast) -> FutureMarketForecast: ...
    async def latest(self, instrument: str) -> FutureMarketForecast | None: ...
    async def history(self, instrument: str, limit: int = 100) -> tuple[FutureMarketForecast, ...]: ...
    async def count(self, instrument: str) -> int: ...


class BoundedInMemoryFutureMarketRepository:
    """Strict rolling window: never retain more than 100 forecasts per instrument."""

    def __init__(self, limit: int = MAX_FORECASTS_PER_INSTRUMENT) -> None:
        self.limit = min(MAX_FORECASTS_PER_INSTRUMENT, max(1, limit))
        self._values: dict[str, deque[FutureMarketForecast]] = defaultdict(
            lambda: deque(maxlen=self.limit)
        )
        self._lock = asyncio.Lock()

    async def save(self, forecast: FutureMarketForecast) -> FutureMarketForecast:
        async with self._lock:
            values = self._values[forecast.instrument]
            if values and values[-1].forecast_id == forecast.forecast_id:
                return values[-1]
            values.append(forecast)
            return forecast

    async def latest(self, instrument: str) -> FutureMarketForecast | None:
        async with self._lock:
            values = self._values.get(instrument)
            return values[-1] if values else None

    async def history(self, instrument: str, limit: int = 100) -> tuple[FutureMarketForecast, ...]:
        bounded = min(self.limit, max(1, limit))
        async with self._lock:
            values = list(self._values.get(instrument, ()))
        return tuple(reversed(values[-bounded:]))

    async def count(self, instrument: str) -> int:
        async with self._lock:
            return len(self._values.get(instrument, ()))
