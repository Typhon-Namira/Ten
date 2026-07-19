"""Provider registry, health monitoring, statistics, ranking, and automatic failover."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import TypeVar

from pydantic import BaseModel, Field

from .exceptions import ProviderUnavailableError
from .models import Candle, Timeframe
from .providers import MarketDataProvider, ProviderName, ProviderQuota, ProviderRequest

ResultT = TypeVar("ResultT")


class ProviderStatistics(BaseModel):
    provider: str
    requests: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    average_latency_ms: float = 0
    last_latency_ms: float = 0
    uptime_ratio: float = 1
    freshness_seconds: float | None = Field(default=None, ge=0)
    confidence: float = Field(default=100, ge=0, le=100)
    quota: ProviderQuota = ProviderQuota()
    healthy: bool = True
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_error: str | None = None


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, MarketDataProvider] = {}

    def register(self, provider: MarketDataProvider) -> None:
        self._providers[provider.provider_name.value] = provider

    def get(self, name: str) -> MarketDataProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ProviderUnavailableError(f"provider {name!r} is not registered") from exc

    def all(self) -> tuple[MarketDataProvider, ...]:
        return tuple(self._providers.values())


class ProviderManager:
    def __init__(self, registry: ProviderRegistry, *, preferred: str = ProviderName.TWELVE_DATA.value, failure_threshold: int = 3, recovery_successes: int = 3) -> None:
        self.registry = registry
        self.preferred = preferred
        self.failure_threshold = failure_threshold
        self.recovery_successes = recovery_successes
        self.statistics: dict[str, ProviderStatistics] = {item.provider_name.value: ProviderStatistics(provider=item.provider_name.value) for item in registry.all()}
        self.current_provider: str | None = None
        self._lock = asyncio.Lock()

    def rankings(self, symbol: str, timeframe: Timeframe, *, realtime: bool = False) -> list[str]:
        eligible = [item for item in self.registry.all() if item.capabilities.supports(symbol, timeframe, realtime=realtime)]
        return [
            item.provider_name.value
            for item in sorted(
                eligible,
                key=lambda provider: (
                    not self.statistics[provider.provider_name.value].healthy,
                    -self.statistics[provider.provider_name.value].confidence,
                    self.statistics[provider.provider_name.value].average_latency_ms,
                    provider.provider_name.value != self.preferred,
                ),
            )
        ]

    async def history(self, request: ProviderRequest) -> list[Candle]:
        return await self._execute(request.symbol, request.timeframe, lambda provider: provider.fetch_history(request), realtime=False)

    async def latest(self, symbol: str, timeframe: Timeframe) -> Candle:
        return await self._execute(symbol, timeframe, lambda provider: provider.fetch_latest(symbol, timeframe), realtime=True)

    async def history_from(self, provider_name: str, request: ProviderRequest) -> list[Candle]:
        provider = self.registry.get(provider_name)
        started = perf_counter()
        try:
            result = await provider.fetch_history(request)
            await self._record_success(provider_name, (perf_counter() - started) * 1000, result)
            return result
        except Exception as exc:
            await self._record_failure(provider_name, (perf_counter() - started) * 1000, type(exc).__name__)
            raise

    async def _execute(self, symbol: str, timeframe: Timeframe, operation: Callable[[MarketDataProvider], Awaitable[ResultT]], *, realtime: bool) -> ResultT:
        errors: list[str] = []
        for name in self.rankings(symbol, timeframe, realtime=realtime):
            provider = self.registry.get(name)
            started = perf_counter()
            try:
                result = await operation(provider)
                latency = (perf_counter() - started) * 1000
                await self._record_success(name, latency, result)
                self.current_provider = name
                return result
            except Exception as exc:
                await self._record_failure(name, (perf_counter() - started) * 1000, type(exc).__name__)
                errors.append(f"{name}: {exc}")
        raise ProviderUnavailableError("all eligible providers failed: " + "; ".join(errors))

    async def _record_success(self, name: str, latency: float, result: object) -> None:
        async with self._lock:
            stats = self.statistics[name]
            stats.requests += 1
            stats.successes += 1
            stats.consecutive_successes += 1
            stats.consecutive_failures = 0
            stats.last_latency_ms = latency
            stats.average_latency_ms += (latency - stats.average_latency_ms) / stats.successes
            stats.last_success_at = datetime.now(UTC)
            stats.last_error = None
            candles = result if isinstance(result, list) else [result]
            timestamps = [item.timestamp for item in candles if isinstance(item, Candle)]
            stats.freshness_seconds = max(0, (datetime.now(UTC) - max(timestamps)).total_seconds()) if timestamps else None
            stats.uptime_ratio = stats.successes / stats.requests
            stats.confidence = min(100, stats.uptime_ratio * 70 + max(0, 20 - stats.average_latency_ms / 100) + 10)
            if not stats.healthy and stats.consecutive_successes >= self.recovery_successes:
                stats.healthy = True
            stats.quota = await self.registry.get(name).quota()

    async def _record_failure(self, name: str, latency: float, error_type: str) -> None:
        async with self._lock:
            stats = self.statistics[name]
            stats.requests += 1
            stats.failures += 1
            stats.consecutive_failures += 1
            stats.consecutive_successes = 0
            stats.last_latency_ms = latency
            stats.last_failure_at = datetime.now(UTC)
            stats.last_error = error_type
            stats.uptime_ratio = stats.successes / stats.requests
            stats.confidence = max(0, stats.uptime_ratio * 70 - stats.consecutive_failures * 10)
            if stats.consecutive_failures >= self.failure_threshold:
                stats.healthy = False

    async def close(self) -> None:
        await asyncio.gather(*(provider.close() for provider in self.registry.all()))
