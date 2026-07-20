from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backend.app.engines.market_data_engine import (
    AnomalyType,
    Candle,
    InMemoryMarketDataProvider,
    MarketDataCache,
    MarketDataService,
    MarketDataValidator,
    ProviderManager,
    ProviderRegistry,
    Timeframe,
)
from backend.app.engines.market_data_engine.events import HistoricalUpdated, NewCandle
from backend.app.engines.market_data_engine.exceptions import ProviderRateLimitedError, ProviderUnavailableError
from backend.app.engines.market_data_engine.metrics import calculate_metrics
from backend.app.engines.market_data_engine.models import DataQualityLevel, MarketSession
from backend.app.engines.market_data_engine.providers import MarketDataProvider, ProviderCapabilities, ProviderName, ProviderRequest
from backend.app.engines.market_data_engine.repository import InMemoryMarketDataRepository
from backend.app.engines.market_data_engine.sessions import MarketSessionEngine
from backend.app.events import InMemoryEventBus


def series(count: int = 5, *, gap: bool = False, provider: str = "memory") -> list[Candle]:
    start = datetime(2026, 7, 13, 8, tzinfo=UTC)
    result: list[Candle] = []
    for index in range(count):
        offset = index + (1 if gap and index >= 2 else 0)
        price = 2400 + index
        result.append(
            Candle(
                timestamp=start + timedelta(minutes=offset),
                timeframe=Timeframe.M1,
                open=price,
                high=price + 2,
                low=price - 1,
                close=price + 1,
                volume=10,
                spread=0.2,
                provider=provider,
                ingestion_timestamp=start + timedelta(minutes=offset, seconds=1),
            )
        )
    return result


def test_validator_detects_gap_and_assigns_quality() -> None:
    report = MarketDataValidator().validate(series(gap=True), now=datetime(2026, 7, 13, 9, tzinfo=UTC))
    assert any(item.type == AnomalyType.MARKET_GAP and item.missing_count == 1 for item in report.anomalies)
    assert all(item.quality_level == DataQualityLevel.NATIVE for item in report.candles)


def test_validator_rejects_future_and_duplicates() -> None:
    candles = series(2)
    with pytest.raises(Exception, match="duplicate"):
        MarketDataValidator().validate([candles[0], candles[0]], now=datetime(2026, 7, 13, 9, tzinfo=UTC))
    with pytest.raises(Exception, match="future"):
        MarketDataValidator().validate(candles, now=datetime(2026, 7, 1, tzinfo=UTC))


@pytest.mark.asyncio
async def test_cache_memory_persistent_and_invalidation(tmp_path: Path) -> None:
    cache = MarketDataCache(tmp_path, max_entries=2)
    candles = series(1)
    await cache.set("history:XAUUSD:M1", candles, 60)
    assert await cache.get("history:XAUUSD:M1") == candles
    second = MarketDataCache(tmp_path)
    assert await second.get("history:XAUUSD:M1") == candles
    await second.invalidate("history")
    assert await second.get("history:XAUUSD:M1") is None
    assert cache.statistics.hit_ratio == 1


class FailingProvider(MarketDataProvider):
    provider_name = ProviderName.TWELVE_DATA
    capabilities = ProviderCapabilities()

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        raise RuntimeError("down")

    async def fetch_latest(self, symbol: str, timeframe: Timeframe) -> Candle:
        raise RuntimeError("down")


class ReversedDuplicateProvider(InMemoryMarketDataProvider):
    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        values = await super().fetch_history(request)
        return [*reversed(values), values[-1]]


@pytest.mark.asyncio
async def test_provider_manager_fails_over_and_records_health() -> None:
    registry = ProviderRegistry()
    registry.register(FailingProvider())
    registry.register(InMemoryMarketDataProvider(series()))
    manager = ProviderManager(registry, failure_threshold=1)
    result = await manager.history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.M1))
    assert len(result) == 5
    assert manager.current_provider == ProviderName.MEMORY.value
    assert manager.statistics[ProviderName.TWELVE_DATA.value].healthy is False
    assert manager.statistics[ProviderName.MEMORY.value].successes == 1


class RateLimitedProvider(MarketDataProvider):
    provider_name = ProviderName.TWELVE_DATA
    capabilities = ProviderCapabilities()

    def __init__(self) -> None:
        self.calls = 0

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        self.calls += 1
        raise ProviderRateLimitedError("rate limited", retry_after_seconds=30)

    async def fetch_latest(self, symbol: str, timeframe: Timeframe) -> Candle:
        self.calls += 1
        raise ProviderRateLimitedError("rate limited", retry_after_seconds=30)


@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_rate_limit_and_skips_further_calls() -> None:
    """A 429 must open the breaker immediately (not after `failure_threshold` failures) and
    every subsequent call within the backoff window must be rejected locally — never a second
    live HTTP call — which is what actually stops a rate-limited provider from being hammered
    across a bootstrap loop's remaining symbol/timeframe iterations."""
    registry = ProviderRegistry()
    limited = RateLimitedProvider()
    registry.register(limited)
    manager = ProviderManager(registry, preferred=ProviderName.TWELVE_DATA.value)
    with pytest.raises(ProviderUnavailableError):
        await manager.history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.M1))
    assert limited.calls == 1
    stats = manager.statistics[ProviderName.TWELVE_DATA.value]
    assert stats.backoff_until is not None
    assert stats.last_error == "ProviderRateLimitedError"
    with pytest.raises(ProviderUnavailableError):
        await manager.history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.M1))
    assert limited.calls == 1


@pytest.mark.asyncio
async def test_circuit_breaker_recovers_after_a_success() -> None:
    registry = ProviderRegistry()
    limited = RateLimitedProvider()
    registry.register(limited)
    manager = ProviderManager(registry, preferred=ProviderName.TWELVE_DATA.value)
    with pytest.raises(ProviderUnavailableError):
        await manager.history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.M1))
    stats = manager.statistics[ProviderName.TWELVE_DATA.value]
    assert stats.backoff_until is not None
    stats.backoff_until = datetime(2000, 1, 1, tzinfo=UTC)  # force the window open again
    limited.fetch_history = InMemoryMarketDataProvider(series()).fetch_history  # type: ignore[method-assign]
    result = await manager.history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.M1))
    assert len(result) == 5
    assert manager.statistics[ProviderName.TWELVE_DATA.value].backoff_until is None


@pytest.mark.asyncio
async def test_history_realtime_replay_and_events(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    candles = series()
    registry.register(InMemoryMarketDataProvider(candles))
    bus = InMemoryEventBus()
    service = MarketDataService(
        ProviderManager(registry, preferred=ProviderName.MEMORY.value),
        repository=InMemoryMarketDataRepository(),
        cache=MarketDataCache(tmp_path),
        event_bus=bus,
    )
    synced = await service.history("XAU/USD", Timeframe.M1, refresh=True)
    latest = await service.latest("XAUUSD", Timeframe.M1, refresh=True)
    replay = await service.replay("XAUUSD", Timeframe.M1, candles[2].timestamp)
    exact = await service.candle_at("XAUUSD", Timeframe.M1, candles[2].timestamp)
    assert synced == candles
    assert latest == candles[-1]
    assert replay[-1] == candles[2]
    assert exact == candles[2]
    assert any(isinstance(event, HistoricalUpdated) for event in bus.history())
    assert any(isinstance(event, NewCandle) for event in bus.history())


@pytest.mark.asyncio
async def test_history_normalizes_provider_order_and_duplicates(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    expected = series()
    registry.register(ReversedDuplicateProvider(expected))
    service = MarketDataService(
        ProviderManager(registry, preferred=ProviderName.MEMORY.value),
        repository=InMemoryMarketDataRepository(),
        cache=MarketDataCache(tmp_path),
    )
    result = await service.history("XAUUSD", Timeframe.M1, refresh=True)
    assert result == expected
    assert await service.repository.count("XAUUSD", Timeframe.M1) == len(expected)


def test_sessions_are_dst_aware_and_weekends_close() -> None:
    engine = MarketSessionEngine()
    assert engine.session_at(datetime(2026, 7, 13, 14, tzinfo=UTC)) == MarketSession.LONDON_NEW_YORK_OVERLAP
    assert engine.session_at(datetime(2026, 7, 11, 14, tzinfo=UTC)) == MarketSession.WEEKEND
    assert engine.is_open(datetime(2026, 7, 11, 14, tzinfo=UTC)) is False


def test_metrics_are_deterministic() -> None:
    candles = series()
    metrics = calculate_metrics(candles, latency_ms=12, now=datetime(2026, 7, 13, 9, tzinfo=UTC))
    assert metrics.atr > 0
    assert metrics.current_spread == 0.2
    assert metrics.latency_ms == 12
    assert metrics.price_velocity > 0
