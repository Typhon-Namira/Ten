from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from backend.app.engines.market_data_engine.cache import MarketDataCache
from backend.app.engines.market_data_engine.events import NewCandle, RealtimeUpdated
from backend.app.engines.market_data_engine.manager import ProviderManager, ProviderRegistry
from backend.app.engines.market_data_engine.models import Candle, Timeframe
from backend.app.engines.market_data_engine.providers import InMemoryMarketDataProvider, ProviderName
from backend.app.engines.market_data_engine.repository import InMemoryMarketDataRepository
from backend.app.engines.market_data_engine.service import MarketDataService
from backend.app.engines.market_data_engine.worker import MarketDataWorker
from backend.app.core.bounded import BoundedSet
from backend.app.events import Event, InMemoryEventBus
from backend.app.features import FeatureRecord, InMemoryFeatureStore

NOW = datetime(2026, 7, 22, 9, tzinfo=UTC)


def candle(timestamp: datetime = NOW, *, timeframe: Timeframe = Timeframe.M1, close: float = 4101) -> Candle:
    return Candle(
        symbol="XAUUSD",
        timeframe=timeframe,
        timestamp=timestamp,
        open=4100,
        high=max(4102, close),
        low=min(4099, close),
        close=close,
        volume=10,
        provider="memory",
        ingestion_timestamp=timestamp + timeframe.duration,
    )


def service(tmp_path: Path, provider: InMemoryMarketDataProvider) -> tuple[MarketDataService, InMemoryMarketDataRepository, InMemoryEventBus]:
    registry = ProviderRegistry()
    registry.register(provider)
    repository = InMemoryMarketDataRepository()
    bus = InMemoryEventBus(history_capacity=20)
    return (
        MarketDataService(
            ProviderManager(registry, preferred=ProviderName.MEMORY.value),
            repository=repository,
            cache=MarketDataCache(tmp_path, max_entries=10),
            event_bus=bus,
        ),
        repository,
        bus,
    )


@pytest.mark.asyncio
async def test_duplicate_closed_candle_is_not_rewritten_or_reanalysed(tmp_path: Path) -> None:
    market, repository, bus = service(tmp_path, InMemoryMarketDataProvider((candle(),)))

    await market.latest("XAUUSD", Timeframe.M1, refresh=True)
    await market.latest("XAUUSD", Timeframe.M1, refresh=True)

    assert len(repository._realtime) == 1
    assert sum(isinstance(event, NewCandle) for event in bus.history()) == 1
    assert sum(isinstance(event, RealtimeUpdated) for event in bus.history()) == 1
    assert market.poll_metrics == {
        "attempts": 2,
        "new_closed_candles": 1,
        "duplicate_responses": 1,
        "corrected_responses": 0,
        "persistence_writes": 1,
    }


@pytest.mark.asyncio
async def test_duplicate_poll_soak_retains_constant_state(tmp_path: Path) -> None:
    market, repository, bus = service(tmp_path, InMemoryMarketDataProvider((candle(),)))

    for _ in range(500):
        await market.latest("XAUUSD", Timeframe.M1, refresh=True)

    assert len(repository._realtime) == 1
    assert len(bus.history()) == 2
    assert market.cache.metrics()["entries"] == 1
    assert market.poll_metrics["duplicate_responses"] == 499


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe", (Timeframe.M1, Timeframe.M5, Timeframe.M15))
async def test_each_timeframe_emits_once_per_canonical_closed_candle(tmp_path: Path, timeframe: Timeframe) -> None:
    value = candle(timeframe=timeframe)
    market, _, bus = service(tmp_path, InMemoryMarketDataProvider((value,)))

    for _ in range(5):
        assert await market.latest("XAUUSD", timeframe, refresh=True) == value

    emitted = [event for event in bus.history() if isinstance(event, NewCandle)]
    assert len(emitted) == 1
    assert emitted[0].payload["timestamp"] == value.timestamp.isoformat().replace("+00:00", "Z")


@pytest.mark.asyncio
async def test_same_identity_correction_updates_storage_without_duplicate_analysis(tmp_path: Path) -> None:
    provider = InMemoryMarketDataProvider((candle(),))
    market, repository, bus = service(tmp_path, provider)
    await market.latest("XAUUSD", Timeframe.M1, refresh=True)
    provider._items = [candle(close=4101.5)]

    corrected = await market.latest("XAUUSD", Timeframe.M1, refresh=True)

    assert corrected is not None and corrected.close == 4101.5
    assert (await repository.candle_at("XAUUSD", Timeframe.M1, NOW)).close == 4101.5  # type: ignore[union-attr]
    assert sum(isinstance(event, NewCandle) for event in bus.history()) == 1
    assert market.poll_metrics["corrected_responses"] == 1


@pytest.mark.asyncio
async def test_missed_closed_candles_are_recovered_in_order_after_downtime(tmp_path: Path) -> None:
    provider = InMemoryMarketDataProvider((candle(),))
    market, repository, bus = service(tmp_path, provider)
    await market.latest("XAUUSD", Timeframe.M1, refresh=True)
    provider._items = [candle(NOW + timedelta(minutes=index)) for index in range(4)]

    await market.latest("XAUUSD", Timeframe.M1, refresh=True)

    stored = await repository.history("XAUUSD", Timeframe.M1, limit=10)
    emitted = [event.payload["timestamp"] for event in bus.history() if isinstance(event, NewCandle)]
    assert [item.timestamp for item in stored] == [NOW + timedelta(minutes=index) for index in range(4)]
    assert len(emitted) == 4
    assert emitted == sorted(emitted)


class SlowProvider(InMemoryMarketDataProvider):
    def __init__(self, items: tuple[Candle, ...]) -> None:
        super().__init__(items)
        self.active = 0
        self.peak_active = 0

    async def fetch_latest(self, symbol: str, timeframe: Timeframe) -> Candle:
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            await asyncio.sleep(0)
            return await super().fetch_latest(symbol, timeframe)
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_overlapping_polls_for_one_series_are_serialized(tmp_path: Path) -> None:
    provider = SlowProvider((candle(),))
    market, repository, bus = service(tmp_path, provider)

    await asyncio.gather(
        market.latest("XAUUSD", Timeframe.M1, refresh=True),
        market.latest("XAUUSD", Timeframe.M1, refresh=True),
    )

    assert provider.peak_active == 1
    assert len(repository._realtime) == 1
    assert sum(isinstance(event, NewCandle) for event in bus.history()) == 1


@pytest.mark.asyncio
async def test_worker_schedules_each_timeframe_at_its_own_next_close() -> None:
    class StubService:
        def __init__(self) -> None:
            self.sessions = type("Sessions", (), {})()

        async def latest(self, symbol: str, timeframe: Timeframe, *, refresh: bool = False) -> Candle:
            return candle(timeframe=timeframe)

    worker = MarketDataWorker(
        StubService(),  # type: ignore[arg-type]
        enabled=True,
        symbols=("XAUUSD",),
        timeframes=(Timeframe.M1, Timeframe.M5, Timeframe.M15),
        bootstrap_enabled=False,
        bootstrap_candles=500,
        poll_seconds=10,
    )
    observed_at = NOW + timedelta(minutes=1)
    for timeframe in (Timeframe.M1, Timeframe.M5, Timeframe.M15):
        await worker._poll("XAUUSD", timeframe, now=observed_at)

    assert worker._next_poll_at[("XAUUSD", Timeframe.M1)] == NOW + timedelta(minutes=2)
    assert worker._next_poll_at[("XAUUSD", Timeframe.M5)] == NOW + timedelta(minutes=10)
    assert worker._next_poll_at[("XAUUSD", Timeframe.M15)] == NOW + timedelta(minutes=30)


@pytest.mark.asyncio
async def test_event_and_feature_histories_never_exceed_configured_bounds() -> None:
    bus = InMemoryEventBus(history_capacity=3)
    for index in range(8):
        await bus.publish(Event(correlation_id=uuid4(), source="test", payload={"index": index}))
    assert len(bus.history()) == 3
    assert bus.metrics()["history_evictions"] == 5

    features = InMemoryFeatureStore(max_entries=3)
    correlations = [uuid4() for _ in range(8)]
    for index, correlation in enumerate(correlations):
        await features.write(
            FeatureRecord(
                correlation_id=correlation,
                namespace="test",
                engine_name="test",
                engine_version="1",
                compatibility_version="1",
                values={"index": index},
            )
        )
    assert features.metrics() == {"entries": 3, "capacity": 3, "evictions": 5}
    assert not (await features.snapshot(correlations[0])).features
    assert (await features.snapshot(correlations[-1])).features["test"]["index"] == 7


def test_bounded_set_refreshes_existing_items_and_evicts_oldest() -> None:
    values = BoundedSet[int](3)
    for value in (1, 2, 3, 1, 4):
        values.add(value)

    assert tuple(values) == (3, 1, 4)
    assert values.evictions == 1


def test_default_poll_and_pool_guardrails_are_validated() -> None:
    from pydantic import ValidationError

    from backend.app.core.config.settings import Settings

    settings = Settings(_env_file=None, market_data_worker_enabled=False)
    assert (settings.market_data_poll_seconds, settings.market_data_idle_poll_seconds) == (10, 30)
    assert (settings.db_pool_size, settings.db_max_overflow, settings.db_pool_timeout_seconds) == (3, 2, 30)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, market_data_worker_enabled=False, max_client_queue_size=0)
