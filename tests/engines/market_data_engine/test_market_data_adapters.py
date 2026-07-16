import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from backend.app.engines.market_data_engine.adapters import (
    AlphaVantageProvider,
    FinancialModelingPrepProvider,
    OandaProvider,
    TwelveDataProvider,
    _parse_timestamp,
    provider_api_key,
)
from backend.app.engines.market_data_engine.cache import MarketDataCache
from backend.app.engines.market_data_engine.config import MarketDataConfig, ProviderConfig
from backend.app.engines.market_data_engine.exceptions import ProviderResponseError, ProviderUnavailableError
from backend.app.engines.market_data_engine.manager import ProviderManager, ProviderRegistry
from backend.app.engines.market_data_engine.metrics import calculate_metrics
from backend.app.engines.market_data_engine.models import Candle, DataQualityLevel, MarketSession, Tick, Timeframe, canonical_symbol
from backend.app.engines.market_data_engine.providers import CsvHistoricalProvider, InMemoryMarketDataProvider, ProviderName, ProviderRequest
from backend.app.engines.market_data_engine.repository import InMemoryMarketDataRepository, SqlAlchemyMarketDataRepository
from backend.app.engines.market_data_engine.service import MarketDataService, build_market_data_service
from backend.app.engines.market_data_engine.sessions import MarketSessionEngine
from backend.app.engines.market_data_engine.validation import AnomalyType, MarketDataValidator


NOW = datetime(2026, 7, 13, 12, tzinfo=UTC)


def candle(at: datetime = NOW, *, provider: str = "memory") -> Candle:
    return Candle(timestamp=at, timeframe=Timeframe.M1, open=2400, high=2402, low=2399, close=2401, volume=5, spread=0.2, provider=provider, ingestion_timestamp=at + timedelta(seconds=1))


@pytest.mark.asyncio
async def test_http_transport_quota_error_and_close() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/bad":
            return httpx.Response(500)
        return httpx.Response(200, json={"ok": True}, headers={"x-ratelimit-limit": "10", "x-ratelimit-remaining": "9"})

    provider = TwelveDataProvider(api_key="key", base_url="https://example.test")
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await provider._get("/ok", params={}) == {"ok": True}
    assert (await provider.quota()).remaining == 9
    with pytest.raises(ProviderResponseError):
        await provider._get("/bad", params={})
    await provider.close()
    with pytest.raises(ValueError):
        TwelveDataProvider(api_key="", base_url="https://example.test")
    assert _parse_timestamp(0) == datetime(1970, 1, 1, tzinfo=UTC)
    assert _parse_timestamp("2026-01-01T00:00:00") == datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_all_provider_payloads_normalize(monkeypatch: pytest.MonkeyPatch) -> None:
    request = ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.M1, limit=2)
    twelve = TwelveDataProvider(api_key="x", base_url="https://x")
    monkeypatch.setattr(twelve, "_get", AsyncMock(return_value={"values": [{"datetime": "2026-07-13T12:00:00Z", "open": "1", "high": "3", "low": "1", "close": "2", "volume": "4"}]}))
    assert (await twelve.fetch_history(request))[0].provider == ProviderName.TWELVE_DATA.value
    monkeypatch.setattr(twelve, "_get", AsyncMock(return_value={"status": "error", "message": "bad"}))
    with pytest.raises(ProviderResponseError):
        await twelve.fetch_history(request)

    alpha = AlphaVantageProvider(api_key="x", base_url="https://x")
    monkeypatch.setattr(alpha, "_get", AsyncMock(return_value={"Time Series FX (1min)": {"2026-07-13T12:00:00": {"1. open": "1", "2. high": "3", "3. low": "1", "4. close": "2"}}}))
    assert len(await alpha.fetch_history(request)) == 1
    monkeypatch.setattr(alpha, "_get", AsyncMock(return_value={"Note": "quota"}))
    with pytest.raises(ProviderResponseError):
        await alpha.fetch_history(request)

    fmp = FinancialModelingPrepProvider(api_key="x", base_url="https://x")
    monkeypatch.setattr(fmp, "_get", AsyncMock(return_value=[{"date": "2026-07-13T12:00:00Z", "open": 1, "high": 3, "low": 1, "close": 2, "volume": 4}]))
    assert len(await fmp.fetch_history(request)) == 1
    monkeypatch.setattr(fmp, "_get", AsyncMock(return_value={"bad": True}))
    with pytest.raises(ProviderResponseError):
        await fmp.fetch_history(request)

    oanda = OandaProvider(api_key="x", base_url="https://x")
    row = {"time": "2026-07-13T12:00:00Z", "complete": True, "volume": 4, "mid": {"o": "1", "h": "3", "l": "1", "c": "2"}, "bid": {"c": "1.9"}, "ask": {"c": "2.1"}}
    monkeypatch.setattr(oanda, "_get", AsyncMock(return_value={"candles": [row, {"complete": False}]}))
    assert (await oanda.fetch_history(request))[0].spread == pytest.approx(0.2)
    monkeypatch.setattr(oanda, "_get", AsyncMock(return_value={}))
    with pytest.raises(ProviderResponseError):
        await oanda.fetch_history(request)
    await alpha.close()
    await fmp.close()
    await oanda.close()


@pytest.mark.asyncio
async def test_csv_provider_and_empty_latest(tmp_path: Path) -> None:
    path = tmp_path / "candles.csv"
    path.write_text("timestamp,open,high,low,close,volume,spread\n2026-07-13T12:00:00Z,1,3,1,2,4,0.2\n", encoding="utf-8")
    provider = CsvHistoricalProvider(path)
    items = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.M1))
    assert items[0].close == 2
    assert (await provider.fetch_latest("XAUUSD", Timeframe.M1)).close == items[0].close
    with pytest.raises(LookupError):
        await InMemoryMarketDataProvider([]).fetch_latest("XAUUSD", Timeframe.M1)


@pytest.mark.asyncio
async def test_service_cache_state_metrics_and_unavailable(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    item = candle()
    registry.register(InMemoryMarketDataProvider([item]))
    repository = InMemoryMarketDataRepository()
    service = MarketDataService(ProviderManager(registry, preferred="memory"), repository=repository, cache=MarketDataCache(tmp_path), config=MarketDataConfig())
    assert await service.history("XAUUSD", Timeframe.M1, refresh=True) == [item]
    assert await service.history("XAUUSD", Timeframe.M1) == [item]
    assert await service.latest("XAUUSD", Timeframe.M1) == item
    assert (await service.metrics("XAUUSD", Timeframe.M1)).atr > 0  # type: ignore[union-attr]
    state = await service.state("XAUUSD", Timeframe.M1, at=NOW + timedelta(minutes=2))
    assert state.current_provider == "memory" and state.data_freshness_seconds == 120
    await service.close()
    empty = MarketDataService(ProviderManager(ProviderRegistry()), cache=MarketDataCache(tmp_path / "empty"))
    assert await empty.metrics("XAUUSD", Timeframe.M1) is None
    with pytest.raises(ProviderUnavailableError):
        await empty.latest("XAUUSD", Timeframe.M1, refresh=True)
    assert (await empty.state("XAUUSD", Timeframe.M1)).provider_health == "unavailable"


def test_session_validation_metrics_and_sql_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = MarketSessionEngine({date(2026, 12, 25)})
    assert sessions.session_at(datetime(2026, 12, 25, tzinfo=UTC)) == MarketSession.HOLIDAY
    assert sessions.session_at(datetime(2026, 7, 13, 8, tzinfo=UTC)) == MarketSession.LONDON
    assert sessions.session_at(datetime(2026, 7, 13, 20, tzinfo=UTC)) == MarketSession.NEW_YORK
    assert sessions.session_at(datetime(2026, 7, 13, 3, tzinfo=UTC)) == MarketSession.ASIA
    assert sessions.session_at(datetime(2026, 7, 13, 22, tzinfo=UTC)) == MarketSession.CLOSED
    validator = MarketDataValidator(holidays={date(2026, 12, 25)})
    holiday = validator.validate([candle(datetime(2026, 12, 24, 23, 59, tzinfo=UTC)), candle(datetime(2026, 12, 26, 0, 1, tzinfo=UTC))], now=datetime(2026, 12, 26, 1, tzinfo=UTC))
    assert holiday.anomalies[0].type == AnomalyType.HOLIDAY_GAP
    peer = candle(provider="other").model_copy(update={"close": 2500.0, "high": 2501.0})
    assert validator.compare([candle()], [peer])[0].type == AnomalyType.PROVIDER_INCONSISTENCY
    with pytest.raises(ValueError):
        calculate_metrics([])
    values = SqlAlchemyMarketDataRepository._values(candle())
    assert values["quality_score"] == 100
    monkeypatch.setenv("TEN_TEST_KEY", "secret")
    assert provider_api_key("TEN_TEST_KEY") == "secret"
    assert build_market_data_service().manager.registry.all() == ()


class AlternateProvider(InMemoryMarketDataProvider):
    provider_name = ProviderName.CSV


@pytest.mark.asyncio
async def test_gap_recovery_and_sql_write_paths(tmp_path: Path) -> None:
    first, second, third = candle(NOW), candle(NOW + timedelta(minutes=1)), candle(NOW + timedelta(minutes=2))
    registry = ProviderRegistry()
    registry.register(InMemoryMarketDataProvider([first, third]))
    registry.register(AlternateProvider([first, second, third]))
    service = MarketDataService(ProviderManager(registry, preferred="memory"), cache=MarketDataCache(tmp_path))
    result = await service.history("XAUUSD", Timeframe.M1, refresh=True)
    assert len(result) == 3 and result[1].quality_score == 95

    session = AsyncMock()
    repository = SqlAlchemyMarketDataRepository(session)
    assert await repository.upsert_historical([first]) == 1
    assert await repository.upsert_historical([]) == 0
    await repository.append_realtime(first)
    assert session.execute.await_count == 2
    assert session.commit.await_count == 2


@pytest.mark.asyncio
async def test_cache_expiry_eviction_and_corruption(tmp_path: Path) -> None:
    cache = MarketDataCache(tmp_path, max_entries=1)
    await cache.set("one", [candle()], 60)
    await cache.set("two", [candle()], 60)
    assert cache.statistics.evictions == 1
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert await cache.get("broken") is None
    await cache.set("expired", [candle()], 1)
    cache._memory["expired"] = (datetime(2000, 1, 1, tzinfo=UTC), [candle()])
    path = cache._path("expired")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["expires"] = datetime(2000, 1, 1, tzinfo=UTC).isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert await cache.get("expired") is None


def test_model_and_quality_edge_contracts() -> None:
    assert canonical_symbol("xau_usd") == "XAUUSD"
    with pytest.raises(ValueError):
        canonical_symbol("/")
    with pytest.raises(ValueError):
        Candle(timestamp=NOW, timeframe=Timeframe.M1, open=-1, high=2, low=1, close=1)
    with pytest.raises(ValueError):
        Candle(timestamp=NOW, timeframe=Timeframe.M1, open=2, high=1, low=1, close=2)
    with pytest.raises(ValueError):
        Tick(timestamp=NOW, bid=2, ask=1)
    tick = Tick(timestamp=NOW, bid=1, ask=2)
    assert tick.symbol == "XAU/USD"
    validator = MarketDataValidator()
    assert validator.score(candle(), [], verified=True).quality_score == 98
    assert validator.score(candle(), [], interpolated=True).quality_level == DataQualityLevel.INTERPOLATED
    assert validator.score(candle(), [], recovered=True).quality_level == DataQualityLevel.RECOVERED


@pytest.mark.asyncio
async def test_provider_factory_activates_configured_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    definitions = (
        ProviderConfig(name="twelve_data", base_url="https://x", api_key_env="KEY"),
        ProviderConfig(name="alpha_vantage", base_url="https://x", api_key_env="KEY"),
        ProviderConfig(name="financial_modeling_prep", base_url="https://x", api_key_env="KEY"),
        ProviderConfig(name="oanda", base_url="https://x", api_key_env="KEY", account_id_env="ACCOUNT"),
    )
    monkeypatch.setenv("KEY", "secret")
    monkeypatch.setenv("ACCOUNT", "account")
    service = build_market_data_service(MarketDataConfig(providers=definitions))
    assert len(service.manager.registry.all()) == 4
    await service.close()


@pytest.mark.asyncio
async def test_sql_history_and_time_travel_read_paths() -> None:
    item = candle()
    record = SimpleNamespace(
        symbol=item.symbol,
        timeframe=item.timeframe.value,
        timestamp=item.timestamp,
        open=item.open,
        high=item.high,
        low=item.low,
        close=item.close,
        volume=item.volume,
        spread=item.spread,
        provider=item.provider,
        quality_score=item.quality_score,
        quality_level=item.quality_level.value,
        ingestion_timestamp=item.ingestion_timestamp,
    )

    class Scalars:
        def all(self) -> list[object]:
            return [record]

    session = AsyncMock()
    session.scalars.return_value = Scalars()
    repository = SqlAlchemyMarketDataRepository(session)
    result = await repository.history("XAUUSD", Timeframe.M1, start=NOW - timedelta(minutes=1), end=NOW + timedelta(minutes=1))
    assert result == [item]
    assert await repository.candle_at("XAUUSD", Timeframe.M1, NOW) == item
