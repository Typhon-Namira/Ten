"""Single-source-of-truth facade for historical, realtime, replay, and state queries."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from backend.app.events import EventBus, InMemoryEventBus

from .adapters import AlphaVantageProvider, FinancialModelingPrepProvider, OandaProvider, TwelveDataProvider, provider_api_key
from .cache import MarketDataCache
from .config import MarketDataConfig
from .events import GapDetected, HistoricalUpdated, NewCandle, RealtimeUpdated
from .manager import ProviderManager, ProviderRegistry
from .metrics import calculate_metrics
from .models import Candle, MarketMetrics, MarketState, RealtimeStatus, SyncStatus, Timeframe, canonical_symbol
from .providers import ProviderRequest
from .repository import InMemoryMarketDataRepository, MarketDataRepository
from .sessions import MarketSessionEngine
from .validation import MarketDataValidator

logger = logging.getLogger(__name__)


class MarketDataService:
    def __init__(
        self,
        manager: ProviderManager,
        *,
        repository: MarketDataRepository | None = None,
        cache: MarketDataCache | None = None,
        validator: MarketDataValidator | None = None,
        sessions: MarketSessionEngine | None = None,
        event_bus: EventBus | None = None,
        config: MarketDataConfig | None = None,
    ) -> None:
        self.config = config or MarketDataConfig()
        self.manager = manager
        self.repository = repository or InMemoryMarketDataRepository()
        self.cache = cache or MarketDataCache(Path(self.config.cache.persistent_directory), max_entries=self.config.cache.memory_max_entries)
        self.validator = validator or MarketDataValidator(self.config.validation)
        self.sessions = sessions or MarketSessionEngine()
        self.event_bus = event_bus or InMemoryEventBus()
        self.sync_status = SyncStatus.IDLE
        self.realtime_status = RealtimeStatus.STOPPED

    async def history(self, symbol: str, timeframe: Timeframe, *, start: datetime | None = None, end: datetime | None = None, limit: int = 500, refresh: bool = False) -> list[Candle]:
        symbol = canonical_symbol(symbol)
        key = self._key("history", symbol, timeframe, start, end, limit)
        if not refresh:
            cached = await self.cache.get(key)
            if cached is not None:
                return cached
            stored = await self.repository.history(symbol, timeframe, start, end, limit)
            if stored:
                await self.cache.set(key, stored, self.config.cache.historical_ttl_seconds)
                return stored
        self.sync_status = SyncStatus.SYNCING
        try:
            raw = await self.manager.history(ProviderRequest(symbol=symbol, timeframe=timeframe, start=start, end=end, limit=limit))
            unique = {(item.timestamp, canonical_symbol(item.symbol), item.timeframe): item for item in raw}
            normalized = sorted(unique.values(), key=lambda item: item.timestamp)
            if len(normalized) != len(raw):
                logger.warning("market_data.duplicates_removed", extra={"symbol": symbol, "timeframe": timeframe.value, "duplicate_count": len(raw) - len(normalized)})
            report = self.validator.validate(normalized)
            if any(anomaly.missing_count for anomaly in report.anomalies):
                recovered = await self._recover_gaps(symbol, timeframe, report.candles, limit)
                if recovered != report.candles:
                    report = self.validator.validate(recovered)
            for anomaly in report.anomalies:
                logger.warning("market_data_anomaly", extra={"anomaly": anomaly.model_dump(mode="json"), "symbol": symbol, "timeframe": timeframe.value})
                if anomaly.missing_count:
                    await self.event_bus.publish(GapDetected(correlation_id=uuid4(), source="market_data", payload=anomaly.model_dump(mode="json")))
            await self.repository.upsert_historical(report.candles)
            await self.cache.set(key, report.candles, self.config.cache.historical_ttl_seconds)
            self.sync_status = SyncStatus.COMPLETE if report.valid else SyncStatus.PARTIAL
            await self.event_bus.publish(
                HistoricalUpdated(
                    correlation_id=uuid4(),
                    source="market_data",
                    payload={"symbol": symbol, "timeframe": timeframe.value, "count": len(report.candles), "anomalies": len(report.anomalies)},
                )
            )
            return report.candles
        except Exception:
            self.sync_status = SyncStatus.FAILED
            raise

    async def latest(self, symbol: str, timeframe: Timeframe, *, refresh: bool = False) -> Candle | None:
        symbol = canonical_symbol(symbol)
        key = self._key("latest", symbol, timeframe)
        if not refresh:
            cached = await self.cache.get(key)
            if cached:
                return cached[-1]
            stored = await self.repository.history(symbol, timeframe, limit=1)
            if stored:
                return stored[-1]
        self.realtime_status = RealtimeStatus.POLLING
        candle = await self.manager.latest(symbol, timeframe)
        normalized = self.validator.validate([candle]).candles[-1]
        await self.repository.append_realtime(normalized)
        await self.cache.set(key, [normalized], self.config.cache.realtime_ttl_seconds)
        await self.event_bus.publish(NewCandle(correlation_id=uuid4(), source="market_data", payload=normalized.model_dump(mode="json")))
        await self.event_bus.publish(RealtimeUpdated(correlation_id=uuid4(), source="market_data", payload={"symbol": symbol, "timeframe": timeframe.value}))
        return normalized

    async def replay(self, symbol: str, timeframe: Timeframe, at: datetime, *, limit: int = 500) -> list[Candle]:
        return await self.repository.history(symbol, timeframe, end=at, limit=limit)

    async def candle_at(self, symbol: str, timeframe: Timeframe, at: datetime) -> Candle | None:
        return await self.repository.candle_at(symbol, timeframe, at)

    async def metrics(self, symbol: str, timeframe: Timeframe, *, limit: int = 100) -> MarketMetrics | None:
        candles = await self.repository.history(symbol, timeframe, limit=limit)
        if not candles:
            return None
        provider = self.manager.current_provider
        latency = self.manager.statistics[provider].last_latency_ms if provider else 0
        return calculate_metrics(candles, latency_ms=latency)

    async def state(self, symbol: str, timeframe: Timeframe, *, at: datetime | None = None) -> MarketState:
        instant = at or datetime.now(UTC)
        candle = await self.repository.candle_at(symbol, timeframe, instant)
        current = self.manager.current_provider
        stats = self.manager.statistics.get(current) if current else None
        return MarketState(
            market_open=self.sessions.is_open(instant),
            session=self.sessions.session_at(instant),
            current_provider=current,
            provider_health="healthy" if stats and stats.healthy else ("unavailable" if stats is None else "degraded"),
            current_latency_ms=stats.last_latency_ms if stats else None,
            data_freshness_seconds=max(0, (instant - candle.timestamp).total_seconds()) if candle else None,
            symbol=canonical_symbol(symbol),
            timeframe=timeframe,
            historical_sync_status=self.sync_status,
            realtime_status=self.realtime_status,
            as_of=instant,
        )

    async def close(self) -> None:
        await self.manager.close()

    async def _recover_gaps(self, symbol: str, timeframe: Timeframe, candles: list[Candle], limit: int) -> list[Candle]:
        if len(candles) < 2:
            return candles
        primary = self.manager.current_provider
        for provider_name in self.manager.rankings(symbol, timeframe):
            if provider_name == primary:
                continue
            try:
                alternate = await self.manager.history_from(
                    provider_name,
                    ProviderRequest(symbol=symbol, timeframe=timeframe, start=candles[0].timestamp, end=candles[-1].timestamp, limit=limit),
                )
            except Exception:
                continue
            merged = {(item.timestamp): item for item in candles}
            for item in alternate:
                if item.timestamp not in merged:
                    merged[item.timestamp] = self.validator.score(item, [], recovered=True)
            return sorted(merged.values(), key=lambda item: item.timestamp)
        return candles

    @staticmethod
    def _key(kind: str, symbol: str, timeframe: Timeframe, start: datetime | None = None, end: datetime | None = None, limit: int | None = None) -> str:
        return ":".join((kind, symbol, timeframe.value, start.isoformat() if start else "", end.isoformat() if end else "", str(limit or "")))


def build_market_data_service(config: MarketDataConfig | None = None) -> MarketDataService:
    resolved = config or MarketDataConfig()
    registry = ProviderRegistry()
    for item in resolved.providers:
        key = provider_api_key(item.api_key_env)
        if not item.enabled or not key:
            continue
        account_id = provider_api_key(item.account_id_env) if item.account_id_env else None
        if item.name == "twelve_data":
            registry.register(TwelveDataProvider(api_key=key, base_url=item.base_url, timeout_seconds=item.request_timeout_seconds, account_id=account_id))
        elif item.name == "alpha_vantage":
            registry.register(AlphaVantageProvider(api_key=key, base_url=item.base_url, timeout_seconds=item.request_timeout_seconds, account_id=account_id))
        elif item.name == "financial_modeling_prep":
            registry.register(FinancialModelingPrepProvider(api_key=key, base_url=item.base_url, timeout_seconds=item.request_timeout_seconds, account_id=account_id))
        elif item.name == "oanda":
            registry.register(OandaProvider(api_key=key, base_url=item.base_url, timeout_seconds=item.request_timeout_seconds, account_id=account_id))
    manager = ProviderManager(registry, preferred=resolved.preferred_provider, failure_threshold=resolved.provider_failure_threshold, recovery_successes=resolved.provider_recovery_successes)
    return MarketDataService(manager, config=resolved)
