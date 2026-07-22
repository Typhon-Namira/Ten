"""Single-source-of-truth facade for historical, realtime, replay, and state queries."""

import logging
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from backend.app.events import EventBus, InMemoryEventBus

from .adapters import (
    AlphaVantageProvider,
    BinanceProvider,
    FinancialModelingPrepProvider,
    HttpMarketDataProvider,
    KrakenGoldProxyProvider,
    LbmaGoldPriceProvider,
    OandaProvider,
    OkxGoldProxyProvider,
    StooqProvider,
    TwelveDataProvider,
    YahooFinanceProvider,
    provider_api_key,
)
from .cache import MarketDataCache
from .config import MarketDataConfig
from .events import GapDetected, HistoricalUpdated, NewCandle, RealtimeUpdated
from .manager import ProviderManager, ProviderRegistry
from .metrics import calculate_metrics
from .models import Candle, MarketMetrics, MarketState, RealtimeStatus, SyncStatus, Timeframe, canonical_symbol
from .providers import ProviderRequest
from .repository import InMemoryMarketDataRepository, MarketDataRepository
from .sessions import MarketSessionEngine
from .validation import DataAnomaly, MarketDataValidator, ValidationReport

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
            report, cross_check_anomalies = await self._apply_cross_validation(symbol, timeframe, report, limit)
            if any(anomaly.missing_count for anomaly in report.anomalies) or any(anomaly.missing_count for anomaly in cross_check_anomalies):
                recovered = await self._recover_gaps(symbol, timeframe, report.candles, limit)
                if recovered != report.candles:
                    report = self.validator.validate(recovered)
            if cross_check_anomalies:
                report = report.model_copy(update={"anomalies": [*report.anomalies, *cross_check_anomalies]})
            if report.anomalies:
                # One aggregated WARNING per `history()` call, never one line per anomaly — a
                # large backfill (hundreds to thousands of candles) can legitimately surface many
                # anomalies in one pass, and logging each individually is exactly what flooded
                # production logs badly enough that Railway's own platform-level log-rate-limiter
                # started dropping messages outright (426 dropped in one reported burst), silently
                # eating whatever real error might have logged alongside them. Full per-anomaly
                # detail is still available, just at DEBUG instead of WARNING.
                counts = Counter(anomaly.type.value for anomaly in report.anomalies)
                logger.warning(
                    "market_data_anomaly_batch",
                    extra={"symbol": symbol, "timeframe": timeframe.value, "anomaly_count": len(report.anomalies), "by_type": dict(counts)},
                )
                for anomaly in report.anomalies:
                    logger.debug("market_data_anomaly", extra={"anomaly": anomaly.model_dump(mode="json"), "symbol": symbol, "timeframe": timeframe.value})
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

    async def _apply_cross_validation(self, symbol: str, timeframe: Timeframe, report: ValidationReport, limit: int) -> tuple[ValidationReport, list[DataAnomaly]]:
        """Best-effort cross-source validation: fetch the same window from one alternate healthy
        provider and compare close prices via `MarketDataValidator.compare()`. A quarantined
        (implausible-outlier) candle is dropped from `report.candles` here — never fabricated back
        in — and surfaces as a `missing_count=1` anomaly so the existing gap-recovery path
        (`_recover_gaps`) gets a chance to backfill it from a third source; if none exists, the gap
        is left standing. Returned anomalies are handed back separately from `report` because a
        subsequent gap-recovery re-`validate()` call would otherwise silently discard them."""
        if not report.candles:
            return report, []
        primary = self.manager.current_provider
        alternate_name = next((name for name in self.manager.rankings(symbol, timeframe) if name != primary), None)
        if alternate_name is None:
            return report, []
        try:
            alternate_candles = await self.manager.history_from(
                alternate_name,
                ProviderRequest(symbol=symbol, timeframe=timeframe, start=report.candles[0].timestamp, end=report.candles[-1].timestamp, limit=limit),
            )
        except Exception:
            # Cross-validation is a quality enhancement, not a correctness requirement — an
            # unhealthy or unreachable alternate must never block the primary result.
            return report, []
        if not alternate_candles:
            return report, []
        inconsistencies, quarantined = self.validator.compare(report.candles, alternate_candles)
        if not quarantined:
            return report, inconsistencies
        logger.warning(
            "market_data.cross_validation.quarantined",
            extra={"symbol": symbol, "timeframe": timeframe.value, "quarantined_count": len(quarantined), "primary_provider": primary, "alternate_provider": alternate_name},
        )
        survivors = [item for item in report.candles if item.timestamp not in quarantined]
        trimmed = ValidationReport(candles=survivors, anomalies=report.anomalies, valid=False)
        return trimmed, inconsistencies

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


#: Keyless public-source providers — constructed with no API key and no env var lookup at all.
#: Includes the permanently disabled-by-default, robots.txt-blocked legacy adapters (Yahoo Finance,
#: Stooq, Binance): they need no key either, so they belong in this branch, not the keyed one below;
#: `enabled: false` in `configs/market_data.yaml` is what actually keeps them out of the registry.
#: Typed as `Callable[..., HttpMarketDataProvider]`, not `type[HttpMarketDataProvider]` — each
#: concrete adapter's `__init__` narrows the base class's signature (no `api_key`/`account_id`
#: required), which is a Liskov-safe narrowing but not a subtype match under `type[Base]`.
_PUBLIC_SOURCE_PROVIDERS: dict[str, Callable[..., HttpMarketDataProvider]] = {
    "lbma_gold_price": LbmaGoldPriceProvider,
    "kraken": KrakenGoldProxyProvider,
    "okx": OkxGoldProxyProvider,
    "yahoo_finance": YahooFinanceProvider,
    "stooq": StooqProvider,
    "binance": BinanceProvider,
}


def build_market_data_service(config: MarketDataConfig | None = None) -> MarketDataService:
    resolved = config or MarketDataConfig()
    registry = ProviderRegistry()
    for item in resolved.providers:
        if not item.enabled:
            continue
        if item.name in _PUBLIC_SOURCE_PROVIDERS:
            provider_class = _PUBLIC_SOURCE_PROVIDERS[item.name]
            registry.register(provider_class(base_url=item.base_url, timeout_seconds=item.request_timeout_seconds, requests_per_minute=item.quota_per_minute))
            continue
        key = provider_api_key(item.api_key_env) if item.api_key_env else ""
        if not key:
            continue
        account_id = provider_api_key(item.account_id_env) if item.account_id_env else None
        if item.name == "twelve_data":
            registry.register(TwelveDataProvider(api_key=key, base_url=item.base_url, timeout_seconds=item.request_timeout_seconds, account_id=account_id, requests_per_minute=item.quota_per_minute))
        elif item.name == "alpha_vantage":
            registry.register(AlphaVantageProvider(api_key=key, base_url=item.base_url, timeout_seconds=item.request_timeout_seconds, account_id=account_id, requests_per_minute=item.quota_per_minute))
        elif item.name == "financial_modeling_prep":
            registry.register(FinancialModelingPrepProvider(api_key=key, base_url=item.base_url, timeout_seconds=item.request_timeout_seconds, account_id=account_id, requests_per_minute=item.quota_per_minute))
        elif item.name == "oanda":
            registry.register(OandaProvider(api_key=key, base_url=item.base_url, timeout_seconds=item.request_timeout_seconds, account_id=account_id, requests_per_minute=item.quota_per_minute))
    manager = ProviderManager(registry, preferred=resolved.preferred_provider, failure_threshold=resolved.provider_failure_threshold, recovery_successes=resolved.provider_recovery_successes)
    return MarketDataService(manager, config=resolved)
