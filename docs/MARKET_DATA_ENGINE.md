# Market Data Engine

The Market Data Engine is TEN's single source of truth for historical and live market observations. Downstream engines consume only the normalized `Candle` and `Tick` contracts; provider payloads never cross the adapter boundary.

## Architecture

The data path is provider adapter → provider manager → validator and quality scorer → repository and multi-layer cache → service/API/event bus. The existing engine registry, pipeline manager, feature store, event bus implementation, and plugin architecture are unchanged.

`ProviderRegistry` accepts any implementation of `MarketDataProvider`. `ProviderManager` filters by capability, ranks eligible providers using health, confidence, uptime, latency, freshness, and preferred-provider status, then fails over automatically. Consecutive successful probes restore a recovered provider.

Current adapters:

- TwelveData
- Alpha Vantage
- Financial Modeling Prep
- OANDA
- Provider-neutral CSV history

Set one or more Railway variables to activate HTTP adapters: `TEN_TWELVE_DATA_API_KEY`, `TEN_ALPHA_VANTAGE_API_KEY`, `TEN_FMP_API_KEY`, `TEN_OANDA_API_KEY`, and optionally `TEN_OANDA_ACCOUNT_ID`. The application remains bootable and reports degraded provider health when no secret is configured.

## Normalization and validation

Every candle contains UTC timestamp, canonical symbol, timeframe, OHLC, volume, spread, provider, quality score/level, and ingestion timestamp. Pydantic rejects missing/non-finite/negative values and invalid OHLC. Series validation rejects future, duplicate, and non-monotonic candles and records clock drift, missing intervals, market/weekend/holiday gaps, cross-provider inconsistencies, and deterministic volatility spikes.

Quality is explicit: native 100, verified 98, recovered 95, interpolated 90, minor anomaly 80, major anomaly 60, corrupted 40. Alternate-provider gap recovery is labelled `recovered`; data is never silently repaired.

## Cache and persistence

`MarketDataCache` combines a bounded LRU memory layer with durable JSON cold storage, independent historical/realtime TTLs, refresh/invalidation, eviction counts, and hit ratio. Cache files are reconstructable normalized candles and contain expiration metadata.

The repository port has deterministic in-memory and PostgreSQL/SQLAlchemy adapters. PostgreSQL uses indexed chronological reads and bulk `ON CONFLICT DO UPDATE` writes. Audit tables cover historical and realtime candles, provider metrics, latency, quality, gaps, synchronization, and cache metadata.

## Replay and time travel

Replay queries are bounded by an `as-of` timestamp and return the exact normalized records persisted at or before that moment. `/market/candle/{timestamp}` returns the last candle visible at the requested instant. `/market/session/{timestamp}` reconstructs the DST-aware session classification. No live provider query participates in replay, preserving repeatability.

## Sessions and metrics

Session classification uses IANA `Europe/London` and `America/New_York` time zones and distinguishes Asia, London, New York, overlap, weekend, holiday, and closed states. Metrics are informational only: ATR, current/average spread, daily/session range, rolling volatility, price velocity, tick frequency, freshness, and provider latency.

## Events

The module publishes typed events through TEN's existing bus: `MarketOpened`, `MarketClosed`, `SessionChanged`, `ProviderChanged`, `ProviderRecovered`, `GapDetected`, `HistoricalUpdated`, `RealtimeUpdated`, `NewCandle`, `DataCorrupted`, and `QualityChanged`.

## Provider extension guide

Implement `MarketDataProvider`, declare `ProviderCapabilities`, normalize all payloads inside the adapter, and register it with `ProviderRegistry`. No manager, service, API, pipeline, or downstream engine change is required.
