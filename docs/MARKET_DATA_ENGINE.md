# Market Data Engine

The Market Data Engine is TEN's single source of truth for historical and live market observations. Downstream engines consume only the normalized `Candle` and `Tick` contracts; provider payloads never cross the adapter boundary.

## Architecture

The data path is provider adapter → provider manager → validator and quality scorer → repository and multi-layer cache → service/API/event bus. The existing engine registry, pipeline manager, feature store, event bus implementation, and plugin architecture are unchanged.

`ProviderRegistry` accepts any implementation of `MarketDataProvider`. `ProviderManager` filters by capability, ranks eligible providers using health, confidence, uptime, latency, freshness, and preferred-provider status, then fails over automatically. Consecutive successful probes restore a recovered provider.

Current adapters:

**Active by default — keyless public sources, no API key required:**

- LBMA Gold Price (`lbma_gold_price`) — the London Bullion Market Association's own daily AM/PM
  gold price fix. The authoritative, non-proxy industry benchmark; daily (`D1`) granularity only.
- Kraken (`kraken`) — public OHLC endpoint for PAXG/USD (Paxos Gold), a gold-token proxy instrument
  used for intraday coverage (`M1`–`H4`) and cross-source validation. Not true spot XAU/USD.
- OKX (`okx`) — public candles endpoint for XAUT/USDT (Tether Gold), a second, independently-issued
  gold-token proxy on a different exchange, for the same intraday/cross-validation role.

**Disabled by default — inert legacy adapters, kept in code but never constructed unless an
operator explicitly re-enables them:**

- TwelveData, Alpha Vantage, Financial Modeling Prep, OANDA — paid/keyed providers. Re-enable by
  setting `enabled: true` in `configs/market_data.yaml` and configuring the corresponding
  `TEN_TWELVE_DATA_API_KEY` / `TEN_ALPHA_VANTAGE_API_KEY` / `TEN_FMP_API_KEY` / `TEN_OANDA_API_KEY`
  (+ optionally `TEN_OANDA_ACCOUNT_ID`) Railway variable. A missing key means these are simply
  never constructed — not a degraded/error state.
- Yahoo Finance, Stooq, Binance — fully implemented but **permanently** disabled: all three hosts'
  `robots.txt` explicitly disallow automated access (Stooq's endpoint additionally gates real
  requests behind a JavaScript proof-of-work challenge). Each adapter also re-checks robots.txt at
  first use and refuses to fetch even if manually enabled, rather than silently bypassing that
  policy — do not enable without first re-verifying the current robots.txt yourself.
- Provider-neutral CSV history — unchanged, for offline/backtest datasets.

Every keyless adapter's `base_url` is SSRF-checked against an explicit domain allowlist at
construction time (`backend/app/engines/market_data_engine/ssrf.py`), matching the pattern
established in the economic calendar engine's public sources.

### Cross-source validation and quarantine

`MarketDataService.history()` best-effort fetches the same window from one alternate healthy,
capability-eligible provider and compares close prices (`MarketDataValidator.compare()`). A
deviation beyond `validation.cross_source_tolerance` (default 1%) is flagged as a
`PROVIDER_INCONSISTENCY` anomaly but the candle is still served; beyond
`validation.cross_source_quarantine_tolerance` (default 5%) the candle is treated as an implausible
outlier and **dropped** — never fabricated or silently kept — surfacing as a `missing_count=1`
anomaly so the existing gap-recovery path gets a chance to backfill it from a third source. If no
third source can, the gap is left standing rather than guessing. Cross-validation is best-effort:
a failed or unavailable alternate never blocks the primary result.

### No-lookahead / partial candles

Every new adapter guarantees a candle is only ever returned once its own period has fully elapsed
(`adapters._period_has_closed()`), computed independently of whatever a source's own API might
claim — OKX's explicit `confirm` flag and OANDA's `complete` flag are trusted as a first signal
where available, but the computed check always runs too as a source-independent backstop. No
adapter has ever emitted a still-forming candle as final. `Timeframe.H4` is synthesized by
deterministic OHLCV aggregation of four consecutive, already-final `H1` candles
(`adapters._aggregate_candles()`) only where a source has no native 4-hour interval (Kraken and OKX
both do; only the disabled Yahoo Finance legacy adapter uses this path) — this is arithmetic on
real, final data, not interpolation, and a bucket is only emitted once every one of its
constituent smaller candles is present.

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
