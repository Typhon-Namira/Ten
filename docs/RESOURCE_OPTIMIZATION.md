# TEN resource-optimization report

Date: 2026-07-23
Scope: local deterministic profiling and implementation. No Railway deployment was performed.

## Executive result

TEN now suppresses duplicate closed-candle persistence and publication, schedules each enabled
timeframe against its own next close, recovers missed candles in timestamp order, bounds
non-authoritative process memory, uses a smaller configurable PostgreSQL pool, reduces repetitive
INFO logging, and refreshes the Economic Calendar hourly with an adaptive safety interval.

No SMC, liquidity, volume-profile, institutional-flow, market-regime, AI-scoring,
confidence, risk, decision, publication-threshold, or replay algorithm was changed. The effective
integration input remains 2,500 canonical candles. This report does **not** claim complete proof of
analytical equivalence: the repository does not yet contain the requested comprehensive
before/after golden-output corpus, and production evidence is unavailable until deployment.

## Measured local baseline and result

`scripts/resource_probe.py` runs 600 identical closed-M1 provider responses through the real market
data worker poll path. The baseline source was commit `01e5da5`; both runs used the same Python
environment and scenario.

| Metric | Baseline | Optimized |
|---|---:|---:|
| Iterations/provider calls | 600 | 600 |
| Elapsed time | 1.362 s | 0.243 s |
| Process CPU | 1.344 s | 0.234 s |
| RSS before | 72,302,592 B | 70,840,320 B |
| RSS delta | 4,005,888 B | 208,896 B |
| Python heap growth | 2,189,828 B | 12,436 B |
| Python heap peak | 2,208,149 B | 29,364 B |
| Events published/retained | 1,200 / 1,200 | 2 / 2 |
| Realtime observations retained | 600 | 1 |
| Durable candles | 1 | 1 |
| In-memory candle-cache entries | 1 | 1 |
| INFO records/bytes | 600 / 14,400 | 0 / 0 |
| Active asyncio tasks | 1 | 1 |
| Threads | 1 | 1 |
| Event-loop lag mean/p95/max | 9.18/15.41/16.56 ms | 16.56/29.01/29.64 ms |

The lag sample is Windows scheduler noise in a sub-second synthetic run and is not evidence of an
improvement. The direct probe intentionally invokes the provider 600 times to isolate duplicate
response handling; the production scheduler now avoids not-due M5/M15 calls, but provider
requests/hour must be measured after deployment.

The baseline required by the brief for idle dashboard, live M1/M5/M15 overlap, calendar sync,
bootstrap, retry, production DB query rate, pool concurrency, queue age, and per-engine executions
cannot be reconstructed from a local process without production telemetry. `/performance` now
exposes pool, session, cache, feature-store, activity-buffer, and poll metrics needed to collect it.

## Confirmed causes and changes

- Every identical provider response was persisted, cached, and emitted as both `NewCandle` and
  `RealtimeUpdated`. A 600-poll run therefore retained 1,200 events and 600 realtime observations.
  Canonical identity plus exact market-value comparison now returns without a write or event.
- The worker polled every symbol/timeframe on every fixed tick. It now records an independent
  next-close time per series. M15 is not externally polled or analyzed on every M1 close.
- Overlapping calls for one instrument/timeframe are serialized. Failures use exponential backoff
  with jitter, capped at 300 seconds.
- A timestamp gap triggers an incremental provider-history request; missing closed candles are
  validated, persisted, and published once in timestamp order.
- Late corrections preserve the existing policy: update durable/realtime state and publish a
  realtime correction, but do not duplicate `NewCandle` analysis.
- Per-poll success, heartbeat, and unchanged-market messages moved from INFO to DEBUG. Errors,
  warnings, terminal cycle events, decisions, failure details, and identifiers remain intact.
  Uvicorn access logs default off and remain configurable.

## Memory and cache inventory

| Collection | Bound | Eviction/source of truth |
|---|---:|---|
| Event-bus diagnostic history | 1,000 | Oldest; durable event/outbox state remains authoritative |
| Feature-store process cache | 10,000 | Oldest; pipeline/durable repositories remain authoritative |
| Market-data memory-cache keys | 256 | LRU; durable file/DB/provider can reload |
| Dashboard activity events | 500 | Oldest; diagnostic stream only |
| Per-client SSE queue | 500 | Oldest replaceable item; dropped/coalesced metrics exposed |
| AI duration samples | 500 | Oldest; metrics only |
| Signal-decision duration samples | 500 | Oldest; metrics only |
| Per-engine publication identities | 10,000 | Oldest; durable idempotency remains authoritative |
| Calendar identity map | 100,000 | Oldest, aligned with durable event retention |

All bounds are startup-validated. Current size, capacity, and eviction/drop counters are exposed
where the collection has operational relevance. The activity stream still preserves the newest
terminal event when a client is slow.

## Historical context and analytical quality

The largest effective live input is `TEN_MARKET_DATA_BOOTSTRAP_CANDLES=2500`, passed into the
integration `maximum_candles` limit. Engine configuration maxima are larger, but the integration
previously supplied 2,500 and still supplies 2,500. No indicator, swing, ATR, BOS/CHOCH, FVG,
order-block, profile, regime, AI, confidence, decision, risk, or publication window changed.

Regression tests verify unchanged OHLCV, one event per M1/M5/M15 canonical close, correction
policy, missed-candle recovery, duplicate suppression, and existing analytical suites. However,
the complete requested semantic golden corpus spanning every listed market state and every
downstream output was not available before the refactor and has not been fabricated afterward.
Consequently, complete analytical-quality preservation remains a rollout gate rather than a claim.

## PostgreSQL

| Setting | Previous SQLAlchemy default | New default |
|---|---:|---:|
| Pool size | 5 | 3 |
| Max overflow | 10 | 2 |
| Pool timeout | 30 s | 30 s |
| Pool recycle | Unlimited/default | 1,800 s |
| Pre-ping | Enabled | Enabled |
| Statement timeout | Unset | 30,000 ms |
| Idle-in-transaction timeout | Unset | 30,000 ms |

Sessions remain sequential per unit of work and are created from one shared session factory, never
shared concurrently. Session opened/completed/failed counts, active/peak sessions, and rolling
duration average/p95 are now reported. Pool checked-in, checked-out, size, and overflow are exposed.
Actual production peak checkout, wait duration, query/minute, rollback count, and timeout count
still require deployed PostgreSQL telemetry; the pool must not be reduced again without it.

## Economic Calendar

Normal synchronization changed from six hours to a configurable 3,600 seconds. Startup performs an
immediate sync. If a high/critical event is within the configured two-hour lookahead, refresh
temporarily changes to 300 seconds. Failures retry from 60 seconds with bounded exponential backoff
up to 900 seconds. The existing lock prevents overlapping synchronization.

Metrics persist/report last attempt, last success, next scheduled sync, coverage, relevant upcoming
event, provider/parser status, failure reason, and duration. Existing unavailable/fail-closed
semantics and revision handling remain unchanged.

## Configuration

- `TEN_MARKET_DATA_POLL_SECONDS=10`
- `TEN_MARKET_DATA_IDLE_POLL_SECONDS=30`
- `TEN_MARKET_DATA_PROVIDER_BACKOFF_MAX_SECONDS=300`
- `TEN_DB_POOL_SIZE=3`
- `TEN_DB_MAX_OVERFLOW=2`
- `TEN_DB_POOL_TIMEOUT_SECONDS=30`
- `TEN_DB_POOL_RECYCLE_SECONDS=1800`
- `TEN_DB_POOL_PRE_PING=true`
- `TEN_DB_STATEMENT_TIMEOUT_MS=30000`
- `TEN_DB_IDLE_TRANSACTION_TIMEOUT_MS=30000`
- `TEN_MAX_EVENT_HISTORY_SIZE=1000`
- `TEN_MAX_FEATURE_STORE_ENTRIES=10000`
- `TEN_MAX_DASHBOARD_EVENT_BUFFER=500`
- `TEN_MAX_CLIENT_QUEUE_SIZE=500`
- `TEN_LOG_ACCESS_REQUESTS=false`
- `TEN_LOG_MARKET_TICK_EVENTS=false`
- `TEN_LOG_HEALTH_UNCHANGED=false`
- Calendar YAML: normal 3,600 s, adaptive 300 s, two-hour lookahead, retry 60–900 s.

## Validation

- Focused market-data, calendar, integration, and resource tests: pass.
- Full backend suite: 559 passed, 1 skipped (560 collected).
- Ruff: pass across `backend`, `tests`, and `scripts`.
- Mypy: 14 pre-existing errors in 11 files, exactly matching the baseline; no new errors.
- TypeScript project build: pass.
- Vite production build: pass, 1,720 modules, 299.57 kB JS (88.36 kB gzip).
- Soak: 600 repeated provider responses reach constant retained state: one durable candle, one
  realtime observation, two events, one cache entry, one task, and one thread.

## Files changed

Core changes cover settings, application wiring, bounded collections, event/feature stores, market
data cache/service/worker, session instrumentation, system performance metrics, activity streaming,
Economic Calendar scheduling, engine diagnostic identity caches, AI/decision duration samples,
Docker logging defaults, configuration YAML, the resource probe, and regression tests.

## Railway rollout and limits

Keep one replica. Do not force a 512 MB limit or split API/worker yet. Deploy with the current
limits, collect average and peak RSS through at least an M1/M5/M15 overlap plus a calendar sync,
then set memory above the measured stable peak with a 25–40% margin. Retain the current limit during
the first optimization deployment; lower it only after confirming no OOM restart, pool starvation,
missed close, delayed decision, or calendar staleness. CPU limits likewise require deployed peak
and decision-latency evidence.

## Remaining production gates

1. Build and run the complete requested pre/post golden-output corpus before claiming analytical
   equivalence.
2. Deploy without changing resource limits.
3. Verify one and only one completed cycle, decision, reasons, confidence, risk, and publication
   result for live M1, M5, and M15 closes.
4. Verify calendar interval, adaptive state, provider status, coverage, and fail-closed behavior.
5. Record before/after Railway RSS/CPU, provider/calendar calls, DB queries/checkouts/waits,
   executions by engine/timeframe, log bytes, queue backlog/age, and latency average/p95.
6. Choose Railway memory/CPU limits only from those production measurements.
