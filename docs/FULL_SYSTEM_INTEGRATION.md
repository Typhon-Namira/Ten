# TEN Full-System Integration Production 1.0

TEN now coordinates the existing ten engines from final Market Data candles to persisted AI Scoring and Signal Decision outputs. Integration is infrastructure, not an eleventh engine. Outputs are analytical only: the platform has no order, brokerage, position, leverage, or execution path.

```text
Production provider -> Market Data -> canonical final-candle envelope
  -> SMC / Liquidity / Volume Profile / Institutional Flow
  -> Market Regime + Economic Calendar -> coherent snapshot barrier
  -> AI Scoring -> Signal Decision -> operational signal + trace -> API/dashboard
```

The canonical envelope is immutable, versioned, timezone-aware, provider-attributed, deterministically hashed, and mode-tagged. Live consumers fail closed on Replay envelopes. Delivery is explicitly at least once. Event identity, processed-event identity, snapshot hashes, and operational-signal semantic hashes make consumers idempotent. The SQL migration defines the transactional event/outbox and durable audit schema; the in-memory adapter is intentionally degraded and is for development/tests only.

Production topology uses an API service, PostgreSQL, one integration/outbox worker, a Market Data worker when provider polling is enabled, and a Replay worker. Do not enable an embedded integration worker in multiple API replicas. Railway's API start command remains `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`; worker services should use their dedicated process command once provisioned. `/integration/readiness` stays unavailable while persistence is ephemeral.

Read endpoints require a viewer API key when `TEN_PUBLIC_READ_ACCESS=false`. Graph, traces, and manual outbox dispatch require admin. `TEN_API_KEYS` is a JSON mapping of secret API keys to `viewer`, `operator`, or `admin`; secrets must only be supplied by the deployment environment.

Operational APIs:

- `GET /integration/health`
- `GET /integration/readiness`
- `GET /integration/signals`
- `GET /integration/signals/latest`
- `GET /integration/traces/{trace_id}` (admin)
- `GET /integration/graph` (admin)
- `POST /integration/outbox/run-once` (admin)

The dashboard retrieves the same operational-signal representation exposed by the API. Stale, rejected, incomplete, or missing evidence cannot be represented as a fresh eligible signal. Replay remains isolated in the existing Replay Engine and cannot publish into the live integration consumer.
