# HTTP API

The API is read-only in the initial foundation.

SMC Milestone 2A adds bounded read-only `/api/v1/smc/state`, `/swings`, `/structure`, `/events`, `/snapshot`, `/replay`, `/health`, `/metrics`, and `/config` endpoints. See [SMC Engine](SMC_ENGINE.md) for filtering and deterministic semantics.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process liveness |
| GET | `/signals` | Recent generated scenarios |
| GET | `/signals/latest` | Newest scenario or 404 |
| GET | `/engines/status` | Registered engine readiness |
| GET | `/market/status` | Indicative UTC weekday/session status |

Interactive OpenAPI documentation is available at `/docs` when the backend is running.
# Market Data API

All market-data routes are provider-neutral.

- `GET /market/latest`
- `GET /market/history`
- `GET /market/replay?at=...`
- `GET /market/candle/{timestamp}`
- `GET /market/session/{timestamp}`
- `GET /market/providers`
- `GET /market/provider/status`
- `GET /market/provider/statistics`
- `GET /market/state`
- `GET /market/metrics`
- `GET /market/health`

Series routes accept `symbol`, `timeframe`, and bounded `limit` parameters. Historical requests also accept `start`, `end`, and `refresh`. Timestamps must be ISO-8601 values with an explicit UTC offset. A refresh requiring an unavailable live provider returns HTTP 503; an absent persisted candle or metric returns HTTP 404.
