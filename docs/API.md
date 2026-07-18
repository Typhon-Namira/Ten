# HTTP API

The API is read-only in the initial foundation.

SMC Production 1.0 exposes bounded read-only `${TEN_API_PREFIX}/smc/state`, `/swings`, `/structure`, `/events`, `/displacements`, `/zones`, `/liquidity-references`, `/dealing-ranges`, `/multi-timeframe`, `/snapshot`, `/replay`, `/health`, `/metrics`, and `/config` endpoints. `TEN_API_PREFIX` is empty by default. See [SMC Engine](SMC_ENGINE.md) for filtering and deterministic semantics.

Liquidity Production 1.0 exposes bounded read-only `${TEN_API_PREFIX}/liquidity/health`, `/metrics`, `/config`, `/state`, `/snapshot`, `/levels`, `/equal-levels`, `/pools`, `/events`, `/sweeps`, `/grabs`, `/raids`, `/stop-hunts`, `/false-breaks`, `/sessions`, `/reference-levels`, `/confluences`, `/targets`, `/map`, `/mtf`, and `/replay`. Collection queries support bounded offset/limit pagination; timestamp queries return only snapshots available at that boundary. No Liquidity mutation route exists. See [Liquidity Engine](LIQUIDITY_ENGINE.md).

Volume Profile Production 1.0 exposes bounded read-only `${TEN_API_PREFIX}/volume-profile/health`, `/metrics`, `/config`, `/state`, `/snapshot`, `/profiles`, `/developing`, `/completed`, `/fixed-range`, `/sessions`, `/daily`, `/weekly`, `/monthly`, `/composite`, `/anchored`, `/poc`, `/value-area`, `/hvn`, `/lvn`, `/shelves`, `/gaps`, `/shapes`, `/migrations`, `/confluences`, `/mtf`, and `/replay`. Fixed ranges require validated start/end timestamps and enforce configured candle/day limits. See [Volume Profile Engine](VOLUME_PROFILE_ENGINE.md).

Market Regime Production 1.0 exposes bounded read-only `${TEN_API_PREFIX}/market-regime/health`, `/config`, `/metrics`, `/state`, `/history`, `/snapshots/{snapshot_id}`, `/trend`, `/volatility`, `/auction`, `/compression`, `/expansion`, `/transitions`, `/persistence`, `/sessions`, `/mtf`, `/evidence`, and `/explanations`. Responses are probabilistic analytical context and never trading instructions. See [Market Regime Engine](MARKET_REGIME_ENGINE.md).

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
## Institutional Flow

Read-only endpoints under `/institutional-flow` expose health, metrics, configuration, snapshots, historical replay, evidence, participation, initiative/responsive activity, absorption-like and exhaustion-like behavior, inventory/campaign inference, directional pressure, persistence, cross-session flow, confluence, explanations, and bounded multi-timeframe context. These are probabilistic analytical observations, not verified participant identity or trading instructions.

## Economic Calendar

Read-only endpoints under `/economic-calendar` expose health, sanitized configuration, metrics, provider capability/status, bounded events and historical reconstruction, observations, revisions, upcoming/recent/active views, snapshots/history, symbol context, clusters, conflicts, and explanations. List windows and page sizes are bounded; `as_of` timestamps reconstruct only state available at that boundary. See [ECONOMIC_CALENDAR_ENGINE.md](ECONOMIC_CALENDAR_ENGINE.md).
