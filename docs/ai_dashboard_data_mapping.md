# TEN AI Dashboard data mapping

All values below are backend-authoritative. Refresh is every five seconds; failed refreshes retain the last good value and surface an error. Market evidence is marked stale when diagnostics report `stale` or the latest candle timestamp is absent.

| UI surface | Endpoint | Primary backend fields |
| --- | --- | --- |
| Header and market status | `GET /api/v1/system/market-intelligence` | instrument, timeframe, latest candle timestamp, session, current price, diagnostics |
| Final decision hero | `GET /api/v1/ai-reasoning/latest` | latest final action, latest publication, runtime operating profile |
| Decision pipeline | Market intelligence, latest quant forecast, latest AI reasoning | evidence availability, forecast status, proposal, final action, publication |
| Market State | `GET /api/v1/system/market-intelligence` | multi-timeframe snapshot, structure, volatility, liquidity, SMC, session, freshness |
| Quantitative Forecast | `GET /api/v1/quant-forecasts/latest` | direction probabilities, expected return, horizon, uncertainty, status |
| AI Market Reasoning | `GET /api/v1/ai-reasoning/latest` | latest forecast direction, confidence, uncertainty, scenarios, reasoning summary |
| Deterministic Guardrails | `GET /api/v1/ai-reasoning/latest` | final action gate evaluations, modifications, approval/publication state |
| Active Analytical Signal | `GET /api/v1/ai-reasoning/latest` | latest publication, managed signal, lifecycle history, outcome |
| Performance | `GET /api/v1/ai-reasoning/latest` | proposal/publication performance reports and validation histories |
| Calibration | `GET /api/v1/quant-forecasts/calibration/latest` | sample count, status, Brier score, log loss, expected calibration error |
| Readiness and health | `GET /api/v1/ai-reasoning/latest` | production readiness, persistence, guardrail, LLM usage, runtime flags |

## Presentation rules

- `null`, missing, degraded, stale, and insufficient-sample values are rendered explicitly.
- Confidence and probabilities are displayed only from persisted backend values.
- The frontend never converts missing evidence to zero and never recomputes eligibility.
- Direction/action colors are semantic: green for favorable/approved states, red for blocked/negative states, amber for degraded or monitoring states, and gray for unavailable states.
- Publication is always labeled analytical. The interface permanently states that broker execution is unavailable.

## API changes made for the redesign

Only read-only response metadata was added:

- AI reasoning health/latest responses expose the current runtime profile and feature-flag snapshot.
- Final-decision health exposes configured daily request/token allowances and LLM concurrency.
- SPA fallback routes were registered for the new top-level frontend destinations.

No analytical contract, scoring threshold, hard gate, persistence schema, feature-flag default, or publication transition changed.
