# Smart Money Concepts Engine — Milestone 2A

TEN's SMC Engine converts normalized candles into deterministic structural facts. It is an analytical layer, not a signal generator, and does not claim to infer actual institutional intent. SMC terminology varies across methodologies; the definitions below are the exact TEN rules.

## Boundary and data flow

`MarketDataService → CandleContext → SwingDetector → StructureAnalyzer → SMCAnalysisSnapshot → existing FeatureStore/EventBus`

Only `MarketDataService` supplies candles. The engine has no provider, broker, network, normalization, session, cache, signal, execution, or AI integration. All read APIs are bounded and all persisted payloads contain engine and stable configuration versions.

## Deterministic methodology

The candle context requires one canonical symbol and timeframe, strictly increasing unique timestamps, and enough history for the configured pivot window. It calculates true range/ATR and propagates the existing candle quality score. Below-threshold input produces `degraded_input` and confidence penalties rather than being silently discarded.

Swings use configurable left/right pivot windows. A candidate at candle `i` is available only when candle `i + right_window` has closed. Stable identifiers include the source timestamp, type, timeframe, symbol, and configuration version. Minimum separation, absolute/ATR excursion, strength, internal/external classification, source IDs, quality, and confidence are recorded. Batch, incremental prefix, and replay processing therefore share the same availability rule.

Internal and external directions are maintained independently. A confirmed break in a neutral or matching direction is BOS. An opposing break is CHoCH and leaves the relevant scope transitional unless displacement crosses the MSS threshold and satisfies the configured protected/external rule; then it is MSS. Each structural level is broken at most once per analysis. The initial analysis boundary may seed a neutral-to-directional BOS when the final close confirms a break of the preceding range.

Close, wick, and hybrid confirmation are configurable, together with absolute and ATR-normalized break distance. Evidence contains the broken level, displacement result, input quality, thresholds, confirmation candle, invalidation rule, and detection version. Historical classification never uses wall-clock time.

## State, persistence, and replay

`SwingPoint`, `StructureLeg`, `StructureEvent`, `MarketStructureState`, and `SMCAnalysisSnapshot` are immutable Pydantic contracts. The in-memory repository supports idempotent chronological snapshots and time travel. The SQLAlchemy PostgreSQL adapter uses conflict-safe immutable writes to indexed `smc_objects` and `smc_analysis_snapshots` tables; `smc_checkpoints` stores bounded-recovery pointers. Schema creation follows TEN's existing SQLAlchemy metadata convention because this repository does not contain an Alembic migration framework.

Replay asks `MarketDataService.replay` for the visible candle prefix and derives a replay-mode snapshot. A swing cannot appear before `confirmed_at`; structural events cannot reference future confirmation candles. Historical corrections use the configured bounded recalculation window.

## Events, features, and API

The existing Event Bus receives typed swing, BOS, CHoCH, MSS, degraded-input, analysis-updated, and replay-completed events. Stable snapshot IDs make repeat publication idempotent within the service lifecycle. The existing Feature Store receives direction, internal/external direction, active/protected levels, last structural event IDs, confidence, quality, analytical timestamp, and version traceability. No entry, exit, stop, target, size, order, recommendation, or profitability field is published.

Read-only endpoints are `/api/v1/smc/state`, `/swings`, `/structure`, `/events`, `/snapshot`, `/replay`, `/health`, `/metrics`, and `/config`. Filters and query sizes are validated by FastAPI and capped at 5,000 objects/candles.

## Configuration and complexity

`configs/smc.yaml` defines pivot windows, separation, excursion, sensitivity, external strength, equal-level tolerance, confirmation method, break distance, displacement/MSS thresholds, minimum history, quality threshold, batch/recalculation/checkpoint limits, and maximum active objects. The SHA-256-derived configuration version changes whenever serialized settings change.

Context, pivot detection, and structure processing are linear in candle count apart from chronological output ordering (`O(n log n)` worst case); active lookups are bounded. Repository queries use `(symbol, timeframe, analysis_timestamp)` indexes.

## Milestone boundary and limitations

Milestone 2A delivers swings, internal/external structure, BOS, CHoCH, MSS, explicit state, evidence/confidence, persistence, replay/time travel, events, features, APIs, metrics, and tests. It intentionally does not expose placeholder zone models or routes. Displacement is currently structural confirmation evidence, not a standalone published object.

Milestone 2B will add displacement objects, FVG/inversion FVG, liquidity voids, Order Blocks, Breakers, Mitigation Blocks, and their complete lifecycles. Milestone 2C will add equal levels, inducement, dealing ranges, premium/discount/equilibrium, nested/MTF structure, expanded confidence, restart checkpoint activation, and performance benchmarking.
