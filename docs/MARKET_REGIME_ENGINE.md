# Market Regime Engine Production 1.0

The Market Regime Engine is TEN's authoritative, probabilistic description of the current market environment. It classifies conditions; it never generates entries, exits, orders, position sizes, or forecasts. Every snapshot, feature, and event carries `probabilistic_inference: true` and `trading_instruction: false`.

## Ownership and dependency direction

The engine consumes time-valid public outputs in this direction: `Market Data -> SMC / Liquidity / Volume Profile -> Institutional Flow -> Market Regime`.

Market Data owns candles and raw observations. SMC owns structure. Liquidity owns pools and sweeps. Volume Profile owns value, POC, nodes, acceptance, and migration. Institutional Flow owns participation and campaign inference. Market Regime only synthesizes trend, volatility, auction, compression/expansion, lifecycle, persistence, maturity, transitions, multi-timeframe state, cross-session state, confidence, ambiguity, and explanations. Upstream packages never import Market Regime.

## Public models and taxonomy

`MarketRegimeSnapshot` is immutable, versioned, bounded, and replay-stable. It exposes separate dominant, trend, volatility, auction, expansion, structural, participation, inventory, lifecycle, persistence, and maturity dimensions. It includes score components, primary and alternative interpretations, transition state, MTF and session context, complete evidence traceability, degradation state, repository/recovery mode, and safety flags.

`MarketRegimeEvidence` retains source engine/version/object/snapshot identifiers, event and availability timestamps, analysis boundary, family, role, direction, raw and normalized strength, source confidence and quality, effective weight, correlation group/discount, decay, acceptance state, and rejection reason. Stable UUIDv5 identifiers and deterministic ordering derive only from semantic inputs.

## Temporal alignment, normalization, and correlation

Evidence with `available_at > historical_boundary` remains visible as unavailable and rejected with zero contribution. Candles beyond the boundary are excluded. Higher-timeframe and session contexts are accepted only when complete at the boundary. Evidence decay is calculated from the historical boundary, never wall-clock time.

Strength is bounded and combined with family weight, source confidence, source quality, age decay, and a configurable correlation-group cap. Multiple views of one causal event cannot create false independence. Original strength, effective weight, discounts, contradictory observations, and unavailable observations remain visible.

## Inference, confidence, and ambiguity

Trend requires independent evidence families and cannot be inferred from one structure event. Volume Profile is the principal auction source; without it, auction state is uncertain. Volatility level is historically normalized and distinct from direction, compression, and expansion. Compression requires converging contraction, overlap, and balance observations and never implies a guaranteed breakout. Expansion distinguishes early, established, late, and decelerating stages. Institutional Flow campaign state is consumed, not recreated, and remains explicitly probabilistic and “-like”.

Confidence is separate from directional strength. Exposed components cover evidence strength, quality, source/family diversity, temporal alignment, persistence, MTF/session alignment, contradiction, correlation, missing data, and instability. Ambiguity covers directional, timeframe, session, missing-source, diversity, and transition conflict. Every result includes an alternative interpretation.

## Persistence, recovery, transitions, and replay

Bounded in-memory and SQLAlchemy/PostgreSQL repositories implement snapshot, evidence, transition, checkpoint, history, latest-state, and pruning operations. Checkpoints use SHA-256 integrity and version compatibility checks. Memory mode is exposed as degraded when durable persistence is required.

`analyze_snapshot`, `update_incremental`, `replay`, and `recover` are supported. Snapshot/evidence/event IDs, ordering, normalized values, classifications, and semantic state are deterministic at equivalent boundaries. Transitions use none, watch, developing, confirmed, failed, and invalidated states and are persisted idempotently.

## Multi-timeframe and sessions

MTF synthesis is bounded, deterministically ordered, and reports included, excluded, unavailable, higher, and lower timeframes without overriding the requested timeframe. Cross-session synthesis reports continuation, handoff, reversal, mixed, or unavailable context and never uses an unfinished future session.

## API, features, events, health, and metrics

Read-only endpoints under `${TEN_API_PREFIX}/market-regime` expose health, config, metrics, state, history, snapshot lookup, trend, volatility, auction, compression, expansion, transitions, persistence, sessions, MTF, evidence, and explanations. `TEN_API_PREFIX` is empty by default. Pagination and evidence limits are bounded.

Feature Store values use the `market_regime` namespace with version and snapshot traceability. Typed stable events cover snapshot and dimension changes, compression/expansion, transition lifecycle, weakening/exhaustion risk, MTF conflict, session handoff, replay, recovery, and degradation. No signal or order event is emitted.

Health reports versions, initialization, repository/recovery mode, dependency state, latest classification/confidence/ambiguity, evidence, transitions, latency, failures, checkpoints, and replay. Metrics avoid unbounded symbol labels.

## Configuration, migration, and deployment

`configs/market_regime.yaml` versions weights, thresholds, evidence/correlation/decay bounds, persistence, processing/retention, repository mode, and MTF hierarchy. Invalid weights, threshold ordering, decay/caps, persistence windows, timeframes, repository modes, and version combinations are rejected.

PostgreSQL migration `migrations/20260718_market_regime_v1.sql` is additive and idempotent. Startup creates metadata, selects SQLAlchemy when available, validates recovery, registers routes, and exposes degradation. Shutdown closes the Market Regime session before disposing the shared engine. Railway uses the repository's established deployment configuration.

## Limitations and safety

The engine observes no verified participant identity or intent and makes no claim about future price. `EXHAUSTION_RISK`, compression, expansion, and transitions are analytical context only. Missing Market Data prevents safe analysis; missing optional upstream sources reduce confidence and are reported explicitly.
