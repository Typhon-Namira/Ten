# Institutional Flow Engine Production 1.0

## Purpose and semantic boundary

The Institutional Flow Engine synthesizes time-valid Market Data, SMC, Liquidity, and Volume Profile evidence into explainable probabilistic inferences. It does not observe participant identity. With ordinary OHLCV or tick volume, the engine cannot prove that a bank, fund, market maker, or other institution acted.

Terms such as absorption-like, accumulation-like, distribution-like, initiative, responsive, and campaign phase mean “behavior consistent with the available evidence.” They are not claims about actual resting orders, inventory, intent, manipulation, or ownership. No output is a trading instruction.

## Architecture and ownership

The core analyzer is independent of FastAPI, SQLAlchemy, providers, Railway, and frontend code:

```text
Market Data + typed SMC + typed Liquidity + typed Volume Profile evidence
                               |
                               v
 temporal alignment -> normalization -> correlation discounting
                               |
                               v
 participation / activity / absorption / exhaustion / inventory
                               |
                               v
 pressure / persistence / campaign / session / MTF / explanation
                               |
                               v
 repository + checkpoint + Feature Store + Event Bus + read-only API
```

Market Data continues to own normalized OHLCV, sessions, calendars, volume semantics, and replay boundaries. SMC owns structure. Liquidity owns pools and sweep lifecycles. Volume Profile owns POC, Value Area, nodes, shapes, and migrations. Institutional Flow consumes typed references and never redetects those objects.

## Evidence model

Every evidence item carries a deterministic ID, source engine and object, evidence type, source and availability timestamps, timeframe, session, direction, strength, confidence, quality, causal role, correlation group, invalidation state, configuration version, and engine version.

Evidence becomes eligible only at its availability timestamp. Invalidated and low-quality evidence is rejected. Duplicate source objects are removed. Contributions beyond the configured maximum for a correlation group are discounted. This prevents related observations—such as BOS plus displacement or POC plus HVN—from being counted as fully independent proof.

Contradictory evidence is retained. Bullish and bearish weights, conflict, ambiguity, source diversity, alternative interpretation, and limitations are exposed rather than collapsed into an unexplained label.

## Analytical outputs

- Participation intensity combines bounded volume, range, structure, liquidity, profile, persistence, quality, and diversity evidence.
- Initiative activity describes movement away from accepted value with expansion or structural consequence.
- Responsive activity describes evidence-supported response near value, structure, nodes, or typed liquidity events.
- Absorption-like behavior requires elevated effort with limited progress, repeated tests, or rejection. It does not assert observed resting orders.
- Exhaustion-like behavior requires declining efficiency, limited progress, or structural failure. It is not a reversal signal.
- Accumulation-like and distribution-like behavior require multiple evidence families and preserve ordinary-balance and ambiguous alternatives.
- Reaccumulation-like and redistribution-like phases require time-valid prior campaign context.
- Directional pressure reports bullish, bearish, and neutral weight, net pressure, conflict, quality, persistence, and confidence.
- Flow persistence reports transient, developing, persistent, strengthening, weakening, or reversing relevance using bounded windows.
- Cross-session analysis uses Market Data session semantics and records continuation, reversal, or handoff only after the relevant evidence is available.
- Multi-timeframe analysis is bounded by configuration and excludes unavailable timeframes or failed/incomplete upstream states.

Campaign names are approximate model-based descriptions, not canonical Wyckoff phase claims.

## Temporal correctness, decay, and replay

The analysis boundary filters candles and evidence. Future SMC confirmation, Liquidity availability, completed profile state, session conclusion, mitigation, invalidation, or continuation cannot enter an earlier snapshot. Stable UUIDv5 identities include the boundary, mode, configuration, and evidence IDs. Prefix tests verify no-lookahead behavior and deterministic replay.

Current relevance uses bounded evidence windows and a configured per-candle decay factor. Historical evidence remains auditable; decay changes current contribution rather than rewriting historical truth.

## Persistence and recovery

`InstitutionalFlowRepository` has in-memory and SQLAlchemy implementations. PostgreSQL persists immutable snapshots, normalized evidence, and one conflict-safe checkpoint per symbol/timeframe/configuration version. Checkpoint payloads use JSON only and are verified with SHA-256 plus engine-version compatibility before recovery. Production health degrades when only ephemeral persistence is active.

Migration: `migrations/20260718_institutional_flow_v1.sql`.

## Feature Store and events

The `institutional_flow` feature namespace publishes participation, initiative, responsive, absorption-like, exhaustion-like, inventory, campaign, pressure, persistence, session flow, confluence, ambiguity, quality, evidence IDs, source boundary, and version metadata. It explicitly records `probabilistic_inference: true` and `trading_instruction: false`.

Typed events cover analysis updates/degradation, participation, initiative, responsive activity, absorption-like and exhaustion-like behavior, inventory, campaign, pressure, cross-session analysis, checkpoint recovery, and replay. Event IDs are deterministic per snapshot/object.

## REST API

All routes are GET-only under `/institutional-flow`:

- `/health`, `/metrics`, `/config`, `/snapshot`, `/state`, `/replay`
- `/participation`, `/initiative`, `/responsive`
- `/absorption`, `/exhaustion`, `/inventory`, `/campaign`
- `/pressure`, `/persistence`, `/cross-session`, `/confluences`
- `/explanation`, `/evidence`, `/mtf`

Symbol, timeframe, timestamp, evidence filters, pagination, and limits are validated and bounded. Errors do not expose internal stack traces.

## Configuration and observability

`configs/flow.yaml` versions weights, evidence limits, quality, decay, correlation caps, inference thresholds, persistence requirements, processing limits, and MTF depth. Metrics report analyses, degradation/failure, candles, accepted/rejected/deduplicated/discounted evidence, conflicts, analytical objects, replay, recovery, publication/persistence failures, average latency, p95 latency, repository mode, and latest successful analysis.

Health includes all four upstream dependencies, database/repository mode, checkpoint state, versions, last analysis, and degradation reasons. It never reports full health when configured production persistence is unavailable.

## Performance and limitations

Evidence normalization is bounded by `maximum_items`; correlation uses indexed group totals; activity calculations scan the bounded accepted tuple; confluence is grouped by direction; MTF depth is capped. Persistence queries are indexed and limited to a latest historical snapshot. Runtime is therefore bounded by configured candle/evidence/timeframe limits, but no universal linear-complexity claim is made.

Ordinary candle volume does not expose exact volume-at-price, aggressor identity, bid/ask delta, or institutional identity. Tick volume is not centralized exchange volume. All directional volume and participant-language outputs remain analytical approximations.
