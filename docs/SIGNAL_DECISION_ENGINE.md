# Signal Decision Engine Production 1.0

## Mission and boundary

The Signal Decision Engine is TEN's deterministic policy and safety gate. AI Scoring answers what point-in-time evidence indicates; Signal Decision answers whether that persisted evidence satisfies the active analytical-use policy. It owns decision states, rules, policy versions, validity, duplicate suppression, cooldown, reversal protection, persistence, and explainability. It does not recalculate upstream analytics.

The engine never executes trades, creates orders, connects to brokers, calculates position size, or creates entry, stop-loss, or take-profit prices.

## Trusted inputs

Production requests identify a persisted `AIScoreSnapshot` by UUID. Raw score injection is not accepted. The service verifies instrument, timeframe, score status, policy/configuration identity, timestamp ordering, mode compatibility, and freshness. Optional typed references are loaded point-in-time from Economic Calendar and Market Regime services. Dependency health is classified as critical, required-for-eligibility, optional, or informational.

For `as_of = T`, only score, event, regime, and decision-history records available by `T` may be used. Missing historical context is explicit; current context is never silently substituted. All temporal behavior uses an injected clock.

## Decision states and precedence

New evaluations use this strict precedence:

1. `invalid`: snapshot, policy, timestamp, or configuration integrity failed.
2. `blocked`: at least one hard safety gate failed.
3. `insufficient_evidence`: valid evidence does not meet observation minimums.
4. `observe_only`: no hard blocker exists, but a soft eligibility gate failed.
5. `eligible`: every mandatory hard, evidence, and soft gate passed.

`expired` describes a persisted decision whose `valid_until` boundary has elapsed. Active lookups use `valid_from <= at < valid_until`; historical evidence is not mutated.

AI directional labels map exhaustively to `bullish`, `bearish`, or `neutral`. Direction is never described as buy, sell, long, or short. Neutral cannot be eligible.

## Rules

The approved, versioned rule registry evaluates stable IDs in deterministic order:

- policy and source-snapshot integrity;
- point-in-time validity and timeframe-specific freshness;
- directional strength, confidence, data quality, and evidence alignment;
- preferred and hard-block market-risk thresholds;
- structured AI Scoring conflict severity and penalty totals;
- Economic Calendar hard/caution phases;
- exhaustive Market Regime outcomes;
- critical and optional dependency health;
- repository duplicate checks;
- persisted same-direction cooldown;
- opposite eligible-direction reversal lock;
- configured temporal validity.

Every rule records category, severity, outcome, observed value, threshold, reason code, version, and evaluation timestamp. Hard failures cannot be overridden by an aggregate score.

## Threshold defaults

- Directional strength: observe at 20, eligible at 45.
- Confidence: observe at 45, eligible at 70. Confidence is evidence quality/agreement, not win probability.
- Risk: below 40 preferred, 40–64.9999 observe-only, 65 or above blocked.
- Data quality: below 10 invalid, observe at 45, eligible at 70.
- Alignment: observe at 35, eligible at 60.
- Severe conflict: explicit severe conflict or total penalty at least 25 blocks.

Exact boundaries are inclusive at the configured pass/block threshold and are exhaustively tested. Bullish and bearish strength are symmetric.

## Eligibility score

State comes from rules. The secondary display/ranking score is:

```text
eligibility_score = 100
  × directional_strength / 100
  × confidence / 100
  × data_quality / 100
  × alignment / 100
  × (1 − market_risk / 100)
  × freshness_factor
```

It is bounded to `[0, 100]`, rounded only at the output boundary, unsigned, deterministic, and unable to bypass a hard gate. It is not a profit or success probability.

## Events, regimes, conflicts, and dependencies

Economic Calendar is consumed through instrument-scoped context. `imminent`, `at_event`, and `overlapping` are hard-block phases; pre/post/cooldown phases are observation cautions. Unavailable event context fails closed by default. Upcoming events affect eligibility/risk, never directional sign.

Supported Market Regime labels are mapped exhaustively in configuration. Trending/expansion regimes may pass, balanced/ranging/transitional regimes are observe-only, and uncertain/insufficient regimes block. Unknown labels fail closed.

AI Scoring conflicts remain structured and visible. Moderate conflicts are soft gates; severe conflicts block. Critical persistence or trusted-score failures block. Feature Store and Event Bus failures degrade publication health but do not corrupt a durably persisted decision.

## Duplicate, cooldown, reversal, and hysteresis

The input fingerprint canonically includes trusted analytical evidence, policy identity, mode, and configuration hash while excluding transient request and health metadata. PostgreSQL uniquely constrains `(input_fingerprint, mode)`, making duplicate creation race-safe across instances. Exact duplicate requests return the existing decision and do not republish events.

Cooldown queries persisted history by instrument/timeframe/direction and historical cutoff. Default repeat windows are 900 seconds for eligible and 300 seconds for observe-only/blocked decisions. Reversal protection blocks a rapid opposite eligible direction for 600 seconds unless both configured strength and confidence improvements are met. Hysteresis reduces re-entry thresholds for an active same-direction eligible decision but never preserves eligibility through a hard block.

## Validity, expiration, and lifecycle

Every result has deterministic `valid_from` and `valid_until`. Eligible and observe-only validity is timeframe-specific; blocked, insufficient, and invalid results use short bounded defaults. New evidence creates a new immutable record linked through previous/supersedes identifiers. Active lookup evaluates expiration on demand. No process-local scheduler was added; distributed expiration/cleanup workers remain an operational extension.

## Replay

Replay requires an explicit `as_of`, trusted score snapshot, policy name/version, mode, and publication flags. Live events and live Feature Store writes are disabled by default. Replay fingerprints and persistence are mode-separated, current active decisions do not leak into earlier history, and repeated fixed-clock inputs produce identical semantic results.

## Persistence and multi-instance safety

PostgreSQL tables store decisions, normalized rule evaluations, and normalized reasons. Foreign keys use `RESTRICT` for trusted AI snapshots and `CASCADE` for decision-owned rules/reasons. Indexed instrument/timeframe/time queries support active lookup and bounded history. Transactional inserts and fingerprint uniqueness provide multi-instance deduplication. In-memory persistence exists only as an explicit degraded test/development adapter.

Retention is configuration-driven: 180 days for live decisions and 30 days for replay by default. Cleanup is bounded and is not run destructively at startup.

## Feature Store and Event Bus

After persistence, the Feature Store receives a compact versioned `signal_decision` record with state, direction, eligibility score, validity, policy, blocker/warning counts, and explicit `trade_execution=false`. Blocked decisions cannot appear as eligible.

The Event Bus emits one bounded state event per new persisted decision. Payloads contain identifiers and summary fields, not raw upstream payloads or secrets. Replay publication is opt-in. Exact duplicates do not republish.

## API

- `GET /signal-decisions/health`
- `GET /signal-decisions/config`
- `GET /signal-decisions/metrics`
- `POST /signal-decisions/evaluate`
- `POST /signal-decisions/replay`
- `GET /signal-decisions/latest`
- `GET /signal-decisions/history`
- `GET /signal-decisions/{decision_id}`
- `GET /signal-decisions/{decision_id}/rules`
- `GET /signal-decisions/{decision_id}/explanation`

History uses bounded offset cursors, page size, and date range. Blocked and insufficient evaluations are valid HTTP 200 responses. Missing snapshots return 404; invalid requests/policies return 422.

## Health, metrics, and logging

Health is healthy only when the service is initialized, trusted AI Scoring is available, and durable persistence is available. Optional publication/context failures produce degraded status. Invalid policy, unavailable AI Scoring, or required persistence failure produces unavailable status.

Metrics count requests, completions, failures, states, duplicates, persistence/publication failures, expiration runs, and bounded latency summaries without high-cardinality identifiers. Structured logs contain bounded engine, instrument, timeframe, direction, state, policy, rule counts, and mode context; secrets and raw payloads are excluded.

## Security and operational limitations

Models forbid unknown fields, validate UTC timestamps and bounded scores, reject invalid versions and policy names, and use parameterized SQLAlchemy statements. Configuration cannot execute expressions or dynamically import policies. APIs enforce UUIDs, enums, page/date bounds, and sanitized domain errors.

Production 1.0 intentionally has no process-local scheduler and no automatic tick subscription. On-demand evaluation works independently. Event-driven evaluation can be added only with a distributed, backpressured, database-coordinated worker. Source-group independence, confidence, and eligibility are policy semantics—not measured profitability.

## Safety statement

The Signal Decision Engine provides policy-controlled analytical decisions only. It does not execute trades, generate orders, calculate position sizes, or guarantee profitability. An eligible decision means configured analytical and safety gates passed; it is not a prediction of trading success.
