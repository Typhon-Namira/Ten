# AI Scoring Engine Production 1.0

## Mission and boundary

The AI Scoring Engine combines point-in-time snapshots from TEN's Market Data, SMC, Liquidity, Volume Profile, Institutional Flow, Market Regime, and Economic Calendar engines. Production 1.0 is a deterministic weighted policy. It does not call an LLM, duplicate upstream detection, execute orders, manage accounts, or create trading instructions.

The engine provides analytical intelligence only. It does not execute trades, recommend position sizes, or guarantee outcomes. Confidence reflects evidence quality and agreement, not trade win probability.

## Score dimensions and formulas

Each source is normalized to direction `[-1,1]` and confidence, risk, and quality `[0,1]`. Source timestamps, observation timestamps, publication timestamps, evidence IDs, versions, degradation, and reason codes remain explicit.

- Direction is `100 × Σ(direction × normalized effective directional weight)`. Effective weight includes configured weight, source quality, and freshness.
- Alignment is the absolute weighted directional balance. Equal opposing evidence approaches zero; evidence on one side approaches 100.
- Data quality is configured-source completeness multiplied by average source quality and freshness. Missing evidence is never treated as neutral.
- Confidence is availability × freshness × quality × alignment adjustment × independent-group coverage, less bounded conflicts. Single-source, stale, and degraded ceilings apply.
- Risk is the normalized weighted sum of source risk, plus bounded stale-input and conflict adjustments. Risk never changes directional sign.
- Composite is `direction × confidence_factor × (1 - 0.25 × risk_factor)`. It is analytical strength, not expected return.

Output values are bounded, finite, rounded only at the output boundary, and normalize negative zero.

## Independence, missing data, and conflicts

Sources are grouped as data, structure, participation, context, and event risk. SMC/Liquidity and Volume Profile/Institutional Flow therefore cannot inflate independent-source confidence as if they were unrelated. The default requires two directional sources across two groups. Missing sources reduce availability and quality; expired evidence cannot satisfy the minimum.

Pairwise directional gaps of 1.20 or greater create structured conflicts. Gaps of 1.60 or greater are severe. Conflict IDs, ordering, description codes, and penalties are deterministic.

## Freshness and point-in-time safety

Every source has configured fresh and stale thresholds. Aging evidence decays; stale evidence is heavily discounted; expired evidence has zero directional weight. Evidence with observation or publication time after `as_of` is rejected. Historical collection requests each upstream service at the historical boundary and never invokes a current-state fallback. Economic Calendar context is requested with `as_of`, so unpublished actuals and later revisions are excluded by its provider-neutral point-in-time repository.

Replay uses the same policy with explicit `replay` mode, deterministic IDs/fingerprints, an injected clock, and event suppression by default. Historical policy bodies are not stored separately in 1.0; snapshots preserve policy/configuration versions and a configuration hash.

## Explainability and lifecycle

Every immutable snapshot contains reconciled components, stable positive/negative/risk contributors, structured conflict codes, missing/degraded sources, freshness states, policy metadata, and safety flags. No generated prose or imperative trade language is used.

The on-demand lifecycle is collect → normalize → score → validate → persist → publish bounded features/events. Equivalent input fingerprints are idempotent. PostgreSQL uniqueness on `(input_fingerprint, mode)` provides multi-instance duplicate protection. The in-memory adapter is for tests/degraded startup only.

## Persistence and retention

`ai_score_snapshots`, `ai_score_components`, and `ai_score_conflicts` use immutable JSON payloads plus indexed query columns and cascading foreign keys. Live and replay retention windows are configurable; cleanup is explicit and bounded and does not run destructively at startup.

## API

- `GET /ai-scoring/health`, `/config`, `/metrics`
- `POST /ai-scoring/score`
- `POST /ai-scoring/replay`
- `GET /ai-scoring/latest`, `/history`
- `GET /ai-scoring/snapshots/{id}` and `/explanation`

History uses bounded limits, offsets, and date ranges. A degraded or insufficient analytical result remains HTTP 200 with an explicit status. Validation errors are sanitized; secrets, raw upstream payloads, paths, and exception traces are not returned.

## Feature Store, Event Bus, metrics, and health

The Feature Store receives only bounded `ai_score` dimensions, version/status/as-of metadata, and `trading_instruction=false`. Events are emitted only after persistence and include stable bounded metadata. Replay events are disabled by default. Publication failures are counted without corrupting a durable score.

Health reports policy, persistence mode, Feature Store/Event Bus state, configured dependencies, and degradation reasons without performing scoring. Metrics use bounded dimensions and do not include arbitrary exception text or snapshot IDs as labels.

## Configuration and operations

`configs/ai_scoring.yaml` defines policy identity, weights, source groups, freshness, label thresholds, confidence ceilings, conflict penalties, API limits, retention, persistence expectations, and replay event behavior. Unknown fields, invalid versions/groups, negative or non-finite weights, freshness inversions, overlapping labels, and incomplete component sets fail validation.

Railway continues to bind Uvicorn to `$PORT`. Optional upstream/provider credentials do not block application startup; absence is visible as degraded health and confidence limitations. PostgreSQL is the durable production store. No local filesystem state is required.

## Known limitations

Production 1.0 uses defensible deterministic aggregation, not calibrated return prediction. Source groups limit obvious double-counting but do not prove statistical independence. Economic Calendar affects direction only when a future version exposes a validated, publication-time-safe surprise adapter; current event context affects risk, confidence, and quality only. Background scheduling/event-triggered recalculation is intentionally not enabled because TEN has no distributed scheduler/outbox abstraction; on-demand scoring is multi-instance safe through database uniqueness.
