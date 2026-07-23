# Dashboard production data-path audit

## Primary root cause

The production worker was successfully completing the legacy
`market data -> engine snapshots -> AI score -> signal decision` pipeline. The redesigned
dashboard, however, reads the Phase 1-6 AI-centric records. Those records were never generated
because `configs/feature_flags.yaml` disables `ai_centric_shadow_mode`,
`ai_signal_proposals`, and `ai_signal_monitoring`, and production had no supported runtime
override for those flags.

This produced two independent symptoms:

1. The old frontend client polled Quant, Calibration, and AI Reasoning independently. Expected
   first-record 404 responses appeared as recurring browser failures and unrelated "latest"
   queries could combine different analysis boundaries.
2. The UI collapsed route errors, no-record responses, disabled stages, and pending stages into
   the same generic empty copy.

The fix adds explicit environment overrides whose default is `None`, so repository defaults and
existing deployments remain unchanged. Production can enable shadow generation, proposals, and
monitoring while leaving publication and adjustments disabled. It also adds one read-only
aggregate that joins persisted rows by `UnifiedMarketState.state_id`; no analysis is executed by
the GET request.

## Authoritative data path

| Dashboard data | Source table(s) | Repository read | API | Frontend consumer |
| --- | --- | --- | --- | --- |
| Market state and engine evidence | `unified_market_states`, `unified_market_state_timeframes`, `evidence_items`, `unified_market_state_evidence_links` | `UnifiedMarketStateRepository.latest_state` | `GET /api/v1/dashboard/latest` | `useAIDashboardData` -> all five AI dashboard views |
| Quant forecast | `quantitative_forecasts` | `QuantForecastRepository.result_for_state` | same aggregate | Quant Forecast and Calibration |
| AI reasoning | `ai_market_forecasts` | `AIReasoningRepository.forecast_for_state` | same aggregate | Overview and Signals |
| AI proposal | `ai_signal_proposals` | `AIReasoningRepository.proposal_for_state` | same aggregate | Overview and Signals |
| Guardrails/final action | `final_system_actions` | `FinalDecisionRepository.action_for_state` | same aggregate | Overview and Signals |
| Publication | `published_analytical_signals` | `FinalDecisionRepository.publication_for_signal` | same aggregate | Overview and Signals |
| Monitoring | `managed_signals` and lifecycle history tables | `AIReasoningRepository.active_signals` / `signal_history` | same aggregate | Signals and Overview |
| Outcome/performance/readiness | `detailed_signal_outcomes`, `ai_performance_reports`, `ai_production_readiness_reports` | same-signal outcome and latest validated reports | same aggregate | Performance, Calibration, and System |
| Runtime health | in-process service health plus feature-flag snapshot | service health methods | same aggregate | System |

Every stage envelope contains `status`, `reason`, `record_id`, `timestamp`, `error_code`,
`retryable`, and `data`. Missing data therefore remains distinguishable from a missing route,
database failure, disabled feature, pending generation, or insufficient validated sample.

## Frontend dependency map

`Overview`, `Signals`, `Performance`, `Calibration`, and `System` are views of the same
`AIDashboard` component and `useAIDashboardData` hook. The hook requests:

- `/api/v1/system/market-intelligence?instrument=...&timeframe=...`
- `/api/v1/dashboard/latest?instrument=...`

It maintains one in-flight request, polls every 15 seconds while visible and every 60 seconds
while hidden, uses bounded exponential backoff up to 120 seconds, and retains the last successful
response during temporary failures. The aggregate replaces the former independent Quant,
Calibration, and AI Reasoning polling calls.

## API contract

`GET /api/v1/dashboard/latest` accepts the canonical `instrument` parameter. The shared backend
normalizer accepts case differences, surrounding whitespace, `XAU/USD`, and `XAU-USD`, returning
`XAUUSD`.

The endpoint always returns HTTP 200 for normal data states:

- `complete`: market state, quant, and AI reasoning are persisted for the same state ID.
- `partial`: at least one of those stages is pending or unavailable.
- `pending`: no synchronized market state exists yet.
- `failed`: a persisted authoritative stage has a failed status.

The existing individual endpoints remain registered for backward compatibility.

## Provider payload boundary

The first deployed synchronized cycle exposed an additional concrete failure: the complete
market state was approximately 22 MB and the old request builder repeated full historical
zone/level collections across several prompt fields. The provider request exceeded practical
model/request limits and persisted `llm_unavailable`.

Unified Market State still persists the complete, unmodified engine payload. At the external AI
boundary only, long historical collections are now represented by their exact total count,
deterministic first/latest samples, and the enclosing evidence ID (which commits to the complete
raw evidence). Every top-level engine field and scalar summary remains present, while derived
prompt categories carry evidence references instead of duplicating the same raw payload. A real
production state that previously generated an unbounded request now produces a roughly 320 KB
request.

## Production configuration

The intended shadow rollout is:

```text
TEN_AI_CENTRIC_SHADOW_MODE=true
TEN_AI_SIGNAL_PROPOSALS=true
TEN_AI_SIGNAL_MONITORING=true
TEN_AI_SIGNAL_PUBLICATION=false
TEN_AI_SIGNAL_ADJUSTMENTS=false
```

Publication and adjustment remain disabled, so the repair cannot broaden publication
eligibility or alter an existing signal. No analytical formula, confidence threshold, strategy,
risk rule, hard gate, position sizing rule, or execution behavior is changed.

## Database and migrations

No schema change or migration is required. All required Phase 1-6 tables and relationships
already exist. The repair adds same-cycle repository queries over existing indexed foreign-key
columns.

## Connection reliability

The dashboard now performs one aggregate poll rather than three independently scheduled
AI-centric polls. Repository operations continue to use short scoped sessions and the GET route
does not call providers or analytical services. This reduces connection acquisition pressure and
prevents browser polling from triggering computation.
