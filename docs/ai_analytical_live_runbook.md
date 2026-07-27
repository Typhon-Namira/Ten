# TEN AI-Centric Analytical Operation Runbook

## Scope and safety boundary

TEN publishes analytical XAUUSD signals only. It does not submit orders, calculate a real
position size, or read authoritative balance, equity, exposure, daily-loss, or drawdown data.
Account-risk gates report `not_applicable` with
`authoritative_account_risk_unavailable`; they never invent account values.

No profitability is guaranteed. Production readiness is based only on persisted measured
outcomes, calibration, failure rates, latency, request usage, and sample size.

## Final architecture

```text
Closed market candle
  -> UnifiedMarketState (M1/M5/M15, point-in-time)
  -> deterministic quantitative forecast
  -> ordered four-account Groq provider pool
  -> immutable AI forecast and immutable AI proposal
  -> versioned deterministic HardGateRegistry
  -> persisted FinalSystemAction and every GateEvaluation
  -> feature-gated analytical publication
  -> persistent lifecycle monitoring and audited revisions
  -> horizon-complete outcome evaluation
  -> calibration, comparison, and readiness reports
```

The legacy production scoring, thresholds, hard gates, and publication path remain independent.
Failure in the AI-centric path is isolated inside the existing shadow-pipeline boundary.

## Operating profiles

The reference profiles live in `configs/ai_operating_profiles.yaml`:

- `safe_test`: state, forecasts, proposals, and monitoring; no publication or adjustments.
- `shadow`: continuous closed-cycle evaluation; no publication or adjustments.
- `analytical_live`: analytical publication and policy-approved adjustments; broker execution
  remains false.

Runtime feature flags remain authoritative. The profile file is a deployment reference and does
not silently change `configs/feature_flags.yaml`.

## Activation procedure

1. Apply migration `20260723_0005` and verify `alembic current` reports the head.
2. Start with `safe_test`; confirm M1/M5/M15 synchronization and no future-data violations.
3. Enable `ai_centric_shadow_mode`, `ai_signal_proposals`, and `ai_signal_monitoring`.
4. Observe structured-output reliability, duplicate prevention, LLM usage, persistence, latency,
   complete outcomes, and calibration for the configured minimum sample.
5. Generate a `ProductionReadinessReport`. Do not enable publication when its status is
   `not_ready`.
6. Enable `ai_signal_publication` only for analytical signals after all required checks pass.
7. Enable `ai_signal_adjustments` only after protective-stop and lifecycle revision tests pass.
8. Confirm the dashboard says `NO BROKER EXECUTION`.

## Required publication inputs

Publication requires a synchronized, fresh market state; available quantitative result; validated
AI forecast and proposal; mandatory setup-specific evidence; open market; known acceptable spread;
valid price geometry and precision; absolute safety Risk-to-Reward; valid lifecycle and persistence;
no duplicate structural opportunity; no prohibited economic-event window; and an available
publication repository.

HTF disagreement, weak volume, ranging/compression regimes, ordinary evidence conflicts, and
missing Order Block/FVG evidence for unrelated setup families are not general hard gates.

## LLM failure and cost controls

- Groq accounts are attempted in ordered failover from `groq_1` through `groq_4`.
- One logical reasoning result is allowed per immutable UMS cycle boundary and provider contract.
- Calls occur on eligible closed analysis cycles, never per tick.
- Context and market memory are bounded.
- Calls use a strict timeout, concurrency limit, bounded retries, and exponential provider backoff.
- Request hashes, prompt/model versions, generation parameters, latency, failures, and tokens
  (when the provider exposes them) are persisted daily.
- Recorded-response and deterministic-baseline replay never contact the live LLM.

## Monitoring and adjustments

Monitoring runs on legally closed cycles. Material actions require deterministic approval.
Supported policy actions include pre-activation entry refinement, cancellation, invalidation,
expiry/risk reduction, partial-profit recommendation, protective Stop movement, remaining-target
invalidation, and closure.

Stop widening is forbidden. Every level change stores the old value, new value, evidence,
policy/model version, reason, and explicit approval rule. Revision and transition history is
append-only.

## Replay and evaluation

Replay reconstructs immutable market states at the replay cursor and rejects evidence whose
availability exceeds that cursor. Manifests pin quantitative-model, prompt, setup-family, policy,
spread, slippage, and LLM replay mode versions.

Outcomes remain `pending` until all required future candles exist. Only then are entry, TP/SL
ordering, MFE, MAE, spread/slippage-adjusted return, realized Risk-to-Reward, expiry, cancellation,
invalidation, and lifetime measured.

Evaluation must reserve a final period that is never used to tune policies.

## Rollback

1. Set `ai_signal_adjustments`, `ai_signal_publication`, `ai_signal_monitoring`,
   `ai_signal_proposals`, and `ai_centric_shadow_mode` to `false`.
2. Restart the service and verify the AI health endpoint reports the flags disabled.
3. Preserve all proposals, final actions, publications, revisions, outcomes, and usage records for
   audit; do not delete history during operational rollback.
4. The legacy pipeline continues independently.
5. Only if schema rollback is explicitly required and its audit data has been exported, run
   `alembic downgrade 20260723_0004`. This removes Phase 5/6 tables but does not alter legacy
   signal tables.

## Known limitations and unresolved risks

- No broker execution or authoritative account-risk integration exists.
- Publication is blocked when market status, spread, required economic context, persistence, or
  publication health is unavailable.
- LLM token counts can be unavailable when the configured provider omits usage telemetry.
- Raw LLM confidence is labeled uncalibrated until sufficient outcome observations demonstrate
  calibration.
- A readiness report with insufficient samples, missing performance measurements, excessive LLM
  failure, or excessive publication failure is `not_ready`.
