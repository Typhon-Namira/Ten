# Forward Market Scenario architecture

The Scenario Forecasting layer is additive. It does not replace or mutate AI
interpretation, Quant forecasting, multi-timeframe signal synthesis, structural
geometry, guardrails, final decisions, lifecycle, or publication.

## Data flow

```text
Completed M5/M15 candle
  -> point-in-time UnifiedMarketState
  -> QuantForecastResult + validated AI interpretation
  -> existing MultiTimeframeSignalSet
  -> ScenarioForecastingEngine
       -> M5 ForwardMarketScenario after an M5 close
       -> M15 ForwardMarketScenario after an M15 close
       -> CombinedForwardScenario after an M15 close
  -> immutable scenario persistence
  -> read-only dashboard projection

Expired scenario + post-cutoff candles through expiry
  -> ScenarioOutcome
  -> completed-only calibration history
  -> reliability annotation for later scenarios
```

## Ownership boundaries

- Existing components remain the owners of evidence extraction, scoring,
  direction, structural geometry, guardrails, final action, lifecycle and
  publication.
- Scenario Forecasting consumes their immutable outputs and owns only
  forward-path hypotheses, scenario geometry, scenario history, outcome
  evaluation and calibration.
- Scenario geometry never overwrites structural geometry and is never fed into
  publication implicitly.
- The dashboard endpoint only reads persisted scenario records and never invokes
  the engine or an AI provider.

## Point-in-time rules

- A scenario requires a fresh frame whose source close equals its expected close.
- All inputs must share one market-state and cycle lineage.
- The next-candle Quant horizon (`candle_count == 1`) is selected explicitly.
- M5 comparison input must have a cutoff at or before the M15 cutoff and remain
  unexpired.
- Outcome evaluation accepts only candles whose timestamp is at or after the
  scenario cutoff and whose completed close is at or before expiry.
- Calibration accepts only outcomes completed at or after scenario expiry.

## Deterministic geometry

Direction and explanation may consume the existing AI interpretation indirectly
through the validated multi-timeframe synthesis. All prices are deterministic:

- expected movement and range come from the Quant next-candle horizon;
- entry is current price or an already-validated, reachable structural entry;
- reachability is bounded by expected movement and a percentage ceiling;
- a structural fact identifier is mandatory;
- stop distance and target distance are bounded fractions of expected movement;
- ordering, target traversal, invalidation traversal, minimum risk/reward and
  expected-move bounds are validated before geometry is persisted.

A valid analytical scenario may therefore persist with unavailable execution
geometry and an exact rejection reason.
