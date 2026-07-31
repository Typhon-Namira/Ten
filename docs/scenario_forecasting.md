# Scenario-first market intelligence architecture

The M15 Primary Scenario is TEN's only production signal authority. Trend, SMC,
liquidity, institutional flow, volume, regime, Quant, AI interpretation, and the
M5/M15 synthesis remain immutable analytical inputs. They cannot independently
publish a direction or geometry.

## Authoritative flow

```text
Completed UTC M5 and M15 candles
  -> point-in-time UnifiedMarketState
  -> Quant forecast + AI interpretation
  -> supporting multi-timeframe synthesis
  -> durable M15 simulation attempt (SCHEDULED -> RUNNING)
  -> 5-10 diverse candidate paths
  -> deterministic candidate scoring
  -> Primary + materially distinct Alternative
  -> Primary geometry validation
  -> Signal Validation Engine (risk and publication guardrails only)
  -> Authoritative Scenario Signal
  -> dashboard + email + lifecycle + outcome + calibration
```

The validator may block the Primary but cannot originate or override its
direction. No Primary means no user-facing BUY/SELL and no email.

## M15 timing and synchronization

- Provider candles use an inclusive open and exclusive close interval. A
  `10:00` M15 candle becomes eligible at `10:15:00 UTC`, never at `10:14:59`.
- The authoritative cutoff is the completed M15 close.
- The M5 source is the latest completed M5 cutoff at or before that M15 cutoff;
  processing timestamps do not need to match.
- Future M5 evidence is prohibited.
- M5 updates between closes never hide the last still-authoritative M15 Primary.

## Durable lifecycle and recovery

Every eligible M15 cutoff has one idempotency boundary:

```text
instrument + M15 + market_cutoff + simulation_version
```

The attempt ledger records `SCHEDULED`, `RUNNING`, `SUCCESS`, `NO_SIGNAL`,
`ANALYTICAL_ONLY`, `BLOCKED`, `FAILED`, or `SKIPPED`, including exact
eligibility, synchronization, failure, and candidate details. `PENDING` is a
presentation state only when no attempt exists; it is not persisted as a
terminal result.

The integration worker performs a bounded startup recovery of the latest
persisted M15 UnifiedMarketState. It uses only Quant and synthesis artifacts
belonging to that immutable state, so recovery cannot introduce future data.
The durable integration outbox then continues processing older unpublished
market events in chronological order.

## Dashboard authority

Overview reads the latest authoritative attempt and latest M15 selection by
market cutoff, not by the latest legacy M5 analytical cycle. A newer terminal
`NO_SIGNAL`, `BLOCKED`, `FAILED`, or `SKIPPED` result replaces the older signal.
A currently running next cutoff may retain the previous valid Primary until the
new attempt resolves.

Legacy analysis and decision records remain readable for audit. They are not
rendered on Overview, do not provide fallback direction or geometry, and cannot
enqueue signal email without a Primary Scenario ID.
