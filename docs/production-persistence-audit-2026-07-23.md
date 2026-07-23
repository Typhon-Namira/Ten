# TEN production persistence audit

Date: 2026-07-23

Scope: SQLAlchemy models, all Alembic revisions through `20260723_0006`, repository
implementations, integration/bootstrap workers, dashboard reads, and in-memory adapters.
The schema was treated as authoritative. This audit changes persistence and operational
reliability only; it does not change analytical formulas, thresholds, gates, prompts, or
publication decisions.

## Root causes and repairs

1. Five analytical snapshot repositories used UUID-only conflict handling while their schema
   enforced a composite analytical boundary. SMC, liquidity, volume profile, institutional flow,
   and market regime now use their exact schema boundary.
2. Economic-calendar revisions, replay outputs, quant feature vectors/results, AI forecasts,
   managed signals/outcomes, guardrail evaluations, and market-state evidence links had the same
   mismatch in different forms. Each now targets its actual unique boundary.
3. Several parent inserts could lose an idempotent race, after which child/checkpoint writes still
   used the incoming UUID. These paths now resolve and propagate the canonical persisted ID.
4. In-memory repositories deduplicated by UUID while PostgreSQL deduplicated by logical identity.
   Shared analytical-boundary code now keeps both adapters aligned.
5. The integration outbox had no durable claim. Concurrent Railway replicas could perform the
   same expensive pipeline work even though final inserts were idempotent. Revision `0006` adds
   leases, `FOR UPDATE SKIP LOCKED` claiming, crash recovery, and bounded retry delay.
6. A dashboard performance read called the mutating outbox `pending()` method. It now uses a
   read-only `oldest_pending()` query and cannot steal work.
7. Field-level dashboard dependency reads had no application timeout. `safe_call()` now has a
   five-second bound; PostgreSQL retains statement, pool, connect, and idle-transaction timeouts.
8. Startup/bootstrap logs did not expose enough deployment identity or retry state. Startup,
   shutdown, bootstrap-unit, retry, transition-to-live, and pipeline failure logs now carry
   structured operational context.

## Persistence inventory

Semantics abbreviations: `I` immutable insert/ignore, `U` deterministic operational upsert,
`C` append-only child/leaf, `L` leased work item. Repository methods own their scoped session;
the method that calls `commit()` is the transaction owner. Exceptions roll back and are re-raised.

| Tables/models | Logical identity / final conflict behavior | Semantics |
|---|---|---|
| `historical_candles` | `(symbol,timeframe,timestamp)`; update the same market bar deterministically | U |
| `realtime_candles` | primary `id`; append provider observation | C |
| `market_data`, `market_cache_metadata` | primary key; current operational state | U |
| `provider_metrics`, `market_quality_history`, `market_gap_history`, `market_latency_history`, `market_synchronization_history`, `market_memory_entries` | declared primary IDs; append-only diagnostics | C |
| `smc_analysis_snapshots` | `(symbol,timeframe,analysis_timestamp,configuration_version,processing_mode)` | I |
| `liquidity_snapshots` | `(symbol,timeframe,analysis_timestamp,configuration_version,processing_mode)` | I |
| `volume_profile_snapshots` | `(symbol,timeframe,analysis_timestamp,configuration_version,processing_mode)` | I |
| `institutional_flow_snapshots` | `(symbol,timeframe,analysis_timestamp,configuration_version,processing_mode)` | I |
| `market_regime_snapshots` | `(symbol,timeframe,analysis_timestamp,configuration_version)` | I |
| `smc_checkpoints`, `liquidity_checkpoints`, `volume_profile_checkpoints`, `institutional_flow_checkpoints`, `market_regime_checkpoints` | `(symbol,timeframe,configuration_version)`; point to canonical snapshot ID | U |
| `smc_objects`, `liquidity_objects`, `volume_profile_objects`, `institutional_flow_evidence`, `market_regime_evidence`, `market_regime_transitions` | deterministic primary ID; no competing unique boundary | C |
| `economic_calendar_provider_observations`, `economic_calendar_events` | deterministic primary ID | I |
| `economic_calendar_event_revisions` | `(event_id,revision_number)` | I |
| `economic_calendar_snapshots`, `economic_calendar_instrument_contexts` | deterministic primary ID; schema defines no competing boundary | I |
| `economic_calendar_sync_state` | `provider_name` | U |
| `economic_calendar_checkpoints` | `(engine_name,configuration_version)` | U |
| `ai_score_snapshots` | `(input_fingerprint,mode)`; canonical snapshot returned to child writers | I |
| `ai_score_components`, `ai_score_conflicts` | deterministic primary ID and parent FK | C |
| `signal_decisions` | `(input_fingerprint,mode)`; canonical decision ID returned | I |
| `signal_decision_rules`, `signal_decision_reasons` | deterministic primary ID and parent FK | C |
| `integration_events` | `event_id` | I |
| `integration_outbox` | unique `event_id`; atomically claimed by lease, completed once | L |
| `integration_processed_events` | `event_id`; `INSERT … SELECT` guarantees visible parent | I |
| `integration_snapshots`, `operational_signals` | `semantic_hash` | I |
| `integration_event_trace`, `integration_data_quality_issues` | deterministic primary ID | C |
| `replay_sessions`, `replay_transitions` | deterministic primary ID | I |
| `replay_checkpoints`, `replay_event_trace` | `(replay_id,sequence)` | I |
| `replay_outputs` | `(replay_id,fingerprint)` | I |
| `market_evidence_frames` | `frame_hash` | I |
| `unified_market_states` | `state_hash` | I |
| `unified_market_state_timeframes` | `(state_id,timeframe)` | I |
| `evidence_items` | `evidence_id` | I |
| `unified_market_state_evidence_links` | `(state_id,ordinal)`; PK `(state_id,evidence_id)` remains enforced | I |
| `quant_forecast_model_metadata` | `(model_name,model_version)` | U |
| `quant_forecast_requests` | `request_id` | I |
| `quant_feature_vectors` | unique `market_state_id`; canonical vector ID propagated | I |
| `quant_feature_references` | `(vector_id,feature_name,evidence_id)` | I |
| `quantitative_forecasts` | unique `request_id`; canonical result ID propagated | I |
| `quantitative_forecast_horizons` | `(result_id,horizon_id)` | I |
| `quant_forecast_outcomes` | `(result_id,horizon_id)` | U |
| `quant_calibration_runs` | `report_id` | I |
| `quant_calibration_buckets` | `(report_id,ordinal)` | I |
| `ai_setup_family_versions` | `(setup_family_id,version)` | I |
| `ai_reasoning_requests` | `request_id` | I |
| `llm_structured_output_failures` | `failure_id` | I |
| `ai_market_forecasts` | unique `request_id`; canonical forecast ID propagated | I |
| `ai_forecast_scenarios` | `(forecast_id,ordinal)` | I |
| `ai_forecast_evidence_links` | `(forecast_id,evidence_id,role)` | I |
| `ai_signal_proposals` | `proposal_id` | I |
| `managed_signals` | unique `structural_opportunity_key`; existing signal is authoritative | U |
| `signal_state_transitions` | `transition_id` | I |
| `signal_monitoring_evaluations` | `evaluation_id` | I |
| `signal_level_revisions` | `revision_id` | I |
| `managed_signal_outcomes` | unique `signal_id` | I |
| `hard_gate_versions` | `(gate_id,gate_version)` | I |
| `final_system_actions` | `final_action_id`; same ID may update deterministic operational state | U |
| `guardrail_evaluations` | `(final_action_id,gate_id)` | I |
| `published_analytical_signals` | unique `signal_id` and unique `final_action_id` | I |
| `llm_usage_metrics` | `metric_id` | I |
| `detailed_signal_outcomes` | unique `signal_id` | U |
| `ai_performance_reports`, `ai_production_readiness_reports` | `report_id` | I |
| `signals`, `analysis_results`, `engine_logs` | legacy primary IDs; no repository conflict target competes with another unique boundary | C |

Foreign-key ordering was verified for integration parent/processed markers and for the complete
Unified Market State → Quant → AI → Final Action → Publication chain. Canonical-ID resolution is
required before dependent child inserts. No repository catches and silently ignores
`IntegrityError`.

## Static conflict review

No repository retains `on_conflict_do_nothing(index_elements=["id"])` on a table with a competing
unique constraint. Remaining primary-ID conflict targets are deterministic leaf records:
analytical objects/evidence, score components/conflicts, decision rules/reasons, calendar
observations/snapshots/contexts, regime transitions, replay sessions/transitions, trace/quality
records, and immutable report/failure records. The schema-consistency regression test validates
all static targets and explicitly guards the high-risk logical-boundary contracts.

## Transaction, retry, and bootstrap behavior

- Every repository call receives a fresh scoped `AsyncSession`; rollback/close prevents a failed
  transaction from poisoning later calls.
- Parent events are committed before historical analysis; processed markers use an atomic
  parent-select insert and are written only after the pipeline's completion/blocking decision.
- Outbox claims survive multiple replicas, expire after 15 minutes if a process dies, and use
  bounded exponential failure delay.
- Bootstrap isolates each symbol/timeframe, retries each unit at most three times with bounded
  backoff, reports failures, completes the remaining units, and then explicitly enters live
  polling.
- Dashboard reads do not claim queue items and field-level dependency calls cannot wait forever.

## Validation boundary

`TEN_TEST_DATABASE_URL` is intentionally mandatory for real PostgreSQL concurrency tests and must
never equal `TEN_DATABASE_URL`. Without that variable, those tests skip and PostgreSQL/Railway
production verification remains unresolved; local unit or mocked-SQL success is not represented as
production proof.
