# TEN Architecture V2 — Mandatory Design Audit

Audit target: `docs/ten-architecture-v2-proposal.md`

Audited revision: 2

Implementation status: **not authorized**

## A. Executive verdict

**REQUIRES ARCHITECTURE REVISION**

Revision 2 corrects the violations found in revision 1, but implementation must remain blocked until the unresolved human decisions in section K are explicitly approved. The verdict is deliberately not “APPROVED FOR IMPLEMENTATION”: infrastructure selection, retention authority, closed-candle correction policy, degraded-analysis policy, and the production storage ceiling are product/operational decisions rather than safe implementation assumptions.

## Architecture violations found

Revision 1 contained these material violations or ambiguities:

1. The Market Data Service was not restricted to M1 and the diagrams implied it could persist generic timeframes.
2. Liquidity and Institutional Flow were absent from the analytical sequence.
3. AI retry language allowed ambiguity between one logical request and multiple physical provider calls.
4. AI failure could leave the cycle outcome dependent on an unspecified “if policy permits” branch.
5. Separate `ai_forecasts` and `final_decisions` rows encouraged append-only duplicate WAIT history.
6. Routine completed-cycle rows and meaningful lifecycle history were not clearly separated.
7. `analysis_events.compact_details` and `outbox_events.compact_payload` left generic JSONB payload escape hatches.
8. `risk_state_current` duplicated state that belongs in the single current-state row.
9. The candle design did not fully specify gap repair or late closed-candle corrections.
10. Storage had a qualitative “few MB/day” target but no class budgets, formula, or failing ceiling.
11. The dashboard contract omitted several mandatory stage and scheduling fields.
12. WAIT terminal semantics were not defined stage by stage.
13. The write path and table-reader ownership were not fully inventoried.
14. Runtime-state reconstruction and Redis-unavailable behavior needed stronger authority rules.
15. V1 writer shutdown had no named kill switch or generation-fencing requirement.

All fifteen were revised in the proposal. Human approval is still required before the remaining product choices can become implementation requirements.

## Exact revisions made

- Restricted live market persistence to canonical M1 only.
- Added Liquidity and Institutional Flow to the in-memory analytical chain.
- Defined deterministic M5/M15 point-in-time aggregation and prohibited their storage.
- Set the live AI policy to one physical HTTP request and zero automatic provider retries.
- Made AI terminal failure complete the cycle with deterministic WAIT rather than strand it.
- Folded compact Quant, AI, and final results into the bounded cycle ledger and singleton current state.
- Replaced generic analytical history with typed, meaningful `lifecycle_events`.
- Removed the proposed outbox, separate AI forecast history, duplicate final-decision history, and separate risk-current table.
- Removed generic full-payload JSONB columns; only a bounded numeric target array remains.
- Added explicit gap detection/repair and late-correction behavior.
- Added exact writer/read inventories, estimated row sizes, daily ceilings, and retention.
- Added a 1.96 MiB/day projected ceiling and 5 MiB/day failing acceptance gate.
- Added complete WAIT stage semantics and a backend-authoritative dashboard contract.
- Added exact cycle sequence, timeouts, retry rules, persistent writes, and failure matrix.
- Added `TEN_V1_WRITERS_ENABLED` and database generation fencing to cutover.

## B. Ten-decision review

### Decision 1 — Redis Streams and runtime hashes

- **Decision:** Use Redis Streams for delivery and Redis hashes with TTL for runtime state.
- **Rationale:** Independent services need visible, acknowledged, ephemeral coordination without turning PostgreSQL into a runtime event log.
- **Benefit:** Clear separation between durable outcomes and in-flight state; consumer ownership and lag are observable.
- **Risk:** Redis outage or split-brain event ordering.
- **Operational consequence:** Redis needs health checks, persistence appropriate to queue recovery, consumer-group monitoring, and monotonic fencing.
- **Storage consequence:** Runtime events add zero normal PostgreSQL growth.
- **Compatibility consequence:** Existing database-derived dashboard status is replaced by a V2 contract; temporary adapters may translate responses.
- **Requirement fit:** Satisfies runtime/durable separation if Redis never owns completed results.
- **Required correction:** Revision 2 defines reconstruction and Redis-unavailable behavior. Human infrastructure approval remains required.

### Decision 2 — UTC is the authoritative clock

- **Decision:** Schedule from absolute UTC five-minute boundaries.
- **Rationale:** It is deterministic across restarts, deployments, and time zones.
- **Benefit:** Reproducible cycle identity and synchronized point-in-time data.
- **Risk:** Host clock drift.
- **Operational consequence:** NTP/clock-drift monitoring is mandatory.
- **Storage consequence:** None.
- **Compatibility consequence:** Startup-relative V1 cadence is intentionally not preserved.
- **Requirement fit:** Fully satisfies the cadence requirement.
- **Required correction:** None.

### Decision 3 — 90-second misfire window; no automatic backlog

- **Decision:** Recover only the current boundary within 90 seconds; skip older live boundaries.
- **Rationale:** Prevent restart storms, old-data AI calls, and delayed “live” signals.
- **Benefit:** Bounded workload and cost.
- **Risk:** A prolonged outage produces missing live cycles.
- **Operational consequence:** Missed cycles are visible as `scheduler_misfire`; replay is operator-controlled and AI-disabled.
- **Storage consequence:** At most one compact skipped cycle row per missed boundary if the scheduler records it; no payload history.
- **Compatibility consequence:** Legacy implicit catch-up is not preserved.
- **Requirement fit:** Satisfies exact live cadence and bounded retries.
- **Required correction:** Human approval of 90 seconds.

### Decision 4 — One global current-state row

- **Decision:** `current_state` contains exactly one row for the current XAUUSD-only product.
- **Rationale:** The product currently has one authoritative symbol; bounded cardinality is explicit.
- **Benefit:** Constant-size current state and simple dashboard recovery.
- **Risk:** It cannot safely add a second symbol without schema/API revision.
- **Operational consequence:** Multi-symbol work requires a reviewed change rather than silently increasing row cardinality.
- **Storage consequence:** Zero row growth; UPSERT only.
- **Compatibility consequence:** Legacy snapshot history is not reconstructed.
- **Requirement fit:** Fully satisfies bounded current state.
- **Required correction:** Human confirmation that V2 launch remains single-symbol.

### Decision 5 — M5/M15 computed, never stored

- **Decision:** Persist canonical M1 only; derive closed UTC-aligned M5/M15 bars point-in-time.
- **Rationale:** Higher timeframes duplicate M1 information and can leak future values if aggregated incorrectly.
- **Benefit:** Single market-data truth, less storage, deterministic replay.
- **Risk:** Slight compute cost and blocked aggregation when M1 has a gap.
- **Operational consequence:** Aggregation requires complete closed members; gaps are exact typed blockers.
- **Storage consequence:** Avoids 384 higher-timeframe rows/day and all duplicate snapshot copies.
- **Compatibility consequence:** Provider-native higher-timeframe behavior is intentionally removed unless separately justified.
- **Requirement fit:** Fully satisfies candle deduplication.
- **Required correction:** None; human approval confirms no provider-native exception.

### Decision 6 — Retention schedule

- **Decision:** M1 hot for 24 months; cycles 90 days; lifecycle events one year unless audit-linked; trades seven years; runtime 24 hours.
- **Rationale:** Preserve reproducibility and financial audit while bounding operational data.
- **Benefit:** Predictable growth and explicit cleanup authority.
- **Risk:** Legal/audit retention may differ by jurisdiction.
- **Operational consequence:** Cleanup is bounded and cannot delete restricted audit rows.
- **Storage consequence:** At 1.96 MiB/day, normal one-symbol V2 data is approximately 0.70 GiB/year before archival.
- **Compatibility consequence:** Legacy indefinite analytical payload retention is not preserved.
- **Requirement fit:** Satisfies exact retention requirement.
- **Required correction:** Compliance/operations approval.

### Decision 7 — Degraded-stage continuation rules

- **Decision:** Missing optional evidence may continue only through typed contracts; missing mandatory market/Quant input blocks AI. AI failure completes the cycle as deterministic WAIT.
- **Rationale:** Never fabricate evidence and never leave cycles permanently incomplete.
- **Benefit:** Accurate degradation and complete terminal semantics.
- **Risk:** Incorrect optional/mandatory classification could weaken analysis.
- **Operational consequence:** Each engine contract needs an approved dependency classification.
- **Storage consequence:** Only reason codes and compact result are stored.
- **Compatibility consequence:** V1 generic failure/fallback behavior is not preserved.
- **Requirement fit:** Directionally satisfies requirements, but the exact engine dependency map needs product approval.
- **Required correction:** Approve the engine dependency matrix before implementation.

### Decision 8 — Restricted AI incident diagnostics

- **Decision:** No prompt/response persistence normally; optionally permit encrypted external diagnostics for seven days under an incident flag.
- **Rationale:** Full prompts/responses are sensitive and large but can help bounded incident diagnosis.
- **Benefit:** Normal storage stays compact; exceptional investigation remains possible.
- **Risk:** Privacy/security exposure and accidental prolonged retention.
- **Operational consequence:** Requires access audit, encryption, redaction, explicit activation, and automatic expiry.
- **Storage consequence:** Zero PostgreSQL growth; external archive capped at 256 KiB/day while enabled.
- **Compatibility consequence:** Existing raw response persistence is removed.
- **Requirement fit:** Satisfies the bounded diagnostic exception.
- **Required correction:** Security approval or remove the feature entirely.

### Decision 9 — 5 MiB/day hard ceiling

- **Decision:** Expected ceiling 1.96 MiB/day; implementation fails at a projected value over 5 MiB/day per active symbol.
- **Rationale:** Allow measured serialization variance while making gigabyte-scale growth impossible.
- **Benefit:** Testable storage behavior.
- **Risk:** Index/WAL behavior in production may exceed fixture estimates even if heap does not.
- **Operational consequence:** Release tests measure heap/index bytes per cycle; production monitors heap, indexes, TOAST, and WAL separately.
- **Storage consequence:** Approximately 1.78 GiB/year at the hard limit, 0.70 GiB/year at projected normal growth.
- **Compatibility consequence:** Full-payload compatibility writes cannot pass.
- **Requirement fit:** Fully satisfies measurable budget requirement.
- **Required correction:** Human approval of the ceiling and whether WAL gets a separate gate.

### Decision 10 — Shadow parity and cutover

- **Decision:** Time-boxed V2 shadowing, then dashboard cutover, generation-fenced V1 writer shutdown, and V2 authority.
- **Rationale:** Compare outcomes without running two authoritative systems indefinitely.
- **Benefit:** Safe observation and immediate rollback.
- **Risk:** Shadow V1 full-payload writes temporarily continue consuming space.
- **Operational consequence:** A cutover deadline and parity criteria are mandatory; `TEN_V1_WRITERS_ENABLED=false` is the kill switch.
- **Storage consequence:** Temporary duplication only during the approved window; no indefinite dual writers.
- **Compatibility consequence:** Compatibility is read-adapter-only after V1 writers stop.
- **Requirement fit:** Satisfies phased cutover.
- **Required correction:** Approve shadow duration and parity thresholds.

## C. Complete PostgreSQL write-path inventory

All sizes include expected heap payload but not the separately calculated index multiplier.

| Service / proposed component | Target | Trigger | Maximum frequency | Operation | Avg payload | Max new rows/day | Retention | Why persistence is required |
|---|---|---|---:|---|---:|---:|---|---|
| Market Data Service / `CandleWriter.apply_latest_m1` | `market_candles` | Each one-minute poll | 1/min/symbol | UPDATE open or INSERT closed | 240 B | 1,440 | 24 months hot | Canonical market truth for future cycles, chart, replay, gap detection |
| Gap Repair Command / `repair_missing_m1` | `market_candles` | Explicit operator job | ≤500/run | INSERT missing only | 240 B | 0 normal; job-capped | 24 months hot | Restore a proven gap without mutating closed rows |
| Scheduler / `CycleRepository.create_if_absent` | `analysis_cycles` | UTC five-minute boundary | 288/day/symbol | INSERT | 1,600 B terminal | 288 | 90 days | Unique cycle identity and enqueue idempotency |
| Analysis Engine / `CycleRepository.claim` | `analysis_cycles` | Queue delivery | ≤1 accepted/cycle | UPDATE | No row growth | 0 | 90 days | Ownership and fencing |
| Analysis Engine / `CycleRepository.reserve_ai` | `analysis_cycles` | Before AI | ≤1/cycle | UPDATE | No row growth | 0 | 90 days | Enforce and audit at-most-one AI request |
| Analysis Engine / `CycleRepository.finalize` | `analysis_cycles` | Terminal cycle | ≤1/cycle | UPDATE | No row growth | 0 | 90 days | Exact compact terminal result and AI metadata |
| Analysis Engine / `CurrentStateRepository.upsert` | `current_state` | Terminal cycle | ≤288/day | UPSERT singleton | 2,048 B | 0 after init | Indefinite | Restart recovery and authoritative latest result |
| Analysis Engine / `LifecycleRepository.append_if_meaningful` | `lifecycle_events` | Approved material change/failure | ≤100/day budget | INSERT | 800 B | 100 | 1 year/7 years linked | Audit of meaningful state/action change |
| Execution Service / `TradeRepository.open_or_update` | `trades` | Actionable approved lifecycle | Product cap 50/day | INSERT/UPDATE | 480 B | 50 | ≥7 years | Current execution and financial audit |
| Execution/Monitoring Service / `TradeEventRepository.append` | `trade_events` | Named trade transition | Product cap 250/day | INSERT | 288 B | 250 | ≥7 years | Immutable trade lifecycle audit |
| Performance Worker / `PerformanceRepository.upsert_daily` | `performance_daily` | Daily close or trade close | ≤24 updates/day; 1 row/day | UPSERT | 400 B | 1 | 7 years | Long-range performance reporting |
| Retention Worker / bounded cleanup | Expirable rows in `analysis_cycles`, `lifecycle_events`, `performance_daily` | Scheduled maintenance | ≤1 batch/min when enabled | DELETE | ≤1,000 rows or 2 s/batch | Negative growth only | Per table | Enforce approved retention without load spikes |

No dashboard endpoint, frontend action, provider-health update, polling heartbeat, runtime-stage update, or unchanged WAIT result writes a PostgreSQL history row. Runtime updates go only to Redis. No unknown or unbounded writer is allowed; the storage acceptance test fails if instrumentation observes an unclassified SQL mutation.

## D. Read-path inventory

| Table | Real readers | Required future use | Replaceable by current state or metrics? |
|---|---|---|---|
| `market_candles` | Analysis Engine, bounded chart API, gap detector, controlled replay/repair validator | Point-in-time M1 source and reproducibility | No; current state cannot provide history |
| `analysis_cycles` | Scheduler dedupe, worker claim/recovery, AI reservation, Dashboard Service recent-cycle view, incident audit, retention worker | Idempotency, in-flight recovery, recent exact AI/cycle outcome | No; metrics cannot enforce uniqueness or recover ownership |
| `current_state` | Dashboard Service, Analysis Engine prior-fingerprint comparison, execution/monitoring startup | Latest durable product state | It is the current-state UPSERT |
| `lifecycle_events` | Dashboard history, incident/audit report, execution reconciliation, retention worker | Meaningful material state/action history | No; current state loses transitions and metrics are not audit records |
| `trades` | Execution and Monitoring Services, Dashboard Service, risk logic, performance worker | Active/closed financial state | No |
| `trade_events` | Monitoring/reconciliation, Dashboard history, performance/incident audit | Immutable trade transitions | No |
| `performance_daily` | Dashboard Service, reporting, risk review, retention worker | Long-period performance without scanning trades | Could be recomputed, but bounded aggregate materially reduces operational reads and has one row/day |

Rejected tables:

- `ai_forecasts`: duplicate per-cycle/current data with no distinct reader need.
- `final_decisions`: duplicate cycle/current data for unchanged WAIT.
- `risk_state_current`: merged into the singleton current state.
- `outbox_events`: no required durable event consumer in the proposed design.
- generic analytical snapshots/evidence frames: prohibited duplicate computation payloads.
- PostgreSQL runtime-stage history: replaced by Redis and metrics/logs.

## E. Exact five-minute cycle sequence

| Step | Input | Output | Timeout | Failure state | Retry policy | Persistent write |
|---|---|---|---:|---|---|---|
| Boundary reached | UTC clock, symbol config | `(symbol,boundary)` | 1 s drift budget | Scheduler `degraded/clock_drift` | Next absolute boundary; no relative sleep | None yet |
| Idempotency insert | Cycle key | New or existing cycle ID | 2 s | Scheduler `failed/database_write_failed` | Capped DB retry before 90-s misfire limit | INSERT cycle only if absent |
| Enqueue | New cycle ID | Stream message | 2 s | Cycle `queued`, scheduler `degraded/enqueue_failed` | Same message/cycle ID; never new row | No new PostgreSQL write |
| Worker claim | Cycle ID, generation | Fencing token | 2 s | Cycle stays queued or becomes terminal timeout | Duplicate deliveries acknowledge; DB retry capped | UPDATE cycle claim |
| Cutoff selection | Boundary, close-settlement policy | `market_cutoff_at` | 5 s | `blocked/stale_or_incomplete_market_data` | One bounded reread before cycle data deadline | UPDATE cutoff only with finalization |
| Load M1 | Symbol, cutoff | Immutable M1 array | 10 s | `failed/market_data_load_failed` | One DB read retry if transient | Read only |
| Aggregate M5/M15 | Complete M1 | Point-in-time bars | 2 s | `blocked/missing_m5_boundary` or typed M15 degradation | No retry; input deterministic | Memory only |
| Analytical engines | Bars/candles, approved contracts | Compact Market/SMC/Liquidity/Volume/Flow results | 30 s total | Stage failed/degraded; dependents blocked per map | No stage rerun in live cycle | Memory only |
| Quant | Compact evidence | Direction/confidence/reason | 5 s | `failed/quant_validation_failed`; downstream blocked | No retry | Memory only |
| Reserve AI | Cycle row | `ai_attempted=true`, start/model | 2 s | AI `failed/ai_reservation_failed`; cycle WAIT | DB retry bounded; no HTTP before success | UPDATE cycle |
| AI provider router | Compact Quant + summaries | One response or typed failure | 20 s | Exact provider/status/reason | One transient retry per provider; ordered fallback | None during HTTP |
| AI normalize | Provider response | Compact AI decision/confidence/reason | 2 s | `degraded` if recoverable; `failed/malformed_ai_response` otherwise | Deterministic local normalization once | Memory only |
| Proposal | Compact AI + Quant | Proposal or `no_proposal/not_required` | 2 s | `failed/proposal_validation_failed`, final WAIT | No retry | Memory only |
| Guardrails | Actionable proposal or none | Approved/rejected/not_required | 2 s | Named rejection or guardrail failure; final WAIT | No retry | Memory only |
| Final action | Compact stage results | BUY/SELL/WAIT and exact reason | 1 s | `failed/finalization_error`; cycle terminal failure | No recomputation | Memory only |
| Transactional finalize | Compact result/current/event predicate | Cycle terminal + singleton current + optional event | 5 s | `failed/database_finalize_failed`; runtime exact circuit reason | Capped DB retry; disk-full opens circuit and stops storm | UPDATE cycle, UPSERT current, optional INSERT event |
| Runtime finalize | Durable commit result | Terminal dashboard stages | 2 s | Runtime `degraded/runtime_registry_unavailable` | Best-effort republish; durable outcome remains authoritative | Redis only |
| Exit/ack | Terminal state | Message acknowledged, process slot released | 2 s | Queue message redelivered and deduped | Redelivery cannot re-execute terminal cycle | None |

The cycle-wide analytical deadline is 75 seconds after worker claim and must finish before the next boundary. A timeout produces a terminal compact failure and deterministic WAIT when sufficient durable state can be written.

## F. Failure-state matrix

| Failure | Authoritative stage state | Final action | Persistent data | Retry | Dashboard message | Terminal? |
|---|---|---|---|---|---|---|
| Provider unavailable | Market `failed/provider_unavailable` | Latest completed decision remains; no new cycle action if data stale | No poll history; runtime only | Next one-minute poll with capped provider backoff | “Market provider unavailable; retry at …” | Poll yes; analysis may block |
| Stale market data | Market `stale/market_data_stale`; analysis `blocked` | No new action | Compact cycle failure if boundary reached | Next poll; no analysis loop | “Analysis blocked: latest closed M1 is stale by N seconds” | Cycle yes |
| Incomplete M1 candle | Market healthy/running; analysis `blocked/incomplete_boundary_candle` | No new action | Compact cycle failure | One bounded settlement reread | “Boundary candle is not closed” | Cycle yes |
| Missing M5 boundary | Aggregation `blocked/missing_m5_boundary` | WAIT/no new action | Compact terminal cycle; event only if operationally material | Gap repair is separate; no live retry | “M5 cannot be built: missing closed M1 at …” | Yes |
| Analysis worker unavailable | Analysis `stale/worker_heartbeat_expired`, queued cycle | No new action | Queued cycle | Queue redelivery within 90-s live window; otherwise terminal misfire | “Analysis worker unavailable; cycle not started” | Eventually |
| Duplicate scheduler execution | Duplicate `skipped/duplicate_cycle` | Existing cycle authoritative | No duplicate row | None | Normally hidden; operations count shows suppression | Yes for duplicate |
| Quant failure | Quant `failed/quant_failure`; AI/proposal blocked | WAIT | Compact terminal cycle; optional terminal-failure event | No live recompute | “Quant failed: exact reason” | Yes |
| Provider 401/403 | AI `failed/authentication_failed` | WAIT | AI attempt metadata + compact cycle/current result | No retry; fallback allowed | “AI authentication rejected” | Yes |
| Provider 402/quota | AI `failed/quota_exhausted` | WAIT | Same | No retry; fallback allowed | “AI quota unavailable” | Yes |
| Provider 429 | AI `failed/rate_limited` | WAIT | Same, including provider retry-after | No retry; fallback allowed | “AI rate limited; retry after …” | Yes |
| Provider timeout | AI `failed/provider_unavailable` | WAIT | Same, elapsed time | One bounded retry, then fallback | “AI provider timed out after N ms” | Yes |
| Malformed AI response | AI `failed/structured_output_invalid` or `degraded/repaired_fields` | WAIT if unrecoverable; otherwise continue | First validation path/reason in compact failure; no raw body | Local deterministic repair once; no new HTTP | Exact invalid field/reason | Yes |
| AI WAIT, no proposal | AI `healthy/completed`; proposal `no_proposal`; guardrails `not_required` | WAIT | Cycle/current only; no event if unchanged | None | “WAIT — no actionable proposal” | Yes, completed |
| Guardrail rejection | Guardrail `healthy/rejected_<name>` | WAIT | Current/cycle + meaningful named rejection event | None | “Proposal rejected by <guardrail>: <reason>” | Yes |
| Publication disabled | Publication `disabled/publication_disabled` | Internal approved action or WAIT; nothing published | Compact result only | None | “Publication disabled by configuration” | Yes |
| PostgreSQL unavailable | Database `failed/database_unavailable`; cycle cannot commit | No authoritative new action | Nothing until commit succeeds | Capped retry within deadline, then queue dedupe/recovery | “Database unavailable; analysis result not committed” | Terminal once failure row can later be reconciled; otherwise queued/quarantined |
| PostgreSQL storage exhausted | Database `failed/storage_exhausted`; circuit open | No new authoritative action | Essential failure update only if possible; otherwise metrics/log | No write storm; probe on circuit schedule | “Storage exhausted; nonessential writes paused” | Cycle quarantined/terminal after recovery |
| Restart during analysis | Runtime `stale` then worker recovery | No duplicate action | Existing claimed cycle/fencing token | Redelivery claims only if lease expired; AI reservation prevents second call | “Cycle recovering after worker restart” | Eventually |

AI/provider failure never leaves Proposal, Guardrails, Publication, Monitoring, or Outcome awaiting work. They become `no_proposal`, `not_required`, and `not_applicable` as appropriate, and the cycle is terminal.

## G. Database schema review

| Table | Purpose | PK / unique | Foreign keys | Indexes | Est. row | Max rows/day | Retention / cleanup | Why not current state or metrics |
|---|---|---|---|---|---:|---:|---|---|
| `market_candles` | Canonical immutable M1 history | PK `(symbol,timeframe,open_time)`; check timeframe M1 | None | `(symbol,open_time DESC)`; partial closed index | 240 B | 1,440 | 24 months hot; archive only after approval | Historical point-in-time input cannot be a singleton or metric |
| `analysis_cycles` | Boundary dedupe, claim recovery, AI reservation, compact recent result | PK `id`; UK `(symbol,analysis_boundary)` | None | unique boundary; `(status,boundary DESC)` | 1,600 B | 288 | 90 days; bounded delete unless referenced | Database uniqueness is required for concurrency and recovery |
| `current_state` | Latest durable result/risk/trade pointer | PK/check `singleton_id=1`; unique latest cycle as appropriate | `cycle_id → analysis_cycles`; `active_trade_id → trades` added after table creation | PK only; cycle lookup | 2,048 B | 0 after init | Indefinite UPSERT | It is the current-state model |
| `lifecycle_events` | Only meaningful material transitions/failures | PK `id`; UK `(cycle_id,event_type,new_fingerprint)` | `cycle_id → analysis_cycles ON DELETE RESTRICT` | cycle; time/type | 800 B | 100 budget | 1 year, 7 years linked; bounded cleanup | Current state loses transitions; metrics lack audit semantics |
| `trades` | Active/closed execution truth | PK `id` | `opening_cycle_id → analysis_cycles RESTRICT` | `(status,opened_at DESC)` | 480 B | 50 | ≥7 years; no automated hard delete | Financial state requires durable records |
| `trade_events` | Immutable named trade transitions | PK `id`; event idempotency key required | `trade_id → trades RESTRICT` | `(trade_id,occurred_at)` | 288 B | 250 | ≥7 years | Current trade row cannot preserve lifecycle audit |
| `performance_daily` | Bounded daily aggregate | PK `(trading_date,symbol)` | None | PK | 400 B | 1 | 7 years; approved partition cleanup | Prevents repeated full trade scans for dashboard/reporting |

Rejected explicitly: any table containing full candle arrays, SMC/Liquidity/Volume/Flow objects, Quant vectors, prompts, provider request bodies, evidence trees, snapshots, graphs, or repeated normalized analytical payloads.

## H. Dashboard contract review

The authoritative endpoint is:

```text
GET /api/v2/dashboard/state
```

It is a pure read and includes:

- market-data status, exact reason, current poll timing, next poll, retry time, latest closed M1, freshness, last success, duration;
- analysis status, exact reason, active cycle/boundary, latest completed boundary, next boundary, last success, duration, retry time;
- Market State, SMC, Liquidity, Volume Profile, Institutional Flow, Quant, AI, Proposal, Guardrails, Final Action, Publication, Monitoring, and Outcome states;
- AI attempted, start, completion, model, terminal outcome, exact failure, retry count;
- current compact result, confidence, setup family, actionable entry/SL/targets only when present, risk;
- database health, exact storage circuit reason, capacity/used/free, growth;
- compact performance and meaningful history.

Each stage has a backend-owned status and exact reason. The frontend performs presentation only. The endpoint needs no join or interpretation across history tables: the Dashboard Service makes bounded reads of the singleton, current cycle, latest M1, and already-filtered meaningful history, then returns one normalized response. Repeated refresh produces zero writes.

## I. Migration and cutover review

| Phase | Change | Required proof | Rollback |
|---|---|---|---|
| 1 | Introduce additive V2 schema/contracts | Migration applies/rolls back on duplicated V1 fixture without touching V1 | Disable V2 routing; leave additive tables |
| 2 | Shadow V2 M1 write path | One-minute cadence, M1 uniqueness, open update, closed immutability | Disable V2 M1 flag |
| 3 | Observe scheduler | Exact UTC boundaries, no startup/poll/dashboard coupling | Disable enqueue flag |
| 4 | Shadow deterministic analysis, AI off | Point-in-time parity and no payload persistence | Disable V2 analysis |
| 5 | Shadow AI | One request/cycle, complete WAIT, typed failure | Disable V2 AI |
| 6 | Measure storage | ≤5 MiB/day projected including indexes; writers classified | Block release |
| 7 | Switch dashboard reads | Pure reads, all states/reasons, no generic `Unavailable` | Route to V1 adapter |
| 8 | Switch authoritative analysis | Generation-fenced single owner, publication still controlled | Fence V2, re-enable selected V1 generation |
| 9 | Stop V1 writers | `TEN_V1_WRITERS_ENABLED=false`; confirm zero V1 SQL writes | Temporarily re-enable only after V2 fenced |
| 10 | Archive old data | Backup, inventory, retention and separate approval | Retain source archive; no destructive initial step |
| 11 | Remove V1 | Explicit later approval only | Restore from retained code/data if approved plan permits |

V1 and V2 full-payload writers receive a fixed shadow end date before phase 2 begins. Missing that date blocks rollout. No compatibility adapter may write legacy payloads.

## J. Acceptance gates

Implementation must not begin until all are true:

- [x] Market polling and analysis are structurally separate in the design.
- [x] Analysis is aligned to absolute UTC five-minute boundaries.
- [x] At most one physical AI request exists per cycle.
- [x] Dashboard refresh is write-free by contract.
- [x] WAIT has a complete terminal state.
- [x] Heavy analytical objects are memory-only.
- [x] Every planned database writer is known and bounded.
- [x] Every planned table has defined readers.
- [x] Projected normal growth is 1.96 MiB/day and has a 5 MiB/day hard gate.
- [x] V1 writers have a named kill switch and generation fencing.
- [x] Rollback is additive and non-destructive.
- [x] Generic `Unavailable` is forbidden.
- [ ] Redis infrastructure choice approved.
- [ ] Single-symbol current-state assumption approved.
- [ ] Retention and restricted-diagnostic policies approved.
- [ ] Engine optional/mandatory dependency map approved.
- [ ] Storage/WAL ceiling approved.
- [ ] Shadow duration and parity thresholds approved.

## K. Unresolved decisions requiring human approval

1. Approve Redis Streams/hashes or select another explicit queue/runtime registry with equivalent failure semantics.
2. Confirm V2 launch is XAUUSD-only and therefore uses one global current-state row.
3. Approve the 90-second misfire window.
4. Approve M1-only persistence and no provider-native M5/M15 exception.
5. Approve retention periods and the legal/audit treatment of trade-linked events.
6. Approve the engine dependency matrix specifying which degraded evidence may reach Quant/AI.
7. Permit or prohibit the seven-day encrypted AI incident archive.
8. Approve the 5 MiB/day PostgreSQL ceiling and decide a separate WAL budget.
9. Approve shadow duration and numerical V1/V2 parity thresholds.
10. Approve the closed-candle policy: automated late corrections remain rejected and require a separately reviewed administrative process.

## Projected daily storage growth

For one continuously active symbol:

```text
rows/day × average heap bytes × index amplification

candles             1,440 × 240   × 1.60 =   552,960 B
cycles                288 × 1,600 × 1.80 =   829,440 B
lifecycle events      100 × 800   × 1.80 =   144,000 B
trades/events          budgeted combined     = 163,200 B
performance              1 × 400   × 1.50 =       600 B
temporary failures                              102,400 B
index/tuple reserve                              262,144 B
                                               -----------
total                                        2,054,744 B
                                             1.96 MiB/day
```

Normal PostgreSQL diagnostics: `0 B/day`. Optional restricted external diagnostics: `≤256 KiB/day` while incident mode is explicitly enabled. The implementation gate fails above `5 MiB/day/symbol`, before retention is credited.

## Recommended phased implementation plan

Implementation is recommended only after section K is approved:

1. Contracts and additive schema, with writer-classification instrumentation.
2. Canonical M1 writer and gap/immutability tests.
3. Redis runtime state and pure-read dashboard contract.
4. Absolute-boundary scheduler, cycle uniqueness, leases, and fencing.
5. Bounded in-memory analytical engine through Quant, without AI.
6. AI reservation and one-request adapter with complete typed terminal outcomes.
7. Proposal, guardrails, final-action, current-state, and meaningful-event policy.
8. Storage budget, disk-full circuit, restart/idempotency, and failure-matrix integration tests.
9. Time-boxed V1/V2 shadow verification.
10. Dashboard cutover, authoritative-engine cutover, immediate V1 writer shutdown, observation, then separately approved archive/removal.

No commit, push, pull request, migration execution, deployment, infrastructure mutation, or production data change was performed by this audit.
