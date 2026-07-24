# Dashboard and persistence production-readiness report

## Measured root cause

Read-only production measurements were taken on 2026-07-24. PostgreSQL contained
1,778,587,327 bytes (1,696 MB) after 4 hours 34 minutes of uptime: approximately
371 MB/hour, or a linear 5 GB exhaustion time of about 13.5 hours.

The same analytical output was serialized into three JSONB ownership locations:

1. `market_evidence_frames.payload`
2. every `evidence_items.payload`
3. `unified_market_states.payload`, including every evidence item again

The writer was `SqlAlchemyUnifiedMarketStateRepository.save_state()`. The corresponding
capture path is `UnifiedMarketStateService._frame()` and `synchronize()`.

Measured production relations:

| Relation | Total bytes | Live rows | Average total bytes/row |
|---|---:|---:|---:|
| `evidence_items` | 602,636,288 | 7,354 | 81,946 |
| `unified_market_states` | 596,213,760 | 353 | 1,688,990 |
| `market_evidence_frames` | 182,099,968 | 356 | 511,516 |
| `smc_analysis_snapshots` | 65,798,144 | 355 | 185,346 |
| `liquidity_snapshots` | 52,289,536 | 352 | 148,549 |
| `volume_profile_snapshots` | 42,745,856 | 352 | 121,437 |

The first three relations used approximately 1.381 GB, 81.4% of the database.
TOAST, not indexes or dead rows, owned almost all of that space. This rules out index
bloat and vacuum lag as the primary cause.

## Remediation

The immutable `market_evidence_frames` row is now the sole owner of the complete raw,
normalized and provenance payload. State and evidence rows persist only identity,
availability, timestamps, classification and relational links. Reads reconstruct the
same validated `UnifiedMarketState` from the immutable frame. Legacy full-payload rows
remain readable, allowing rolling deployment without a destructive migration.

The implementation also adds:

- one `unified_market_state_current` pointer per instrument;
- fingerprint-unique stage history and one current row per instrument/stage;
- an authoritative 13-stage `/api/dashboard/system-status` response;
- database/relation/row/dead-row diagnostics;
- a bounded cleanup worker which may delete only allowlisted, non-protected transient
  relations;
- a 4 GB database-size alert (configurable with
  `TEN_DATABASE_SIZE_ALERT_BYTES`);
- a five-minute `storage_exhausted` circuit breaker to prevent outbox retry storms.

Historical candles already use an idempotent
`(symbol, timeframe, timestamp)` conflict target. Realtime observations remain transient
and have a seven-day retention policy.

## Retention and audit policy

| Data | Retention | Behavior |
|---|---:|---|
| Canonical historical candles | Indefinite | Idempotent upsert |
| Realtime candle observations | 7 days | Bounded automatic cleanup |
| Pipeline stage transitions | 30 days | Only meaningful fingerprint changes |
| Current stage/state rows | Indefinite | Upserted current pointer |
| UMS and immutable evidence | 30 days minimum | Protected; archive/backup confirmation required |
| AI decisions/publications/outcomes | Indefinite | Audit records, no automatic deletion |

The cleanup worker cannot delete a relation merely because a database policy row names
it. Relation and timestamp identifiers must also be compiled into its allowlist, and no
transaction deletes more than 5,000 rows.

## Expected growth change

The production-measured duplicate owners represented 81.4% of the database. A read-only
`pg_column_size` sample over the latest 100 production states measured:

| Per-cycle component | Legacy bytes | Compact bytes |
|---|---:|---:|
| Unified state row | 1,653,007 | 872 |
| Evidence rows | 1,659,552 | 16,619 |
| Immutable full frame | 464,994 | 464,994 |
| Timeframe references | 243 | 243 |
| Evidence links | 1,260 | 1,260 |
| **Total** | **3,779,056** | **483,988** |

That is an 87.19% reduction in UMS persistence bytes per analytical cycle.

At the same read-only measurement point production had 433 states, 78.79 cycles/hour,
2,159,580,863 database bytes and an observed whole-database growth rate of
390,960,219 bytes/hour. The UMS ownership redesign changes its contribution from
283.96 MiB/hour to 36.37 MiB/hour. Holding all other writers constant, remaining total
growth is projected at 125.26 MiB/hour or 2.94 GiB/day. This is materially safer but
still too high for a small volume; the dashboard therefore keeps growth visible and the
protected snapshot families require a separately approved archive/retention phase.

These are measured row sizes and a projection from the observed cycle rate, not a claimed
post-deployment production measurement. Exact post-deployment bytes/hour still must be
measured over at least 60 minutes after deployment.

No automatic rewrite or deletion of the existing 1.381 GB is performed because a JSONB
rewrite can temporarily require additional disk and the backup/archive status is not
known. Old rows can be compacted in bounded batches after backup verification.

## Deployment and rollback

1. Back up PostgreSQL and verify restore metadata.
2. Deploy the migration before the application revision.
3. Verify `/api/dashboard/system-status`, UMS reconstruction and one full analytical cycle.
4. Record database bytes and top-relation bytes at 15-minute intervals for one hour.
5. Keep immutable-state cleanup protected until archive verification is complete.

Before the new application accepts writes, the migration can be downgraded directly.
After the first compact row is written, do not roll back to the previous application:
the new application reads both legacy and compact rows, but the previous application
cannot read compact rows. Restore the pre-deployment backup or run an approved
re-expansion migration before reverting application code.
