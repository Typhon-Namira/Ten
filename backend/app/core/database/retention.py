"""Cross-cutting age-based retention for tables with no engine-specific pruning.

`ai_scoring_engine`, `signal_decision_engine`, and `market_regime_engine` already have their own
retention: each engine's repository owns the constraints its own writes create, so its own
`prune()`/`prune_history()` is the safest place for that logic to live, and this module does not
duplicate it — `RetentionWorker` just calls those existing methods (`replay_engine` too) alongside
the deletes below.

The deletes below cover tables that never got an engine-specific retention path at all. Every
target here was checked against the full foreign-key graph in `backend/app/storage/models.py`
before being included — either the table has no incoming `ForeignKey` reference (safe to delete
by age alone), or an explicit `NOT EXISTS` guard is used to skip rows a `RESTRICT` constraint
would otherwise reject. `unified_market_states` and everything hanging off it (`ai_reasoning_*`,
`ai_market_forecasts`, `managed_signals`, `final_system_actions`, `guardrail_evaluations`,
`published_analytical_signals`, the `quant_forecast_*`/`quant_feature_*` tables, and
`market_evidence_frames`/`evidence_items`) are deliberately NOT included — that cluster has a deep,
multi-hop mix of `CASCADE` and `RESTRICT` foreign keys (e.g. `final_system_actions` RESTRICTs on
`unified_market_states`, `quantitative_forecasts`, AND `ai_market_forecasts` simultaneously) that
needs its own dependency-ordered design and verification against a real PostgreSQL instance before
any row in it is deleted — see docs for the follow-up this defers to.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
from typing import Any, Protocol

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import InstrumentedAttribute

from backend.app.storage.models import (
    GapHistoryRecord,
    InstitutionalFlowEvidenceRecord,
    InstitutionalFlowSnapshotRecord,
    IntegrationDataQualityIssueRecord,
    IntegrationEventRecord,
    IntegrationEventTraceRecord,
    IntegrationOutboxRecord,
    IntegrationSnapshotRecord,
    LatencyHistoryRecord,
    LiquidityObjectRecord,
    LiquiditySnapshotRecord,
    MarketRegimeEvidenceRecord,
    OperationalSignalRecord,
    ProviderMetricRecord,
    QualityHistoryRecord,
    RealtimeCandleRecord,
    SMCAnalysisSnapshotRecord,
    SMCObjectRecord,
    SynchronizationHistoryRecord,
    VolumeProfileObjectRecord,
    VolumeProfileSnapshotRecord,
)

logger = logging.getLogger(__name__)

# Hard ceiling on how many batches one target processes per `run_once()` cycle — bounds a single
# retention pass even if a target is carrying years of backlog, so the worker always returns
# control to its own poll loop instead of monopolizing a session indefinitely.
_MAXIMUM_BATCHES_PER_TARGET = 20


class _Cleanable(Protocol):
    async def cleanup(self) -> int: ...


@dataclass(frozen=True)
class _DeleteTarget:
    label: str
    table: type
    pk_column: InstrumentedAttribute[Any]
    ts_column: InstrumentedAttribute[Any]
    guard: Any | None = None  # extra WHERE clause, e.g. a NOT EXISTS subquery


class RetentionRepository:
    """Bounded, batched age-based deletes for tables without their own retention path."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None) -> None:
        # `None` is only valid when the owning `RetentionWorker` is constructed with
        # `enabled=False` (in-memory/dev mode, no PostgreSQL configured) — `start()` then never
        # runs the loop that would call `prune()`, so the factory is never dereferenced.
        self._session_factory = session_factory

    async def _delete_batch(self, session: AsyncSession, target: _DeleteTarget, cutoff: datetime, batch_size: int) -> int:
        query = select(target.pk_column).where(target.ts_column < cutoff)
        if target.guard is not None:
            query = query.where(target.guard)
        query = query.order_by(target.ts_column).limit(batch_size)
        ids = (await session.scalars(query)).all()
        if not ids:
            return 0
        await session.execute(delete(target.table).where(target.pk_column.in_(ids)))
        return len(ids)

    async def _prune_target(self, target: _DeleteTarget, cutoff: datetime, batch_size: int) -> int:
        if self._session_factory is None:
            raise RuntimeError("RetentionRepository.prune() called without a configured database session factory")
        deleted = 0
        for _ in range(_MAXIMUM_BATCHES_PER_TARGET):
            async with self._session_factory() as session:
                try:
                    removed = await self._delete_batch(session, target, cutoff, batch_size)
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise
            deleted += removed
            if removed < batch_size:
                break
        return deleted

    def analytical_object_targets(self, cutoff: datetime) -> tuple[_DeleteTarget, ...]:
        return (
            _DeleteTarget("smc_objects", SMCObjectRecord, SMCObjectRecord.id, SMCObjectRecord.created_at),
            _DeleteTarget("liquidity_objects", LiquidityObjectRecord, LiquidityObjectRecord.id, LiquidityObjectRecord.created_at),
            _DeleteTarget("volume_profile_objects", VolumeProfileObjectRecord, VolumeProfileObjectRecord.id, VolumeProfileObjectRecord.created_at),
            _DeleteTarget("institutional_flow_evidence", InstitutionalFlowEvidenceRecord, InstitutionalFlowEvidenceRecord.id, InstitutionalFlowEvidenceRecord.created_at),
            _DeleteTarget("market_regime_evidence", MarketRegimeEvidenceRecord, MarketRegimeEvidenceRecord.id, MarketRegimeEvidenceRecord.created_at),
        )

    def analytical_snapshot_targets(self, cutoff: datetime) -> tuple[_DeleteTarget, ...]:
        # market_regime_snapshots is intentionally excluded: `MarketRegimeService` already calls
        # `repository.prune_history()` inline after every save (a working, tested path) — adding a
        # second, differently-windowed pruner for the same table here would race it for no benefit.
        return (
            _DeleteTarget("smc_analysis_snapshots", SMCAnalysisSnapshotRecord, SMCAnalysisSnapshotRecord.id, SMCAnalysisSnapshotRecord.created_at),
            _DeleteTarget("liquidity_snapshots", LiquiditySnapshotRecord, LiquiditySnapshotRecord.id, LiquiditySnapshotRecord.created_at),
            _DeleteTarget("volume_profile_snapshots", VolumeProfileSnapshotRecord, VolumeProfileSnapshotRecord.id, VolumeProfileSnapshotRecord.created_at),
            _DeleteTarget("institutional_flow_snapshots", InstitutionalFlowSnapshotRecord, InstitutionalFlowSnapshotRecord.id, InstitutionalFlowSnapshotRecord.created_at),
        )

    def integration_audit_targets(self, cutoff: datetime) -> tuple[_DeleteTarget, ...]:
        # `integration_events` cascades to `integration_outbox`/`integration_processed_events` on
        # delete, but only once its own outbox item is actually completed — the guard below skips
        # any event whose outbox row is still unpublished, so an event never disappears out from
        # under a job that hasn't drained it yet.
        pending_outbox = select(IntegrationOutboxRecord.event_id).where(
            IntegrationOutboxRecord.event_id == IntegrationEventRecord.event_id, IntegrationOutboxRecord.published_at.is_(None)
        )
        # `integration_snapshots` is RESTRICT-referenced by `operational_signals.snapshot_id` —
        # only prune a snapshot once no operational signal still points at it.
        referencing_signal = select(OperationalSignalRecord.operational_signal_id).where(OperationalSignalRecord.snapshot_id == IntegrationSnapshotRecord.snapshot_id)
        return (
            _DeleteTarget("integration_events", IntegrationEventRecord, IntegrationEventRecord.event_id, IntegrationEventRecord.available_at, guard=~pending_outbox.exists()),
            _DeleteTarget("integration_snapshots", IntegrationSnapshotRecord, IntegrationSnapshotRecord.snapshot_id, IntegrationSnapshotRecord.analytical_boundary, guard=~referencing_signal.exists()),
            _DeleteTarget("integration_event_trace", IntegrationEventTraceRecord, IntegrationEventTraceRecord.trace_record_id, IntegrationEventTraceRecord.started_at),
            _DeleteTarget("integration_data_quality_issues", IntegrationDataQualityIssueRecord, IntegrationDataQualityIssueRecord.issue_id, IntegrationDataQualityIssueRecord.observed_at),
        )

    def operational_signal_targets(self, cutoff: datetime) -> tuple[_DeleteTarget, ...]:
        return (_DeleteTarget("operational_signals", OperationalSignalRecord, OperationalSignalRecord.operational_signal_id, OperationalSignalRecord.expires_at),)

    def market_data_history_targets(self, cutoff: datetime) -> tuple[_DeleteTarget, ...]:
        # `historical_candles` is deliberately absent — it is the deterministic OHLCV series every
        # analytical engine reads its lookback window from, not diagnostic exhaust.
        return (
            _DeleteTarget("provider_metrics", ProviderMetricRecord, ProviderMetricRecord.id, ProviderMetricRecord.captured_at),
            _DeleteTarget("market_quality_history", QualityHistoryRecord, QualityHistoryRecord.id, QualityHistoryRecord.timestamp),
            _DeleteTarget("market_gap_history", GapHistoryRecord, GapHistoryRecord.id, GapHistoryRecord.start_at),
            _DeleteTarget("market_latency_history", LatencyHistoryRecord, LatencyHistoryRecord.id, LatencyHistoryRecord.captured_at),
            _DeleteTarget("market_synchronization_history", SynchronizationHistoryRecord, SynchronizationHistoryRecord.id, SynchronizationHistoryRecord.started_at),
            _DeleteTarget("realtime_candles", RealtimeCandleRecord, RealtimeCandleRecord.id, RealtimeCandleRecord.received_at),
        )

    async def prune(
        self,
        *,
        now: datetime,
        analytical_object_retention_days: int,
        analytical_snapshot_retention_days: int,
        integration_audit_retention_days: int,
        operational_signal_retention_days: int,
        market_data_history_retention_days: int,
        batch_size: int,
    ) -> dict[str, int]:
        groups = (
            (self.analytical_object_targets, analytical_object_retention_days),
            (self.analytical_snapshot_targets, analytical_snapshot_retention_days),
            (self.integration_audit_targets, integration_audit_retention_days),
            (self.operational_signal_targets, operational_signal_retention_days),
            (self.market_data_history_targets, market_data_history_retention_days),
        )
        deleted: dict[str, int] = {}
        for builder, retention_days in groups:
            cutoff = now - timedelta(days=retention_days)
            for target in builder(cutoff):
                deleted[target.label] = await self._prune_target(target, cutoff, batch_size)
        return deleted


class RetentionWorker:
    """Periodic housekeeping loop: age-based deletes plus the engines' own `cleanup()` methods.

    Mirrors `MarketDataWorker`/`IntegrationWorker`'s crash-guard shape (`backend/app/engines/
    market_data_engine/worker.py`, `backend/app/integration/worker.py`): a per-iteration
    try/except that logs and continues, plus an `add_done_callback` backstop so a bug that
    somehow escapes the loop body is still visible as a CRITICAL log line and a `status()["crashed"]`
    flag instead of silently stopping the worker with no trace anywhere.
    """

    def __init__(
        self,
        repository: RetentionRepository,
        *,
        enabled: bool,
        interval_seconds: float,
        batch_size: int,
        analytical_object_retention_days: int,
        analytical_snapshot_retention_days: int,
        integration_audit_retention_days: int,
        operational_signal_retention_days: int,
        market_data_history_retention_days: int,
        cleanable_services: tuple[_Cleanable, ...] = (),
        clock: Any = lambda: datetime.now(UTC),
    ) -> None:
        self.repository = repository
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self.analytical_object_retention_days = analytical_object_retention_days
        self.analytical_snapshot_retention_days = analytical_snapshot_retention_days
        self.integration_audit_retention_days = integration_audit_retention_days
        self.operational_signal_retention_days = operational_signal_retention_days
        self.market_data_history_retention_days = market_data_history_retention_days
        self.cleanable_services = cleanable_services
        self.clock = clock
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.last_heartbeat_at: datetime | None = None
        self.last_success_at: datetime | None = None
        self.last_error: str | None = None
        self.last_result: dict[str, int] = {}
        self.consecutive_failures = 0
        self.last_fatal_error: str | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self.run(), name="ten-retention-worker")
            self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.last_fatal_error = f"{type(exc).__name__}: {exc}"
            logger.critical("retention.worker.task_died", exc_info=exc, extra={"worker": "retention", "error_type": type(exc).__name__})

    async def run(self) -> None:
        while not self._stop.is_set():
            self.last_heartbeat_at = self.clock()
            logger.info("worker.heartbeat", extra={"worker": "retention"})
            try:
                result = await self.run_once()
                self.last_result = result
                self.last_success_at = self.clock()
                self.last_error = None
                self.consecutive_failures = 0
                total = sum(result.values())
                if total:
                    logger.info("retention.cycle.completed", extra={"worker": "retention", "deleted_by_table": result, "deleted_total": total})
            except Exception as exc:
                self.last_error = type(exc).__name__
                self.consecutive_failures += 1
                logger.exception("retention.worker.failed", extra={"worker": "retention", "error_type": self.last_error})
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                pass

    async def run_once(self) -> dict[str, int]:
        now = self.clock()
        deleted = await self.repository.prune(
            now=now,
            analytical_object_retention_days=self.analytical_object_retention_days,
            analytical_snapshot_retention_days=self.analytical_snapshot_retention_days,
            integration_audit_retention_days=self.integration_audit_retention_days,
            operational_signal_retention_days=self.operational_signal_retention_days,
            market_data_history_retention_days=self.market_data_history_retention_days,
            batch_size=self.batch_size,
        )
        for service in self.cleanable_services:
            label = type(service).__name__
            started = self.clock()
            try:
                count = await service.cleanup()
            except Exception:
                logger.exception("retention.engine_cleanup.failed", extra={"worker": "retention", "service": label})
                continue
            duration_ms = (self.clock() - started).total_seconds() * 1000
            deleted[label] = count
            if count:
                logger.info("retention.engine_cleanup.completed", extra={"worker": "retention", "service": label, "deleted": count, "duration_ms": duration_ms})
        return deleted

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    def status(self) -> dict[str, object]:
        running = self._task is not None and not self._task.done()
        return {
            "enabled": self.enabled,
            "running": running,
            "crashed": self.enabled and not running,
            "last_heartbeat_at": self.last_heartbeat_at,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "last_fatal_error": self.last_fatal_error,
            "consecutive_failures": self.consecutive_failures,
            "last_result": self.last_result,
        }
