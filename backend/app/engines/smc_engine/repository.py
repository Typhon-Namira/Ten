"""SMC persistence ports and deterministic in-process adapter."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging
from time import perf_counter
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.market_data_engine.models import canonical_symbol
from backend.app.storage.models import SMCAnalysisSnapshotRecord, SMCCheckpointRecord, SMCObjectRecord
from backend.app.storage.batching import DEFAULT_BIND_PARAMETER_BUDGET, bounded_insert_chunks, maximum_rows_per_insert
from backend.app.storage.logical_identity import analytical_snapshot_boundary, ensure_analytical_determinism, returned_identity
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import SMCAnalysisSnapshot, stable_id

logger = logging.getLogger(__name__)


@dataclass
class SMCPersistenceMetrics:
    attempted_objects: int = 0
    inserted_objects: int = 0
    skipped_objects: int = 0
    chunk_count: int = 0
    duration_ms: float = 0.0
    failures: int = 0

    def snapshot(self) -> dict[str, int | float]:
        return dict(vars(self))


class SMCRepository(ABC):
    @abstractmethod
    async def save(self, snapshot: SMCAnalysisSnapshot, *, correlation_id: UUID | None = None) -> None:
        """Persist an immutable versioned analysis snapshot."""

    @abstractmethod
    async def latest(self, symbol: str, timeframe: Timeframe) -> SMCAnalysisSnapshot | None:
        """Return the latest snapshot for a series."""

    @abstractmethod
    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> SMCAnalysisSnapshot | None:
        """Return the last snapshot available at or before timestamp."""

    @abstractmethod
    async def checkpoints(self) -> tuple[SMCAnalysisSnapshot, ...]:
        """Restore the latest durable state for every configured series."""


class InMemorySMCRepository(SMCRepository):
    def __init__(self) -> None:
        self._snapshots: dict[tuple[str, Timeframe], list[SMCAnalysisSnapshot]] = {}
        self._boundaries: dict[tuple[str, ...], SMCAnalysisSnapshot] = {}
        self._lock = asyncio.Lock()

    async def save(self, snapshot: SMCAnalysisSnapshot, *, correlation_id: UUID | None = None) -> None:
        async with self._lock:
            boundary = analytical_snapshot_boundary(snapshot, include_processing_mode=True)
            existing = self._boundaries.get(boundary)
            if existing is not None:
                ensure_analytical_determinism(existing, snapshot, entity_type="smc_analysis_snapshot", include_processing_mode=True)
                return
            key = (canonical_symbol(snapshot.symbol), snapshot.timeframe)
            items = self._snapshots.setdefault(key, [])
            items.append(snapshot)
            items.sort(key=lambda item: item.analysis_timestamp)
            self._boundaries[boundary] = snapshot

    async def latest(self, symbol: str, timeframe: Timeframe) -> SMCAnalysisSnapshot | None:
        async with self._lock:
            items = self._snapshots.get((canonical_symbol(symbol), timeframe), [])
            return items[-1] if items else None

    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> SMCAnalysisSnapshot | None:
        async with self._lock:
            items = self._snapshots.get((canonical_symbol(symbol), timeframe), [])
            eligible = [item for item in items if item.analysis_timestamp <= timestamp]
            return eligible[-1] if eligible else None

    async def checkpoints(self) -> tuple[SMCAnalysisSnapshot, ...]:
        async with self._lock:
            return tuple(items[-1] for items in self._snapshots.values() if items)


class SqlAlchemySMCRepository(SMCRepository, ScopedSessionRepository):
    """PostgreSQL adapter with idempotent writes and indexed time travel.

    Each method call opens and closes its own `AsyncSession` (see `scoped_session`) — no session
    is ever shared across concurrent callers or held beyond one call.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        ScopedSessionRepository.__init__(self, session_factory)
        self.metrics = SMCPersistenceMetrics()

    @scoped_session
    async def save(self, snapshot: SMCAnalysisSnapshot, *, correlation_id: UUID | None = None) -> None:
        values = {"id": snapshot.id, "symbol": canonical_symbol(snapshot.symbol), "timeframe": snapshot.timeframe.value, "analysis_timestamp": snapshot.analysis_timestamp, "market_data_boundary": snapshot.market_data_boundary, "status": snapshot.status.value, "processing_mode": snapshot.processing_mode.value, "engine_version": snapshot.engine_version, "configuration_version": snapshot.configuration_version, "payload": snapshot.model_dump(mode="json"), "created_at": snapshot.created_at}
        started = perf_counter()
        objects = sorted(self._objects(snapshot), key=lambda item: (item["analytical_timestamp"], item["object_type"], str(item["id"])))
        chunks = bounded_insert_chunks(objects)
        parameters_per_row = len(objects[0]) if objects else 0
        rows_per_chunk = maximum_rows_per_insert(parameters_per_row) if parameters_per_row else 0
        attempted = len(objects)
        inserted = 0
        context = {
            "correlation_id": str(correlation_id) if correlation_id else None,
            "cycle_id": str(snapshot.id),
            "instrument": canonical_symbol(snapshot.symbol),
            "timeframe": snapshot.timeframe.value,
            "total_objects": attempted,
            "total_chunks": len(chunks),
            "parameters_per_row": parameters_per_row,
            "bind_parameter_budget": DEFAULT_BIND_PARAMETER_BUDGET,
            "rows_per_chunk": rows_per_chunk,
        }
        logger.info("smc.persistence.started", extra=context)
        try:
            insert_result = await self.session.execute(
                insert(SMCAnalysisSnapshotRecord)
                .values(values)
                .on_conflict_do_nothing(
                    index_elements=[
                        SMCAnalysisSnapshotRecord.symbol,
                        SMCAnalysisSnapshotRecord.timeframe,
                        SMCAnalysisSnapshotRecord.analysis_timestamp,
                        SMCAnalysisSnapshotRecord.configuration_version,
                        SMCAnalysisSnapshotRecord.processing_mode,
                    ]
                )
                .returning(SMCAnalysisSnapshotRecord.id)
            )
            inserted_id = returned_identity(insert_result, snapshot.id)
            persisted_snapshot = snapshot
            if inserted_id is None:
                persisted_record = (
                    await self.session.scalars(
                        select(SMCAnalysisSnapshotRecord).where(
                            SMCAnalysisSnapshotRecord.symbol == canonical_symbol(snapshot.symbol),
                            SMCAnalysisSnapshotRecord.timeframe == snapshot.timeframe.value,
                            SMCAnalysisSnapshotRecord.analysis_timestamp == snapshot.analysis_timestamp,
                            SMCAnalysisSnapshotRecord.configuration_version == snapshot.configuration_version,
                            SMCAnalysisSnapshotRecord.processing_mode == snapshot.processing_mode.value,
                        )
                    )
                ).first()
                if persisted_record is None:
                    raise RuntimeError("SMC snapshot insert did not resolve a canonical row")
                persisted_snapshot = SMCAnalysisSnapshot.model_validate(persisted_record.payload)
            ensure_analytical_determinism(
                persisted_snapshot,
                snapshot,
                entity_type="smc_analysis_snapshot",
                include_processing_mode=True,
            )
            for chunk_index, chunk in enumerate(chunks, start=1):
                chunk_context = {**context, "chunk_index": chunk_index, "chunk_size": len(chunk)}
                chunk_inserted = 0
                logger.info("smc.persistence.chunk.started", extra=chunk_context)
                try:
                    chunk_ids = [item["id"] for item in chunk]
                    existing_ids = set((await self.session.scalars(select(SMCObjectRecord.id).where(SMCObjectRecord.id.in_(chunk_ids)))).all())
                    pending = [item for item in chunk if item["id"] not in existing_ids]
                    if pending:
                        result = await self.session.execute(insert(SMCObjectRecord).values(pending).on_conflict_do_nothing(index_elements=["id"]))
                        rowcount = getattr(result, "rowcount", None)
                        chunk_inserted = rowcount if isinstance(rowcount, int) and rowcount >= 0 else len(pending)
                        inserted += chunk_inserted
                except Exception as exc:
                    logger.exception("smc.persistence.chunk.failed", extra={**chunk_context, "exception_type": type(exc).__name__})
                    raise
                logger.info("smc.persistence.chunk.completed", extra={**chunk_context, "inserted_objects": chunk_inserted, "skipped_objects": len(chunk) - chunk_inserted, "duration_ms": (perf_counter() - started) * 1000})
            checkpoint = {"symbol": canonical_symbol(snapshot.symbol), "timeframe": snapshot.timeframe.value, "configuration_version": snapshot.configuration_version, "snapshot_id": persisted_snapshot.id, "last_processed_candle": persisted_snapshot.analysis_timestamp, "state_payload": persisted_snapshot.model_dump(mode="json"), "updated_at": persisted_snapshot.created_at}
            statement = insert(SMCCheckpointRecord).values(checkpoint)
            await self.session.execute(statement.on_conflict_do_update(index_elements=["symbol", "timeframe", "configuration_version"], set_={name: getattr(statement.excluded, name) for name in ("snapshot_id", "last_processed_candle", "state_payload", "updated_at")}))
            await self.session.commit()
            self.metrics = SMCPersistenceMetrics(attempted_objects=attempted, inserted_objects=inserted, skipped_objects=max(0, attempted - inserted), chunk_count=len(chunks), duration_ms=(perf_counter() - started) * 1000, failures=self.metrics.failures)
            logger.info("smc.persistence.completed", extra={**context, **self.metrics.snapshot()})
        except Exception as exc:
            await self.session.rollback()
            self.metrics = SMCPersistenceMetrics(attempted_objects=attempted, inserted_objects=0, skipped_objects=0, chunk_count=len(chunks), duration_ms=(perf_counter() - started) * 1000, failures=self.metrics.failures + 1)
            logger.exception("smc.persistence.failed", extra={**context, "duration_ms": self.metrics.duration_ms, "exception_type": type(exc).__name__})
            raise

    @scoped_session
    async def latest(self, symbol: str, timeframe: Timeframe) -> SMCAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, None)

    @scoped_session
    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> SMCAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, timestamp)

    @scoped_session
    async def checkpoints(self) -> tuple[SMCAnalysisSnapshot, ...]:
        records = list((await self.session.scalars(select(SMCCheckpointRecord).order_by(SMCCheckpointRecord.updated_at))).all())
        return tuple(SMCAnalysisSnapshot.model_validate(item.state_payload) for item in records)

    async def _query(self, symbol: str, timeframe: Timeframe, timestamp: datetime | None) -> SMCAnalysisSnapshot | None:
        statement = select(SMCAnalysisSnapshotRecord).where(SMCAnalysisSnapshotRecord.symbol == canonical_symbol(symbol), SMCAnalysisSnapshotRecord.timeframe == timeframe.value)
        if timestamp is not None:
            statement = statement.where(SMCAnalysisSnapshotRecord.analysis_timestamp <= timestamp)
        record = (await self.session.scalars(statement.order_by(SMCAnalysisSnapshotRecord.analysis_timestamp.desc()).limit(1))).first()
        return SMCAnalysisSnapshot.model_validate(record.payload) if record is not None else None

    @staticmethod
    def _objects(snapshot: SMCAnalysisSnapshot) -> list[dict[str, object]]:
        values: list[dict[str, object]] = []
        for object_type, items in (("swing", snapshot.swings), ("structure_leg", snapshot.structure_legs), ("structure_event", snapshot.structure_events), ("displacement", snapshot.displacements), ("zone", snapshot.zones), ("liquidity_reference", snapshot.liquidity_references), ("dealing_range", snapshot.dealing_ranges)):
            for item in items:
                analytical_timestamp = getattr(item, "timestamp", snapshot.analysis_timestamp)
                availability_timestamp = getattr(item, "confirmed_at", None) or analytical_timestamp
                version = int(getattr(item, "version", 1))
                lifecycle = str(getattr(getattr(item, "lifecycle_state", None), "value", "confirmed"))
                record_id = stable_id("object-version", snapshot.symbol, snapshot.timeframe, item.id, version, lifecycle)
                values.append({"id": record_id, "object_type": object_type, "symbol": canonical_symbol(snapshot.symbol), "timeframe": snapshot.timeframe.value, "analytical_timestamp": analytical_timestamp, "availability_timestamp": availability_timestamp, "lifecycle_state": lifecycle, "confidence_score": item.confidence_score, "quality_score": getattr(item, "quality_score", 100.0), "algorithm_version": getattr(item, "algorithm_version", snapshot.engine_version), "configuration_version": snapshot.configuration_version, "payload": item.model_dump(mode="json"), "created_at": snapshot.created_at})
        return values
