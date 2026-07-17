"""SMC persistence ports and deterministic in-process adapter."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.market_data_engine.models import canonical_symbol
from backend.app.storage.models import SMCAnalysisSnapshotRecord, SMCCheckpointRecord, SMCObjectRecord

from .models import SMCAnalysisSnapshot, stable_id


class SMCRepository(ABC):
    @abstractmethod
    async def save(self, snapshot: SMCAnalysisSnapshot) -> None:
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
        self._ids: set[object] = set()
        self._lock = asyncio.Lock()

    async def save(self, snapshot: SMCAnalysisSnapshot) -> None:
        async with self._lock:
            if snapshot.id in self._ids:
                return
            key = (canonical_symbol(snapshot.symbol), snapshot.timeframe)
            items = self._snapshots.setdefault(key, [])
            items.append(snapshot)
            items.sort(key=lambda item: item.analysis_timestamp)
            self._ids.add(snapshot.id)

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


class SqlAlchemySMCRepository(SMCRepository):
    """PostgreSQL adapter with idempotent writes and indexed time travel."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, snapshot: SMCAnalysisSnapshot) -> None:
        values = {"id": snapshot.id, "symbol": canonical_symbol(snapshot.symbol), "timeframe": snapshot.timeframe.value, "analysis_timestamp": snapshot.analysis_timestamp, "market_data_boundary": snapshot.market_data_boundary, "status": snapshot.status.value, "processing_mode": snapshot.processing_mode.value, "engine_version": snapshot.engine_version, "configuration_version": snapshot.configuration_version, "payload": snapshot.model_dump(mode="json"), "created_at": snapshot.created_at}
        await self.session.execute(insert(SMCAnalysisSnapshotRecord).values(values).on_conflict_do_nothing(index_elements=["id"]))
        objects = self._objects(snapshot)
        if objects:
            await self.session.execute(insert(SMCObjectRecord).values(objects).on_conflict_do_nothing(index_elements=["id"]))
        checkpoint = {"symbol": canonical_symbol(snapshot.symbol), "timeframe": snapshot.timeframe.value, "configuration_version": snapshot.configuration_version, "snapshot_id": snapshot.id, "last_processed_candle": snapshot.analysis_timestamp, "state_payload": snapshot.model_dump(mode="json"), "updated_at": snapshot.created_at}
        statement = insert(SMCCheckpointRecord).values(checkpoint)
        await self.session.execute(statement.on_conflict_do_update(index_elements=["symbol", "timeframe", "configuration_version"], set_={name: getattr(statement.excluded, name) for name in ("snapshot_id", "last_processed_candle", "state_payload", "updated_at")}))
        await self.session.commit()

    async def latest(self, symbol: str, timeframe: Timeframe) -> SMCAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, None)

    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> SMCAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, timestamp)

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
