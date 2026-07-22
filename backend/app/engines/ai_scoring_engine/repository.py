from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.batching import bounded_insert_chunks
from backend.app.storage.models import AIScoreComponentRecord, AIScoreConflictRecord, AIScoreSnapshotRecord
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .exceptions import AIScoringPersistenceError
from .models import AIScoreSnapshot, ScoreMode, ScoreStatus


class AIScoringRepository(ABC):
    @abstractmethod
    async def save_snapshot(self, snapshot: AIScoreSnapshot) -> AIScoreSnapshot: ...

    @abstractmethod
    async def get_snapshot(self, snapshot_id: UUID) -> AIScoreSnapshot | None: ...

    @abstractmethod
    async def get_latest_snapshot(self, instrument: str, timeframe: str, policy_version: str | None = None) -> AIScoreSnapshot | None: ...

    @abstractmethod
    async def list_snapshots(
        self,
        instrument: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        status: ScoreStatus | None = None,
        policy_version: str | None = None,
        mode: ScoreMode | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[AIScoreSnapshot, ...]: ...

    @abstractmethod
    async def find_by_fingerprint(self, fingerprint: str, mode: ScoreMode) -> AIScoreSnapshot | None: ...

    @abstractmethod
    async def prune(self, before: datetime, mode: ScoreMode, limit: int) -> int: ...


class InMemoryAIScoringRepository(AIScoringRepository):
    def __init__(self) -> None:
        self._snapshots: dict[UUID, AIScoreSnapshot] = {}
        self._fingerprints: dict[tuple[str, ScoreMode], UUID] = {}
        self._lock = asyncio.Lock()

    async def save_snapshot(self, snapshot: AIScoreSnapshot) -> AIScoreSnapshot:
        key = (snapshot.metadata.input_fingerprint, snapshot.mode)
        async with self._lock:
            existing = self._fingerprints.get(key)
            if existing is not None:
                return self._snapshots[existing]
            self._snapshots[snapshot.snapshot_id] = snapshot
            self._fingerprints[key] = snapshot.snapshot_id
            return snapshot

    async def get_snapshot(self, snapshot_id: UUID) -> AIScoreSnapshot | None:
        async with self._lock:
            return self._snapshots.get(snapshot_id)

    async def get_latest_snapshot(self, instrument: str, timeframe: str, policy_version: str | None = None) -> AIScoreSnapshot | None:
        values = await self.list_snapshots(instrument, timeframe, policy_version=policy_version, limit=1)
        return values[0] if values else None

    async def list_snapshots(
        self,
        instrument: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        status: ScoreStatus | None = None,
        policy_version: str | None = None,
        mode: ScoreMode | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[AIScoreSnapshot, ...]:
        async with self._lock:
            values = [
                item
                for item in self._snapshots.values()
                if item.instrument == instrument
                and item.timeframe == timeframe
                and (start is None or item.as_of >= start)
                and (end is None or item.as_of <= end)
                and (status is None or item.status == status)
                and (policy_version is None or item.policy_version == policy_version)
                and (mode is None or item.mode == mode)
            ]
        values.sort(key=lambda item: (item.as_of, str(item.snapshot_id)), reverse=True)
        return tuple(values[offset : offset + limit])

    async def find_by_fingerprint(self, fingerprint: str, mode: ScoreMode) -> AIScoreSnapshot | None:
        async with self._lock:
            identifier = self._fingerprints.get((fingerprint, mode))
            return self._snapshots.get(identifier) if identifier else None

    async def prune(self, before: datetime, mode: ScoreMode, limit: int) -> int:
        async with self._lock:
            expired = sorted((item for item in self._snapshots.values() if item.mode == mode and item.as_of < before), key=lambda item: item.as_of)[:limit]
            for item in expired:
                self._snapshots.pop(item.snapshot_id, None)
                self._fingerprints.pop((item.metadata.input_fingerprint, item.mode), None)
            return len(expired)


class SqlAlchemyAIScoringRepository(AIScoringRepository, ScopedSessionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        ScopedSessionRepository.__init__(self, session_factory)

    @scoped_session
    async def save_snapshot(self, snapshot: AIScoreSnapshot) -> AIScoreSnapshot:
        try:
            existing = await self.find_by_fingerprint(snapshot.metadata.input_fingerprint, snapshot.mode)
            if existing:
                return existing
            await self.session.execute(
                insert(AIScoreSnapshotRecord).values(
                    id=snapshot.snapshot_id,
                    instrument=snapshot.instrument,
                    timeframe=snapshot.timeframe,
                    as_of=snapshot.as_of,
                    calculated_at=snapshot.calculated_at,
                    mode=snapshot.mode.value,
                    status=snapshot.status.value,
                    policy_name=snapshot.policy_name,
                    policy_version=snapshot.policy_version,
                    configuration_version=snapshot.metadata.configuration_version,
                    configuration_hash=snapshot.metadata.configuration_hash,
                    input_fingerprint=snapshot.metadata.input_fingerprint,
                    directional_score=snapshot.directional_score,
                    confidence_score=snapshot.confidence_score,
                    market_risk_score=snapshot.market_risk_score,
                    payload=snapshot.model_dump(mode="json"),
                ).on_conflict_do_nothing(index_elements=["input_fingerprint", "mode"])
            )
            component_rows = [
                {
                    "id": f"{snapshot.snapshot_id}:{index}",
                    "snapshot_id": snapshot.snapshot_id,
                    "source_engine": item.source_engine,
                    "source_group": item.source_group,
                    "payload": item.model_dump(mode="json"),
                }
                for index, item in enumerate(snapshot.components)
            ]
            conflict_rows = [
                {
                    "id": item.conflict_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "severity": item.severity,
                    "payload": item.model_dump(mode="json"),
                }
                for item in snapshot.conflicts
            ]
            if component_rows:
                for chunk in bounded_insert_chunks(component_rows):
                    await self.session.execute(insert(AIScoreComponentRecord).values(list(chunk)).on_conflict_do_nothing(index_elements=["id"]))
            if conflict_rows:
                for chunk in bounded_insert_chunks(conflict_rows):
                    await self.session.execute(insert(AIScoreConflictRecord).values(list(chunk)).on_conflict_do_nothing(index_elements=["id"]))
            await self.session.commit()
            return await self.find_by_fingerprint(snapshot.metadata.input_fingerprint, snapshot.mode) or snapshot
        except Exception as exc:
            await self.session.rollback()
            raise AIScoringPersistenceError("AI Scoring snapshot persistence failed") from exc

    @scoped_session
    async def get_snapshot(self, snapshot_id: UUID) -> AIScoreSnapshot | None:
        record = await self.session.get(AIScoreSnapshotRecord, snapshot_id)
        return AIScoreSnapshot.model_validate(record.payload) if record else None

    @scoped_session
    async def get_latest_snapshot(self, instrument: str, timeframe: str, policy_version: str | None = None) -> AIScoreSnapshot | None:
        query = select(AIScoreSnapshotRecord).where(AIScoreSnapshotRecord.instrument == instrument, AIScoreSnapshotRecord.timeframe == timeframe)
        if policy_version:
            query = query.where(AIScoreSnapshotRecord.policy_version == policy_version)
        record = (await self.session.scalars(query.order_by(AIScoreSnapshotRecord.as_of.desc()).limit(1))).first()
        return AIScoreSnapshot.model_validate(record.payload) if record else None

    @scoped_session
    async def list_snapshots(
        self,
        instrument: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        status: ScoreStatus | None = None,
        policy_version: str | None = None,
        mode: ScoreMode | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[AIScoreSnapshot, ...]:
        query = select(AIScoreSnapshotRecord).where(AIScoreSnapshotRecord.instrument == instrument, AIScoreSnapshotRecord.timeframe == timeframe)
        if start:
            query = query.where(AIScoreSnapshotRecord.as_of >= start)
        if end:
            query = query.where(AIScoreSnapshotRecord.as_of <= end)
        if status:
            query = query.where(AIScoreSnapshotRecord.status == status.value)
        if policy_version:
            query = query.where(AIScoreSnapshotRecord.policy_version == policy_version)
        if mode:
            query = query.where(AIScoreSnapshotRecord.mode == mode.value)
        records = (await self.session.scalars(query.order_by(AIScoreSnapshotRecord.as_of.desc(), AIScoreSnapshotRecord.id.desc()).offset(offset).limit(limit))).all()
        return tuple(AIScoreSnapshot.model_validate(item.payload) for item in records)

    @scoped_session
    async def find_by_fingerprint(self, fingerprint: str, mode: ScoreMode) -> AIScoreSnapshot | None:
        query = select(AIScoreSnapshotRecord).where(AIScoreSnapshotRecord.input_fingerprint == fingerprint, AIScoreSnapshotRecord.mode == mode.value).limit(1)
        record = (await self.session.scalars(query)).first()
        return AIScoreSnapshot.model_validate(record.payload) if record else None

    @scoped_session
    async def prune(self, before: datetime, mode: ScoreMode, limit: int) -> int:
        identifiers = (
            await self.session.scalars(
                select(AIScoreSnapshotRecord.id).where(AIScoreSnapshotRecord.as_of < before, AIScoreSnapshotRecord.mode == mode.value).order_by(AIScoreSnapshotRecord.as_of).limit(limit)
            )
        ).all()
        if not identifiers:
            return 0
        try:
            await self.session.execute(delete(AIScoreSnapshotRecord).where(AIScoreSnapshotRecord.id.in_(identifiers)))
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return len(identifiers)
