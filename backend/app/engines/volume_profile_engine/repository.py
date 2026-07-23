import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from hashlib import sha256

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.market_data_engine.models import canonical_symbol
from backend.app.storage.batching import bounded_insert_chunks
from backend.app.storage.logical_identity import analytical_snapshot_boundary, ensure_analytical_determinism, returned_identity
from backend.app.storage.models import VolumeProfileCheckpointRecord, VolumeProfileObjectRecord, VolumeProfileSnapshotRecord
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import VolumeProfileAnalysisSnapshot, stable_id


class VolumeProfileRepository(ABC):
    @abstractmethod
    async def save(self, snapshot: VolumeProfileAnalysisSnapshot) -> None: ...
    @abstractmethod
    async def latest(self, symbol: str, timeframe: Timeframe) -> VolumeProfileAnalysisSnapshot | None: ...
    @abstractmethod
    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> VolumeProfileAnalysisSnapshot | None: ...
    @abstractmethod
    async def checkpoints(self) -> tuple[VolumeProfileAnalysisSnapshot, ...]: ...


class InMemoryVolumeProfileRepository(VolumeProfileRepository):
    def __init__(self) -> None:
        self._items: dict[tuple[str, Timeframe], list[VolumeProfileAnalysisSnapshot]] = {}
        self._boundaries: dict[tuple[str, ...], VolumeProfileAnalysisSnapshot] = {}
        self._lock = asyncio.Lock()

    async def save(self, snapshot: VolumeProfileAnalysisSnapshot) -> None:
        async with self._lock:
            boundary = analytical_snapshot_boundary(snapshot, include_processing_mode=True)
            existing = self._boundaries.get(boundary)
            if existing is not None:
                ensure_analytical_determinism(existing, snapshot, entity_type="volume_profile_snapshot", include_processing_mode=True)
                return
            items = self._items.setdefault((canonical_symbol(snapshot.symbol), snapshot.timeframe), [])
            items.append(snapshot)
            items.sort(key=lambda x: x.analysis_timestamp)
            self._boundaries[boundary] = snapshot

    async def latest(self, symbol: str, timeframe: Timeframe) -> VolumeProfileAnalysisSnapshot | None:
        async with self._lock:
            items = self._items.get((canonical_symbol(symbol), timeframe), [])
            return items[-1] if items else None

    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> VolumeProfileAnalysisSnapshot | None:
        async with self._lock:
            values = [x for x in self._items.get((canonical_symbol(symbol), timeframe), []) if x.analysis_timestamp <= timestamp]
            return values[-1] if values else None

    async def checkpoints(self) -> tuple[VolumeProfileAnalysisSnapshot, ...]:
        async with self._lock:
            return tuple(x[-1] for x in self._items.values() if x)


class SqlAlchemyVolumeProfileRepository(VolumeProfileRepository, ScopedSessionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        ScopedSessionRepository.__init__(self, session_factory)

    @scoped_session
    async def save(self, snapshot: VolumeProfileAnalysisSnapshot) -> None:
        payload = snapshot.model_dump(mode="json")
        try:
            await self._save(snapshot, payload)
        except Exception:
            await self.session.rollback()
            raise

    async def _save(self, snapshot: VolumeProfileAnalysisSnapshot, payload: dict) -> None:
        insert_result = await self.session.execute(
            insert(VolumeProfileSnapshotRecord)
            .values(
                id=snapshot.id,
                symbol=canonical_symbol(snapshot.symbol),
                timeframe=snapshot.timeframe.value,
                analysis_timestamp=snapshot.analysis_timestamp,
                processing_mode=snapshot.processing_mode.value,
                configuration_version=snapshot.configuration_version,
                engine_version=snapshot.engine_version,
                payload=payload,
                created_at=snapshot.created_at,
            )
            .on_conflict_do_nothing(
                index_elements=[
                    VolumeProfileSnapshotRecord.symbol,
                    VolumeProfileSnapshotRecord.timeframe,
                    VolumeProfileSnapshotRecord.analysis_timestamp,
                    VolumeProfileSnapshotRecord.configuration_version,
                    VolumeProfileSnapshotRecord.processing_mode,
                ]
            )
            .returning(VolumeProfileSnapshotRecord.id)
        )
        inserted_id = returned_identity(insert_result, snapshot.id)
        persisted_snapshot = snapshot
        if inserted_id is None:
            persisted_record = (
                await self.session.scalars(
                    select(VolumeProfileSnapshotRecord).where(
                        VolumeProfileSnapshotRecord.symbol == canonical_symbol(snapshot.symbol),
                        VolumeProfileSnapshotRecord.timeframe == snapshot.timeframe.value,
                        VolumeProfileSnapshotRecord.analysis_timestamp == snapshot.analysis_timestamp,
                        VolumeProfileSnapshotRecord.configuration_version == snapshot.configuration_version,
                        VolumeProfileSnapshotRecord.processing_mode == snapshot.processing_mode.value,
                    )
                )
            ).first()
            if persisted_record is None:
                raise RuntimeError("volume profile snapshot insert did not resolve a canonical row")
            persisted_snapshot = VolumeProfileAnalysisSnapshot.model_validate(persisted_record.payload)
        ensure_analytical_determinism(
            persisted_snapshot,
            snapshot,
            entity_type="volume_profile_snapshot",
            include_processing_mode=True,
        )
        objects = []
        for profile in snapshot.profiles:
            objects.append(
                dict(
                    id=stable_id("profile-object", profile.id),
                    logical_id=profile.logical_id,
                    object_type=profile.profile_type.value,
                    symbol=canonical_symbol(profile.symbol),
                    timeframe=profile.timeframe.value,
                    availability_timestamp=profile.availability_timestamp,
                    lifecycle_state=profile.lifecycle_state.value,
                    confidence_score=profile.confidence_score,
                    quality_score=profile.quality_score,
                    configuration_version=profile.configuration_version,
                    engine_version=profile.engine_version,
                    payload=profile.model_dump(mode="json"),
                    created_at=snapshot.created_at,
                )
            )
        if objects:
            for chunk in bounded_insert_chunks(objects):
                await self.session.execute(insert(VolumeProfileObjectRecord).values(list(chunk)).on_conflict_do_nothing(index_elements=["id"]))
        statement = insert(VolumeProfileCheckpointRecord).values(
            symbol=canonical_symbol(snapshot.symbol),
            timeframe=snapshot.timeframe.value,
            configuration_version=snapshot.configuration_version,
            engine_version=snapshot.engine_version,
            snapshot_id=persisted_snapshot.id,
            last_processed_candle=persisted_snapshot.analysis_timestamp,
            state_hash=sha256(persisted_snapshot.model_dump_json().encode()).hexdigest(),
            state_payload=persisted_snapshot.model_dump(mode="json"),
            updated_at=persisted_snapshot.created_at,
        )
        await self.session.execute(
            statement.on_conflict_do_update(
                index_elements=["symbol", "timeframe", "configuration_version"],
                set_={
                    name: getattr(statement.excluded, name)
                    for name in ("engine_version", "snapshot_id", "last_processed_candle", "state_hash", "state_payload", "updated_at")
                },
            )
        )
        await self.session.commit()

    @scoped_session
    async def latest(self, symbol: str, timeframe: Timeframe) -> VolumeProfileAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, None)

    @scoped_session
    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> VolumeProfileAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, timestamp)

    async def _query(self, symbol: str, timeframe: Timeframe, timestamp: datetime | None) -> VolumeProfileAnalysisSnapshot | None:
        query = select(VolumeProfileSnapshotRecord).where(
            VolumeProfileSnapshotRecord.symbol == canonical_symbol(symbol), VolumeProfileSnapshotRecord.timeframe == timeframe.value
        )
        if timestamp is not None:
            query = query.where(VolumeProfileSnapshotRecord.analysis_timestamp <= timestamp)
        record = (await self.session.scalars(query.order_by(VolumeProfileSnapshotRecord.analysis_timestamp.desc()).limit(1))).first()
        return VolumeProfileAnalysisSnapshot.model_validate(record.payload) if record else None

    @scoped_session
    async def checkpoints(self) -> tuple[VolumeProfileAnalysisSnapshot, ...]:
        records = list((await self.session.scalars(select(VolumeProfileCheckpointRecord))).all())
        result = []
        for record in records:
            try:
                snapshot = VolumeProfileAnalysisSnapshot.model_validate(record.state_payload)
                if sha256(snapshot.model_dump_json().encode()).hexdigest() == record.state_hash and snapshot.engine_version == record.engine_version:
                    result.append(snapshot)
            except (ValueError, TypeError):
                continue
        return tuple(result)
