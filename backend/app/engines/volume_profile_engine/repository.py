import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from hashlib import sha256

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.market_data_engine.models import canonical_symbol
from backend.app.storage.models import VolumeProfileCheckpointRecord, VolumeProfileObjectRecord, VolumeProfileSnapshotRecord

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
        self._ids: set[object] = set()
        self._lock = asyncio.Lock()

    async def save(self, snapshot: VolumeProfileAnalysisSnapshot) -> None:
        async with self._lock:
            if snapshot.id in self._ids:
                return
            items = self._items.setdefault((canonical_symbol(snapshot.symbol), snapshot.timeframe), [])
            items.append(snapshot)
            items.sort(key=lambda x: x.analysis_timestamp)
            self._ids.add(snapshot.id)

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


class SqlAlchemyVolumeProfileRepository(VolumeProfileRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, snapshot: VolumeProfileAnalysisSnapshot) -> None:
        payload = snapshot.model_dump(mode="json")
        try:
            await self._save(snapshot, payload)
        except Exception:
            await self.session.rollback()
            raise

    async def _save(self, snapshot: VolumeProfileAnalysisSnapshot, payload: dict) -> None:
        await self.session.execute(
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
            .on_conflict_do_nothing(index_elements=["id"])
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
            await self.session.execute(insert(VolumeProfileObjectRecord).values(objects).on_conflict_do_nothing(index_elements=["id"]))
        digest = sha256(snapshot.model_dump_json().encode()).hexdigest()
        statement = insert(VolumeProfileCheckpointRecord).values(
            symbol=canonical_symbol(snapshot.symbol),
            timeframe=snapshot.timeframe.value,
            configuration_version=snapshot.configuration_version,
            engine_version=snapshot.engine_version,
            snapshot_id=snapshot.id,
            last_processed_candle=snapshot.analysis_timestamp,
            state_hash=digest,
            state_payload=payload,
            updated_at=snapshot.created_at,
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

    async def latest(self, symbol: str, timeframe: Timeframe) -> VolumeProfileAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, None)

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
