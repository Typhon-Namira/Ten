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
from backend.app.storage.models import LiquidityCheckpointRecord, LiquidityObjectRecord, LiquiditySnapshotRecord
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import LiquidityAnalysisSnapshot, stable_id


class LiquidityRepository(ABC):
    @abstractmethod
    async def save(self, snapshot: LiquidityAnalysisSnapshot) -> None: ...
    @abstractmethod
    async def latest(self, symbol: str, timeframe: Timeframe) -> LiquidityAnalysisSnapshot | None: ...
    @abstractmethod
    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> LiquidityAnalysisSnapshot | None: ...
    @abstractmethod
    async def checkpoints(self) -> tuple[LiquidityAnalysisSnapshot, ...]: ...


class InMemoryLiquidityRepository(LiquidityRepository):
    def __init__(self) -> None:
        self._items: dict[tuple[str, Timeframe], list[LiquidityAnalysisSnapshot]] = {}
        self._boundaries: dict[tuple[str, ...], LiquidityAnalysisSnapshot] = {}
        self._lock = asyncio.Lock()

    async def save(self, snapshot: LiquidityAnalysisSnapshot) -> None:
        async with self._lock:
            boundary = analytical_snapshot_boundary(snapshot, include_processing_mode=True)
            existing = self._boundaries.get(boundary)
            if existing is not None:
                ensure_analytical_determinism(existing, snapshot, entity_type="liquidity_snapshot", include_processing_mode=True)
                return
            items = self._items.setdefault((canonical_symbol(snapshot.symbol), snapshot.timeframe), [])
            items.append(snapshot)
            items.sort(key=lambda x: x.analysis_timestamp)
            self._boundaries[boundary] = snapshot

    async def latest(self, symbol: str, timeframe: Timeframe) -> LiquidityAnalysisSnapshot | None:
        async with self._lock:
            items = self._items.get((canonical_symbol(symbol), timeframe), [])
            return items[-1] if items else None

    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> LiquidityAnalysisSnapshot | None:
        async with self._lock:
            eligible = [x for x in self._items.get((canonical_symbol(symbol), timeframe), []) if x.analysis_timestamp <= timestamp]
            return eligible[-1] if eligible else None

    async def checkpoints(self) -> tuple[LiquidityAnalysisSnapshot, ...]:
        async with self._lock:
            return tuple(x[-1] for x in self._items.values() if x)


class SqlAlchemyLiquidityRepository(LiquidityRepository, ScopedSessionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        ScopedSessionRepository.__init__(self, session_factory)

    @scoped_session
    async def save(self, snapshot: LiquidityAnalysisSnapshot) -> None:
        payload = snapshot.model_dump(mode="json")
        try:
            await self._save(snapshot, payload)
        except Exception:
            await self.session.rollback()
            raise

    async def _save(self, snapshot: LiquidityAnalysisSnapshot, payload: dict[str, object]) -> None:
        insert_result = await self.session.execute(
            insert(LiquiditySnapshotRecord)
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
                    LiquiditySnapshotRecord.symbol,
                    LiquiditySnapshotRecord.timeframe,
                    LiquiditySnapshotRecord.analysis_timestamp,
                    LiquiditySnapshotRecord.configuration_version,
                    LiquiditySnapshotRecord.processing_mode,
                ]
            )
            .returning(LiquiditySnapshotRecord.id)
        )
        inserted_id = returned_identity(insert_result, snapshot.id)
        persisted_snapshot = snapshot
        if inserted_id is None:
            persisted_record = (
                await self.session.scalars(
                    select(LiquiditySnapshotRecord).where(
                        LiquiditySnapshotRecord.symbol == canonical_symbol(snapshot.symbol),
                        LiquiditySnapshotRecord.timeframe == snapshot.timeframe.value,
                        LiquiditySnapshotRecord.analysis_timestamp == snapshot.analysis_timestamp,
                        LiquiditySnapshotRecord.configuration_version == snapshot.configuration_version,
                        LiquiditySnapshotRecord.processing_mode == snapshot.processing_mode.value,
                    )
                )
            ).first()
            if persisted_record is None:
                raise RuntimeError("liquidity snapshot insert did not resolve a canonical row")
            persisted_snapshot = LiquidityAnalysisSnapshot.model_validate(persisted_record.payload)
        ensure_analytical_determinism(
            persisted_snapshot,
            snapshot,
            entity_type="liquidity_snapshot",
            include_processing_mode=True,
        )
        objects = []
        for kind, values in (
            ("level", snapshot.levels),
            ("pool", snapshot.pools),
            ("event", snapshot.events),
            ("session", snapshot.sessions),
            ("reference", snapshot.reference_levels),
            ("confluence", snapshot.confluences),
            ("target", snapshot.targets),
        ):
            for item in values:
                logical = item.id
                version = int(getattr(item, "version", 1))
                state = str(getattr(getattr(item, "lifecycle_state", None), "value", "active"))
                available = getattr(item, "available_at", snapshot.analysis_timestamp)
                objects.append(
                    dict(
                        id=stable_id("liquidity-object-version", snapshot.symbol, snapshot.timeframe, logical, version, state),
                        logical_id=logical,
                        object_type=kind,
                        symbol=canonical_symbol(snapshot.symbol),
                        timeframe=snapshot.timeframe.value,
                        availability_timestamp=available,
                        lifecycle_state=state,
                        confidence_score=float(getattr(item, "confidence_score", 100)),
                        quality_score=float(getattr(item, "quality_score", 100)),
                        configuration_version=snapshot.configuration_version,
                        engine_version=snapshot.engine_version,
                        payload=item.model_dump(mode="json"),
                        created_at=snapshot.created_at,
                    )
                )
        if objects:
            for chunk in bounded_insert_chunks(objects):
                await self.session.execute(insert(LiquidityObjectRecord).values(list(chunk)).on_conflict_do_nothing(index_elements=["id"]))
        statement = insert(LiquidityCheckpointRecord).values(
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
                    x: getattr(statement.excluded, x)
                    for x in ("engine_version", "snapshot_id", "last_processed_candle", "state_hash", "state_payload", "updated_at")
                },
            )
        )
        await self.session.commit()

    @scoped_session
    async def latest(self, symbol: str, timeframe: Timeframe) -> LiquidityAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, None)

    @scoped_session
    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> LiquidityAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, timestamp)

    async def _query(self, symbol: str, timeframe: Timeframe, timestamp: datetime | None) -> LiquidityAnalysisSnapshot | None:
        query = select(LiquiditySnapshotRecord).where(
            LiquiditySnapshotRecord.symbol == canonical_symbol(symbol), LiquiditySnapshotRecord.timeframe == timeframe.value
        )
        if timestamp is not None:
            query = query.where(LiquiditySnapshotRecord.analysis_timestamp <= timestamp)
        record = (await self.session.scalars(query.order_by(LiquiditySnapshotRecord.analysis_timestamp.desc()).limit(1))).first()
        return LiquidityAnalysisSnapshot.model_validate(record.payload) if record else None

    @scoped_session
    async def checkpoints(self) -> tuple[LiquidityAnalysisSnapshot, ...]:
        records = list((await self.session.scalars(select(LiquidityCheckpointRecord))).all())
        result = []
        for record in records:
            try:
                snapshot = LiquidityAnalysisSnapshot.model_validate(record.state_payload)
                if sha256(snapshot.model_dump_json().encode()).hexdigest() == record.state_hash and snapshot.engine_version == record.engine_version:
                    result.append(snapshot)
            except (ValueError, TypeError):
                continue
        return tuple(result)
