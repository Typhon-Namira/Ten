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
from backend.app.storage.models import InstitutionalFlowCheckpointRecord, InstitutionalFlowEvidenceRecord, InstitutionalFlowSnapshotRecord
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import InstitutionalFlowAnalysisSnapshot, stable_id


class InstitutionalFlowRepository(ABC):
    @abstractmethod
    async def save(self, snapshot: InstitutionalFlowAnalysisSnapshot) -> None: ...

    @abstractmethod
    async def latest(self, symbol: str, timeframe: Timeframe) -> InstitutionalFlowAnalysisSnapshot | None: ...

    @abstractmethod
    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> InstitutionalFlowAnalysisSnapshot | None: ...

    @abstractmethod
    async def checkpoints(self) -> tuple[InstitutionalFlowAnalysisSnapshot, ...]: ...


class InMemoryInstitutionalFlowRepository(InstitutionalFlowRepository):
    def __init__(self) -> None:
        self._items: dict[tuple[str, Timeframe], list[InstitutionalFlowAnalysisSnapshot]] = {}
        self._boundaries: dict[tuple[str, ...], InstitutionalFlowAnalysisSnapshot] = {}
        self._lock = asyncio.Lock()

    async def save(self, snapshot: InstitutionalFlowAnalysisSnapshot) -> None:
        async with self._lock:
            boundary = analytical_snapshot_boundary(snapshot, include_processing_mode=True)
            existing = self._boundaries.get(boundary)
            if existing is not None:
                ensure_analytical_determinism(existing, snapshot, entity_type="institutional_flow_snapshot", include_processing_mode=True)
                return
            items = self._items.setdefault((canonical_symbol(snapshot.symbol), snapshot.timeframe), [])
            items.append(snapshot)
            items.sort(key=lambda item: item.analysis_timestamp)
            self._boundaries[boundary] = snapshot

    async def latest(self, symbol: str, timeframe: Timeframe) -> InstitutionalFlowAnalysisSnapshot | None:
        async with self._lock:
            items = self._items.get((canonical_symbol(symbol), timeframe), [])
            return items[-1] if items else None

    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> InstitutionalFlowAnalysisSnapshot | None:
        async with self._lock:
            values = [item for item in self._items.get((canonical_symbol(symbol), timeframe), []) if item.analysis_timestamp <= timestamp]
            return values[-1] if values else None

    async def checkpoints(self) -> tuple[InstitutionalFlowAnalysisSnapshot, ...]:
        async with self._lock:
            return tuple(items[-1] for items in self._items.values() if items)


class SqlAlchemyInstitutionalFlowRepository(InstitutionalFlowRepository, ScopedSessionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        ScopedSessionRepository.__init__(self, session_factory)

    @scoped_session
    async def save(self, snapshot: InstitutionalFlowAnalysisSnapshot) -> None:
        payload = snapshot.model_dump(mode="json")
        try:
            await self._save(snapshot, payload)
        except Exception:
            await self.session.rollback()
            raise

    async def _save(self, snapshot: InstitutionalFlowAnalysisSnapshot, payload: dict[str, object]) -> None:
        insert_result = await self.session.execute(
            insert(InstitutionalFlowSnapshotRecord)
            .values(
                id=snapshot.id,
                symbol=canonical_symbol(snapshot.symbol),
                timeframe=snapshot.timeframe.value,
                analysis_timestamp=snapshot.analysis_timestamp,
                processing_mode=snapshot.processing_mode.value,
                status=snapshot.status.value,
                configuration_version=snapshot.configuration_version,
                engine_version=snapshot.engine_version,
                payload=payload,
                created_at=snapshot.created_at,
            )
            .on_conflict_do_nothing(
                index_elements=[
                    InstitutionalFlowSnapshotRecord.symbol,
                    InstitutionalFlowSnapshotRecord.timeframe,
                    InstitutionalFlowSnapshotRecord.analysis_timestamp,
                    InstitutionalFlowSnapshotRecord.configuration_version,
                    InstitutionalFlowSnapshotRecord.processing_mode,
                ]
            )
            .returning(InstitutionalFlowSnapshotRecord.id)
        )
        inserted_id = returned_identity(insert_result, snapshot.id)
        persisted_snapshot = snapshot
        if inserted_id is None:
            persisted_record = (
                await self.session.scalars(
                    select(InstitutionalFlowSnapshotRecord).where(
                        InstitutionalFlowSnapshotRecord.symbol == canonical_symbol(snapshot.symbol),
                        InstitutionalFlowSnapshotRecord.timeframe == snapshot.timeframe.value,
                        InstitutionalFlowSnapshotRecord.analysis_timestamp == snapshot.analysis_timestamp,
                        InstitutionalFlowSnapshotRecord.configuration_version == snapshot.configuration_version,
                        InstitutionalFlowSnapshotRecord.processing_mode == snapshot.processing_mode.value,
                    )
                )
            ).first()
            if persisted_record is None:
                raise RuntimeError("institutional flow snapshot insert did not resolve a canonical row")
            persisted_snapshot = InstitutionalFlowAnalysisSnapshot.model_validate(persisted_record.payload)
        ensure_analytical_determinism(
            persisted_snapshot,
            snapshot,
            entity_type="institutional_flow_snapshot",
            include_processing_mode=True,
        )
        evidence_rows = [
            dict(
                id=stable_id("evidence-record", item.id),
                evidence_id=item.id,
                source_engine=item.source_engine.value,
                evidence_type=item.evidence_type.value,
                symbol=canonical_symbol(snapshot.symbol),
                timeframe=item.timeframe.value,
                availability_timestamp=item.availability_timestamp,
                direction=item.direction.value,
                confidence=item.confidence,
                quality=item.quality,
                payload=item.model_dump(mode="json"),
                created_at=snapshot.created_at,
            )
            for item in snapshot.evidence.accepted
        ]
        if evidence_rows:
            for chunk in bounded_insert_chunks(evidence_rows):
                await self.session.execute(insert(InstitutionalFlowEvidenceRecord).values(list(chunk)).on_conflict_do_nothing(index_elements=["id"]))
        statement = insert(InstitutionalFlowCheckpointRecord).values(
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
                set_={name: getattr(statement.excluded, name) for name in ("engine_version", "snapshot_id", "last_processed_candle", "state_hash", "state_payload", "updated_at")},
            )
        )
        await self.session.commit()

    @scoped_session
    async def latest(self, symbol: str, timeframe: Timeframe) -> InstitutionalFlowAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, None)

    @scoped_session
    async def at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> InstitutionalFlowAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, timestamp)

    async def _query(self, symbol: str, timeframe: Timeframe, timestamp: datetime | None) -> InstitutionalFlowAnalysisSnapshot | None:
        query = select(InstitutionalFlowSnapshotRecord).where(
            InstitutionalFlowSnapshotRecord.symbol == canonical_symbol(symbol), InstitutionalFlowSnapshotRecord.timeframe == timeframe.value
        )
        if timestamp is not None:
            query = query.where(InstitutionalFlowSnapshotRecord.analysis_timestamp <= timestamp)
        record = (await self.session.scalars(query.order_by(InstitutionalFlowSnapshotRecord.analysis_timestamp.desc()).limit(1))).first()
        return InstitutionalFlowAnalysisSnapshot.model_validate(record.payload) if record else None

    @scoped_session
    async def checkpoints(self) -> tuple[InstitutionalFlowAnalysisSnapshot, ...]:
        records = list((await self.session.scalars(select(InstitutionalFlowCheckpointRecord))).all())
        result = []
        for record in records:
            try:
                snapshot = InstitutionalFlowAnalysisSnapshot.model_validate(record.state_payload)
                if sha256(snapshot.model_dump_json().encode()).hexdigest() == record.state_hash and snapshot.engine_version == record.engine_version:
                    result.append(snapshot)
            except (ValueError, TypeError):
                continue
        return tuple(result)
