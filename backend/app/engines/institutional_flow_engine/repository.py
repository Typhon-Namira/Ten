import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from hashlib import sha256

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.market_data_engine.models import canonical_symbol
from backend.app.storage.models import InstitutionalFlowCheckpointRecord, InstitutionalFlowEvidenceRecord, InstitutionalFlowSnapshotRecord

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
        self._ids: set[object] = set()
        self._lock = asyncio.Lock()

    async def save(self, snapshot: InstitutionalFlowAnalysisSnapshot) -> None:
        async with self._lock:
            if snapshot.id in self._ids:
                return
            items = self._items.setdefault((canonical_symbol(snapshot.symbol), snapshot.timeframe), [])
            items.append(snapshot)
            items.sort(key=lambda item: item.analysis_timestamp)
            self._ids.add(snapshot.id)

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


class SqlAlchemyInstitutionalFlowRepository(InstitutionalFlowRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, snapshot: InstitutionalFlowAnalysisSnapshot) -> None:
        payload = snapshot.model_dump(mode="json")
        await self.session.execute(
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
            .on_conflict_do_nothing(index_elements=["id"])
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
            await self.session.execute(insert(InstitutionalFlowEvidenceRecord).values(evidence_rows).on_conflict_do_nothing(index_elements=["id"]))
        digest = sha256(snapshot.model_dump_json().encode()).hexdigest()
        statement = insert(InstitutionalFlowCheckpointRecord).values(
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
                set_={name: getattr(statement.excluded, name) for name in ("engine_version", "snapshot_id", "last_processed_candle", "state_hash", "state_payload", "updated_at")},
            )
        )
        await self.session.commit()

    async def latest(self, symbol: str, timeframe: Timeframe) -> InstitutionalFlowAnalysisSnapshot | None:
        return await self._query(symbol, timeframe, None)

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
