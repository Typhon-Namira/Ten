import asyncio
from abc import ABC, abstractmethod
from hashlib import sha256

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.market_data_engine.models import canonical_symbol
from backend.app.storage.models import MarketRegimeCheckpointRecord, MarketRegimeEvidenceRecord, MarketRegimeSnapshotRecord, MarketRegimeTransitionRecord

from .models import MarketRegimeEvidence, MarketRegimeSnapshot, RegimeTransition, stable_id


class MarketRegimeRepository(ABC):
    @abstractmethod
    async def save_snapshot(self, snapshot: MarketRegimeSnapshot) -> None: ...
    @abstractmethod
    async def get_latest_snapshot(self, symbol: str, timeframe: Timeframe) -> MarketRegimeSnapshot | None: ...
    @abstractmethod
    async def get_snapshot(self, snapshot_id: object) -> MarketRegimeSnapshot | None: ...
    @abstractmethod
    async def list_snapshots(self, symbol: str, timeframe: Timeframe, offset: int = 0, limit: int = 100) -> tuple[MarketRegimeSnapshot, ...]: ...
    @abstractmethod
    async def save_evidence(self, snapshot: MarketRegimeSnapshot) -> None: ...
    @abstractmethod
    async def list_evidence(self, symbol: str, timeframe: Timeframe, offset: int = 0, limit: int = 100) -> tuple[MarketRegimeEvidence, ...]: ...
    @abstractmethod
    async def save_transition(self, transition: RegimeTransition) -> None: ...
    @abstractmethod
    async def get_transition(self, transition_id: object) -> RegimeTransition | None: ...
    @abstractmethod
    async def list_transitions(self, symbol: str, timeframe: Timeframe, offset: int = 0, limit: int = 100) -> tuple[RegimeTransition, ...]: ...
    @abstractmethod
    async def save_checkpoint(self, snapshot: MarketRegimeSnapshot) -> None: ...
    @abstractmethod
    async def load_checkpoint(self, symbol: str, timeframe: Timeframe) -> MarketRegimeSnapshot | None: ...
    @abstractmethod
    async def checkpoints(self) -> tuple[MarketRegimeSnapshot, ...]: ...
    @abstractmethod
    async def prune_history(self, symbol: str, timeframe: Timeframe, keep: int) -> int: ...


class InMemoryMarketRegimeRepository(MarketRegimeRepository):
    def __init__(self) -> None:
        self._snapshots: dict[object, MarketRegimeSnapshot] = {}
        self._series: dict[tuple[str, Timeframe], list[object]] = {}
        self._evidence: dict[object, MarketRegimeEvidence] = {}
        self._transitions: dict[object, RegimeTransition] = {}
        self._checkpoints: dict[tuple[str, Timeframe], tuple[MarketRegimeSnapshot, str]] = {}
        self._lock = asyncio.Lock()

    async def save_snapshot(self, snapshot: MarketRegimeSnapshot) -> None:
        async with self._lock:
            if snapshot.snapshot_id in self._snapshots:
                return
            self._snapshots[snapshot.snapshot_id] = snapshot
            key = (canonical_symbol(snapshot.symbol), snapshot.timeframe)
            self._series.setdefault(key, []).append(snapshot.snapshot_id)
            self._series[key].sort(key=lambda identifier: self._snapshots[identifier].analysis_timestamp)

    async def get_latest_snapshot(self, symbol: str, timeframe: Timeframe) -> MarketRegimeSnapshot | None:
        async with self._lock:
            values = self._series.get((canonical_symbol(symbol), timeframe), [])
            return self._snapshots[values[-1]] if values else None

    async def get_snapshot(self, snapshot_id: object) -> MarketRegimeSnapshot | None:
        async with self._lock:
            return self._snapshots.get(snapshot_id)

    async def list_snapshots(self, symbol: str, timeframe: Timeframe, offset: int = 0, limit: int = 100) -> tuple[MarketRegimeSnapshot, ...]:
        async with self._lock:
            values = self._series.get((canonical_symbol(symbol), timeframe), [])
            return tuple(self._snapshots[item] for item in values[::-1][offset : offset + limit])

    async def save_evidence(self, snapshot: MarketRegimeSnapshot) -> None:
        async with self._lock:
            self._evidence.update((item.evidence_id, item) for item in snapshot.evidence)

    async def list_evidence(self, symbol: str, timeframe: Timeframe, offset: int = 0, limit: int = 100) -> tuple[MarketRegimeEvidence, ...]:
        async with self._lock:
            values = sorted(
                (item for item in self._evidence.values() if canonical_symbol(item.symbol) == canonical_symbol(symbol) and item.timeframe == timeframe),
                key=lambda item: (item.available_at, str(item.evidence_id)),
                reverse=True,
            )
            return tuple(values[offset : offset + limit])

    async def save_transition(self, transition: RegimeTransition) -> None:
        async with self._lock:
            self._transitions.setdefault(transition.transition_id, transition)

    async def get_transition(self, transition_id: object) -> RegimeTransition | None:
        async with self._lock:
            return self._transitions.get(transition_id)

    async def list_transitions(self, symbol: str, timeframe: Timeframe, offset: int = 0, limit: int = 100) -> tuple[RegimeTransition, ...]:
        async with self._lock:
            values = sorted(
                (item for item in self._transitions.values() if canonical_symbol(item.symbol) == canonical_symbol(symbol) and item.timeframe == timeframe),
                key=lambda item: item.started_at,
                reverse=True,
            )
            return tuple(values[offset : offset + limit])

    async def save_checkpoint(self, snapshot: MarketRegimeSnapshot) -> None:
        payload = snapshot.model_dump_json()
        async with self._lock:
            self._checkpoints[(canonical_symbol(snapshot.symbol), snapshot.timeframe)] = (snapshot, sha256(payload.encode()).hexdigest())

    async def load_checkpoint(self, symbol: str, timeframe: Timeframe) -> MarketRegimeSnapshot | None:
        async with self._lock:
            value = self._checkpoints.get((canonical_symbol(symbol), timeframe))
            if not value:
                return None
            snapshot, digest = value
            if sha256(snapshot.model_dump_json().encode()).hexdigest() != digest:
                raise ValueError("checkpoint payload integrity validation failed")
            return snapshot

    async def checkpoints(self) -> tuple[MarketRegimeSnapshot, ...]:
        async with self._lock:
            return tuple(value[0] for value in self._checkpoints.values())

    async def prune_history(self, symbol: str, timeframe: Timeframe, keep: int) -> int:
        async with self._lock:
            key = (canonical_symbol(symbol), timeframe)
            values = self._series.get(key, [])
            removed = values[:-keep] if keep else values[:]
            self._series[key] = values[-keep:] if keep else []
            for identifier in removed:
                self._snapshots.pop(identifier, None)
            return len(removed)


class SqlAlchemyMarketRegimeRepository(MarketRegimeRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save_snapshot(self, snapshot: MarketRegimeSnapshot) -> None:
        try:
            await self.session.execute(
                insert(MarketRegimeSnapshotRecord)
                .values(
                    id=snapshot.snapshot_id,
                    symbol=canonical_symbol(snapshot.symbol),
                    timeframe=snapshot.timeframe.value,
                    analysis_timestamp=snapshot.analysis_timestamp,
                    dominant_regime=snapshot.dominant_regime.value,
                    configuration_version=snapshot.configuration_version,
                    engine_version=snapshot.engine_version,
                    payload=snapshot.model_dump(mode="json"),
                    created_at=snapshot.created_at,
                )
                .on_conflict_do_nothing(index_elements=["id"])
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    async def get_latest_snapshot(self, symbol: str, timeframe: Timeframe) -> MarketRegimeSnapshot | None:
        query = (
            select(MarketRegimeSnapshotRecord)
            .where(MarketRegimeSnapshotRecord.symbol == canonical_symbol(symbol), MarketRegimeSnapshotRecord.timeframe == timeframe.value)
            .order_by(MarketRegimeSnapshotRecord.analysis_timestamp.desc())
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        return MarketRegimeSnapshot.model_validate(record.payload) if record else None

    async def get_snapshot(self, snapshot_id: object) -> MarketRegimeSnapshot | None:
        record = await self.session.get(MarketRegimeSnapshotRecord, snapshot_id)
        return MarketRegimeSnapshot.model_validate(record.payload) if record else None

    async def list_snapshots(self, symbol: str, timeframe: Timeframe, offset: int = 0, limit: int = 100) -> tuple[MarketRegimeSnapshot, ...]:
        query = (
            select(MarketRegimeSnapshotRecord)
            .where(MarketRegimeSnapshotRecord.symbol == canonical_symbol(symbol), MarketRegimeSnapshotRecord.timeframe == timeframe.value)
            .order_by(MarketRegimeSnapshotRecord.analysis_timestamp.desc())
            .offset(offset)
            .limit(limit)
        )
        return tuple(MarketRegimeSnapshot.model_validate(item.payload) for item in (await self.session.scalars(query)).all())

    async def save_evidence(self, snapshot: MarketRegimeSnapshot) -> None:
        rows = [
            dict(
                id=stable_id("record", item.evidence_id),
                evidence_id=item.evidence_id,
                snapshot_id=snapshot.snapshot_id,
                symbol=canonical_symbol(item.symbol),
                timeframe=item.timeframe.value,
                source_engine=item.source_engine,
                family=item.family.value,
                available_at=item.available_at,
                accepted=item.accepted,
                payload=item.model_dump(mode="json"),
                created_at=snapshot.created_at,
            )
            for item in snapshot.evidence
        ]
        if rows:
            try:
                await self.session.execute(insert(MarketRegimeEvidenceRecord).values(rows).on_conflict_do_nothing(index_elements=["id"]))
                await self.session.commit()
            except Exception:
                await self.session.rollback()
                raise

    async def list_evidence(self, symbol: str, timeframe: Timeframe, offset: int = 0, limit: int = 100) -> tuple[MarketRegimeEvidence, ...]:
        query = (
            select(MarketRegimeEvidenceRecord)
            .where(MarketRegimeEvidenceRecord.symbol == canonical_symbol(symbol), MarketRegimeEvidenceRecord.timeframe == timeframe.value)
            .order_by(MarketRegimeEvidenceRecord.available_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return tuple(MarketRegimeEvidence.model_validate(item.payload) for item in (await self.session.scalars(query)).all())

    async def save_transition(self, transition: RegimeTransition) -> None:
        try:
            await self.session.execute(
                insert(MarketRegimeTransitionRecord)
                .values(
                    id=transition.transition_id,
                    symbol=canonical_symbol(transition.symbol),
                    timeframe=transition.timeframe.value,
                    from_regime=transition.from_regime.value,
                    to_regime=transition.to_regime.value,
                    state=transition.state.value,
                    started_at=transition.started_at,
                    confirmed_at=transition.confirmed_at,
                    payload=transition.model_dump(mode="json"),
                )
                .on_conflict_do_nothing(index_elements=["id"])
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    async def get_transition(self, transition_id: object) -> RegimeTransition | None:
        record = await self.session.get(MarketRegimeTransitionRecord, transition_id)
        return RegimeTransition.model_validate(record.payload) if record else None

    async def list_transitions(self, symbol: str, timeframe: Timeframe, offset: int = 0, limit: int = 100) -> tuple[RegimeTransition, ...]:
        query = (
            select(MarketRegimeTransitionRecord)
            .where(MarketRegimeTransitionRecord.symbol == canonical_symbol(symbol), MarketRegimeTransitionRecord.timeframe == timeframe.value)
            .order_by(MarketRegimeTransitionRecord.started_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return tuple(RegimeTransition.model_validate(item.payload) for item in (await self.session.scalars(query)).all())

    async def save_checkpoint(self, snapshot: MarketRegimeSnapshot) -> None:
        payload = snapshot.model_dump(mode="json")
        digest = sha256(snapshot.model_dump_json().encode()).hexdigest()
        statement = insert(MarketRegimeCheckpointRecord).values(
            symbol=canonical_symbol(snapshot.symbol),
            timeframe=snapshot.timeframe.value,
            engine_name=snapshot.engine_name,
            engine_version=snapshot.engine_version,
            schema_version=snapshot.schema_version,
            configuration_version=snapshot.configuration_version,
            algorithm_version=snapshot.algorithm_version,
            snapshot_id=snapshot.snapshot_id,
            analysis_boundary=snapshot.analysis_timestamp,
            payload_hash=digest,
            state_payload=payload,
            created_at=snapshot.created_at,
        )
        try:
            await self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=["symbol", "timeframe", "configuration_version"],
                    set_={
                        name: getattr(statement.excluded, name)
                        for name in (
                            "engine_version",
                            "schema_version",
                            "algorithm_version",
                            "snapshot_id",
                            "analysis_boundary",
                            "payload_hash",
                            "state_payload",
                            "created_at",
                        )
                    },
                )
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    async def load_checkpoint(self, symbol: str, timeframe: Timeframe) -> MarketRegimeSnapshot | None:
        query = (
            select(MarketRegimeCheckpointRecord)
            .where(MarketRegimeCheckpointRecord.symbol == canonical_symbol(symbol), MarketRegimeCheckpointRecord.timeframe == timeframe.value)
            .order_by(MarketRegimeCheckpointRecord.analysis_boundary.desc())
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        if not record:
            return None
        snapshot = MarketRegimeSnapshot.model_validate(record.state_payload)
        if sha256(snapshot.model_dump_json().encode()).hexdigest() != record.payload_hash:
            raise ValueError("checkpoint payload integrity validation failed")
        return snapshot

    async def checkpoints(self) -> tuple[MarketRegimeSnapshot, ...]:
        records = (await self.session.scalars(select(MarketRegimeCheckpointRecord))).all()
        result = []
        for record in records:
            try:
                snapshot = MarketRegimeSnapshot.model_validate(record.state_payload)
                if sha256(snapshot.model_dump_json().encode()).hexdigest() == record.payload_hash:
                    result.append(snapshot)
            except (ValueError, TypeError):
                continue
        return tuple(result)

    async def prune_history(self, symbol: str, timeframe: Timeframe, keep: int) -> int:
        query = (
            select(MarketRegimeSnapshotRecord.id)
            .where(MarketRegimeSnapshotRecord.symbol == canonical_symbol(symbol), MarketRegimeSnapshotRecord.timeframe == timeframe.value)
            .order_by(MarketRegimeSnapshotRecord.analysis_timestamp.desc())
            .offset(keep)
        )
        ids = tuple((await self.session.scalars(query)).all())
        if ids:
            try:
                await self.session.execute(delete(MarketRegimeSnapshotRecord).where(MarketRegimeSnapshotRecord.id.in_(ids)))
                await self.session.commit()
            except Exception:
                await self.session.rollback()
                raise
        return len(ids)
