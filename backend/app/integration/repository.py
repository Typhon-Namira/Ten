from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Protocol
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.storage.models import IntegrationDataQualityIssueRecord, IntegrationEventRecord, IntegrationEventTraceRecord, IntegrationOutboxRecord, IntegrationProcessedEventRecord, IntegrationSnapshotRecord, OperationalSignalRecord

from .models import CanonicalEventEnvelope, DataQualityIssue, IntegrationSnapshot, IntegrationTraceRecord, OperationalSignal, OutboxItem, semantic_uuid


class IntegrationRepository(Protocol):
    async def enqueue(self, envelope: CanonicalEventEnvelope) -> OutboxItem: ...
    async def pending(self, now: datetime, limit: int) -> tuple[OutboxItem, ...]: ...
    async def complete(self, outbox_id: UUID, now: datetime) -> None: ...
    async def fail(self, outbox_id: UUID, code: str) -> None: ...
    async def processed(self, event_id: str) -> bool: ...
    async def mark_processed(self, event_id: str) -> None: ...
    async def save_snapshot(self, value: IntegrationSnapshot) -> IntegrationSnapshot: ...
    async def save_signal(self, value: OperationalSignal) -> OperationalSignal: ...
    async def latest_signal(self, instrument: str | None = None, timeframe: str | None = None) -> OperationalSignal | None: ...
    async def signals(self, limit: int = 100) -> tuple[OperationalSignal, ...]: ...
    async def trace(self, trace_id: UUID) -> tuple[IntegrationTraceRecord, ...]: ...
    async def save_trace(self, value: IntegrationTraceRecord) -> None: ...
    async def save_issue(self, value: DataQualityIssue) -> None: ...
    def metrics(self) -> dict[str, int]: ...


class InMemoryIntegrationRepository:
    """Bounded adapter used for development and tests; production uses the durable engine stores."""

    def __init__(self, limit: int = 10_000) -> None:
        self._limit = limit
        self._events: dict[str, CanonicalEventEnvelope] = {}
        self._outbox: dict[UUID, OutboxItem] = {}
        self._processed: set[str] = set()
        self._snapshots: dict[UUID, IntegrationSnapshot] = {}
        self._signals: dict[UUID, OperationalSignal] = {}
        self._signal_hashes: dict[str, UUID] = {}
        self._traces: dict[UUID, list[IntegrationTraceRecord]] = defaultdict(list)
        self._issues: list[DataQualityIssue] = []
        self._lock = asyncio.Lock()

    async def enqueue(self, envelope: CanonicalEventEnvelope) -> OutboxItem:
        async with self._lock:
            existing = next((item for item in self._outbox.values() if item.envelope.event_id == envelope.event_id), None)
            if existing is not None:
                return existing
            item = OutboxItem(outbox_id=semantic_uuid("outbox", envelope.event_id), envelope=envelope, available_at=envelope.available_at)
            self._events[envelope.event_id] = envelope
            self._outbox[item.outbox_id] = item
            self._trim()
            return item

    async def pending(self, now: datetime, limit: int) -> tuple[OutboxItem, ...]:
        async with self._lock:
            values = [item for item in self._outbox.values() if item.published_at is None and item.available_at <= now]
        return tuple(sorted(values, key=lambda item: (item.available_at, str(item.outbox_id)))[:limit])

    async def complete(self, outbox_id: UUID, now: datetime) -> None:
        async with self._lock:
            item = self._outbox[outbox_id]
            self._outbox[outbox_id] = item.model_copy(update={"published_at": now})

    async def fail(self, outbox_id: UUID, code: str) -> None:
        async with self._lock:
            item = self._outbox[outbox_id]
            self._outbox[outbox_id] = item.model_copy(update={"attempts": item.attempts + 1, "last_error_code": code})

    async def processed(self, event_id: str) -> bool:
        async with self._lock:
            return event_id in self._processed

    async def mark_processed(self, event_id: str) -> None:
        async with self._lock:
            self._processed.add(event_id)

    async def save_snapshot(self, value: IntegrationSnapshot) -> IntegrationSnapshot:
        async with self._lock:
            self._snapshots[value.snapshot_id] = value
        return value

    async def save_signal(self, value: OperationalSignal) -> OperationalSignal:
        async with self._lock:
            existing = self._signal_hashes.get(value.semantic_hash)
            if existing is not None:
                return self._signals[existing]
            self._signals[value.operational_signal_id] = value
            self._signal_hashes[value.semantic_hash] = value.operational_signal_id
        return value

    async def latest_signal(self, instrument: str | None = None, timeframe: str | None = None) -> OperationalSignal | None:
        async with self._lock:
            values = [item for item in self._signals.values() if (instrument is None or item.instrument == instrument) and (timeframe is None or item.timeframe == timeframe)]
        return max(values, key=lambda item: (item.effective_at, str(item.operational_signal_id)), default=None)

    async def signals(self, limit: int = 100) -> tuple[OperationalSignal, ...]:
        async with self._lock:
            values = sorted(self._signals.values(), key=lambda item: (item.effective_at, str(item.operational_signal_id)), reverse=True)
        return tuple(values[:limit])

    async def trace(self, trace_id: UUID) -> tuple[IntegrationTraceRecord, ...]:
        async with self._lock:
            return tuple(self._traces.get(trace_id, ()))

    async def save_trace(self, value: IntegrationTraceRecord) -> None:
        async with self._lock:
            if all(item.trace_record_id != value.trace_record_id for item in self._traces[value.trace_id]):
                self._traces[value.trace_id].append(value)

    async def save_issue(self, value: DataQualityIssue) -> None:
        async with self._lock:
            self._issues.append(value)

    def metrics(self) -> dict[str, int]:
        return {"events": len(self._events), "outbox_backlog": sum(item.published_at is None for item in self._outbox.values()), "processed": len(self._processed), "snapshots": len(self._snapshots), "signals": len(self._signals), "quality_issues": len(self._issues)}

    def _trim(self) -> None:
        while len(self._events) > self._limit:
            self._events.pop(next(iter(self._events)))


class SqlAlchemyIntegrationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._metrics = {"events": 0, "outbox_backlog": 0, "processed": 0, "snapshots": 0, "signals": 0, "quality_issues": 0}

    async def enqueue(self, envelope: CanonicalEventEnvelope) -> OutboxItem:
        outbox_id = semantic_uuid("outbox", envelope.event_id)
        try:
            await self.session.execute(insert(IntegrationEventRecord).values(event_id=envelope.event_id, event_type=envelope.event_type, trace_id=envelope.trace_id, correlation_id=envelope.correlation_id, mode=envelope.mode.value, instrument=envelope.instrument_id, timeframe=envelope.timeframe, occurred_at=envelope.occurred_at, available_at=envelope.available_at, payload_hash=envelope.payload_hash, payload=envelope.model_dump(mode="json")).on_conflict_do_nothing(index_elements=["event_id"]))
            await self.session.execute(insert(IntegrationOutboxRecord).values(outbox_id=outbox_id, event_id=envelope.event_id, available_at=envelope.available_at, attempts=0).on_conflict_do_nothing(index_elements=["event_id"]))
            await self.session.commit()
            self._metrics["events"] += 1
            self._metrics["outbox_backlog"] += 1
        except Exception:
            await self.session.rollback()
            raise
        record = await self.session.get(IntegrationOutboxRecord, outbox_id)
        assert record is not None
        return OutboxItem(outbox_id=record.outbox_id, envelope=envelope, available_at=record.available_at, attempts=record.attempts, published_at=record.published_at, last_error_code=record.last_error_code)

    async def pending(self, now: datetime, limit: int) -> tuple[OutboxItem, ...]:
        query = select(IntegrationOutboxRecord, IntegrationEventRecord).join(IntegrationEventRecord, IntegrationEventRecord.event_id == IntegrationOutboxRecord.event_id).where(IntegrationOutboxRecord.published_at.is_(None), IntegrationOutboxRecord.available_at <= now).order_by(IntegrationOutboxRecord.available_at).limit(limit)
        rows = (await self.session.execute(query)).all()
        return tuple(OutboxItem(outbox_id=outbox.outbox_id, envelope=CanonicalEventEnvelope.model_validate(event.payload), available_at=outbox.available_at, attempts=outbox.attempts, published_at=outbox.published_at, last_error_code=outbox.last_error_code) for outbox, event in rows)

    async def complete(self, outbox_id: UUID, now: datetime) -> None:
        await self.session.execute(update(IntegrationOutboxRecord).where(IntegrationOutboxRecord.outbox_id == outbox_id).values(published_at=now))
        await self.session.commit()
        self._metrics["outbox_backlog"] = max(0, self._metrics["outbox_backlog"] - 1)

    async def fail(self, outbox_id: UUID, code: str) -> None:
        await self.session.execute(update(IntegrationOutboxRecord).where(IntegrationOutboxRecord.outbox_id == outbox_id).values(attempts=IntegrationOutboxRecord.attempts + 1, last_error_code=code))
        await self.session.commit()

    async def processed(self, event_id: str) -> bool:
        return await self.session.get(IntegrationProcessedEventRecord, event_id) is not None

    async def mark_processed(self, event_id: str) -> None:
        await self.session.execute(insert(IntegrationProcessedEventRecord).values(event_id=event_id, processed_at=datetime.now(UTC)).on_conflict_do_nothing(index_elements=["event_id"]))
        await self.session.commit()
        self._metrics["processed"] += 1

    async def save_snapshot(self, value: IntegrationSnapshot) -> IntegrationSnapshot:
        await self.session.execute(insert(IntegrationSnapshotRecord).values(snapshot_id=value.snapshot_id, semantic_hash=value.semantic_hash, trace_id=semantic_uuid("trace", value.market_event_id), mode=value.mode.value, instrument=value.instrument, timeframe=value.timeframe, analytical_boundary=value.analytical_boundary, status=value.status.value, payload=value.model_dump(mode="json")).on_conflict_do_nothing(index_elements=["semantic_hash"]))
        await self.session.commit()
        self._metrics["snapshots"] += 1
        return value

    async def save_signal(self, value: OperationalSignal) -> OperationalSignal:
        await self.session.execute(insert(OperationalSignalRecord).values(operational_signal_id=value.operational_signal_id, semantic_hash=value.semantic_hash, decision_id=value.decision_id, ai_score_id=value.ai_score_id, snapshot_id=value.snapshot_id, trace_id=value.trace_id, mode=value.mode.value, instrument=value.instrument, timeframe=value.timeframe, effective_at=value.effective_at, expires_at=value.expires_at, payload=value.model_dump(mode="json")).on_conflict_do_nothing(index_elements=["semantic_hash"]))
        await self.session.commit()
        self._metrics["signals"] += 1
        query = select(OperationalSignalRecord).where(OperationalSignalRecord.semantic_hash == value.semantic_hash)
        record = (await self.session.scalars(query)).first()
        return OperationalSignal.model_validate(record.payload) if record else value

    async def latest_signal(self, instrument: str | None = None, timeframe: str | None = None) -> OperationalSignal | None:
        query = select(OperationalSignalRecord)
        if instrument:
            query = query.where(OperationalSignalRecord.instrument == instrument)
        if timeframe:
            query = query.where(OperationalSignalRecord.timeframe == timeframe)
        record = (await self.session.scalars(query.order_by(OperationalSignalRecord.effective_at.desc()).limit(1))).first()
        return OperationalSignal.model_validate(record.payload) if record else None

    async def signals(self, limit: int = 100) -> tuple[OperationalSignal, ...]:
        records = (await self.session.scalars(select(OperationalSignalRecord).order_by(OperationalSignalRecord.effective_at.desc()).limit(limit))).all()
        return tuple(OperationalSignal.model_validate(item.payload) for item in records)

    async def trace(self, trace_id: UUID) -> tuple[IntegrationTraceRecord, ...]:
        records = (await self.session.scalars(select(IntegrationEventTraceRecord).where(IntegrationEventTraceRecord.trace_id == trace_id).order_by(IntegrationEventTraceRecord.started_at))).all()
        return tuple(IntegrationTraceRecord.model_validate(item.payload) for item in records)

    async def save_trace(self, value: IntegrationTraceRecord) -> None:
        await self.session.execute(insert(IntegrationEventTraceRecord).values(trace_record_id=value.trace_record_id, trace_id=value.trace_id, event_id=value.event_id, status=value.status.value, started_at=value.started_at, completed_at=value.completed_at, payload=value.model_dump(mode="json")).on_conflict_do_nothing(index_elements=["trace_record_id"]))
        await self.session.commit()

    async def save_issue(self, value: DataQualityIssue) -> None:
        await self.session.execute(insert(IntegrationDataQualityIssueRecord).values(issue_id=value.issue_id, event_id=value.event_id, provider=value.provider, status=value.status.value, observed_at=value.observed_at, payload=value.model_dump(mode="json")).on_conflict_do_nothing(index_elements=["issue_id"]))
        await self.session.commit()
        self._metrics["quality_issues"] += 1

    def metrics(self) -> dict[str, int]:
        return dict(self._metrics)
