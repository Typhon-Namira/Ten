from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from hashlib import sha256
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    EconomicCalendarCheckpointRecord,
    EconomicCalendarContextRecord,
    EconomicCalendarEventRecord,
    EconomicCalendarObservationRecord,
    EconomicCalendarRevisionRecord,
    EconomicCalendarSnapshotRecord,
    EconomicCalendarSyncStateRecord,
)
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import (
    EconomicCalendarCheckpoint,
    EconomicCalendarSnapshot,
    EconomicEvent,
    EconomicEventRevision,
    EconomicEventStatus,
    InstrumentEventContext,
    ProviderEventObservation,
)


class EconomicCalendarRepository(ABC):
    @abstractmethod
    async def save_provider_observations(self, values: tuple[ProviderEventObservation, ...]) -> int: ...
    @abstractmethod
    async def get_provider_observation(self, observation_id: UUID) -> ProviderEventObservation | None: ...
    @abstractmethod
    async def list_provider_observations(self, event_id: UUID | None = None, limit: int = 100) -> tuple[ProviderEventObservation, ...]: ...
    @abstractmethod
    async def save_event(self, event: EconomicEvent) -> None: ...
    @abstractmethod
    async def get_event(self, event_id: UUID) -> EconomicEvent | None: ...
    @abstractmethod
    async def list_events(
        self, start: datetime | None = None, end: datetime | None = None, as_of: datetime | None = None, limit: int = 500
    ) -> tuple[EconomicEvent, ...]: ...
    @abstractmethod
    async def get_event_at_boundary(self, event_id: UUID, boundary: datetime) -> EconomicEvent | None: ...
    @abstractmethod
    async def save_revision(self, revision: EconomicEventRevision) -> None: ...
    @abstractmethod
    async def list_revisions(self, event_id: UUID, limit: int = 500) -> tuple[EconomicEventRevision, ...]: ...
    @abstractmethod
    async def save_snapshot(self, snapshot: EconomicCalendarSnapshot) -> None: ...
    @abstractmethod
    async def get_snapshot(self, snapshot_id: UUID) -> EconomicCalendarSnapshot | None: ...
    @abstractmethod
    async def list_snapshots(self, limit: int = 100) -> tuple[EconomicCalendarSnapshot, ...]: ...
    @abstractmethod
    async def save_instrument_context(self, context: InstrumentEventContext) -> None: ...
    @abstractmethod
    async def get_instrument_context(self, symbol: str, boundary: datetime | None = None) -> InstrumentEventContext | None: ...
    @abstractmethod
    async def save_sync_state(self, provider: str, state: dict[str, Any]) -> None: ...
    @abstractmethod
    async def load_sync_state(self, provider: str) -> dict[str, Any] | None: ...
    @abstractmethod
    async def save_checkpoint(self, checkpoint: EconomicCalendarCheckpoint) -> None: ...
    @abstractmethod
    async def load_checkpoint(self) -> EconomicCalendarCheckpoint | None: ...
    @abstractmethod
    async def prune_history(self, keep_events: int, keep_observations: int, keep_snapshots: int) -> dict[str, int]: ...


class InMemoryEconomicCalendarRepository(EconomicCalendarRepository):
    def __init__(self) -> None:
        self._observations: dict[UUID, ProviderEventObservation] = {}
        self._events: dict[UUID, list[EconomicEvent]] = {}
        self._revisions: dict[UUID, list[EconomicEventRevision]] = {}
        self._snapshots: dict[UUID, EconomicCalendarSnapshot] = {}
        self._contexts: dict[str, list[InstrumentEventContext]] = {}
        self._sync: dict[str, dict[str, Any]] = {}
        self._checkpoint: EconomicCalendarCheckpoint | None = None
        self._lock = asyncio.Lock()

    async def save_provider_observations(self, values: tuple[ProviderEventObservation, ...]) -> int:
        async with self._lock:
            before = len(self._observations)
            for item in values:
                self._observations.setdefault(item.observation_id, item)
            return len(self._observations) - before

    async def get_provider_observation(self, observation_id: UUID) -> ProviderEventObservation | None:
        async with self._lock:
            return self._observations.get(observation_id)

    async def list_provider_observations(self, event_id: UUID | None = None, limit: int = 100) -> tuple[ProviderEventObservation, ...]:
        async with self._lock:
            ids = set(self._events[event_id][-1].provider_records) if event_id in self._events else None
            values = [item for item in self._observations.values() if ids is None or item.observation_id in ids]
            return tuple(sorted(values, key=lambda item: (item.available_at, str(item.observation_id)), reverse=True)[:limit])

    async def save_event(self, event: EconomicEvent) -> None:
        async with self._lock:
            history = self._events.setdefault(event.event_id, [])
            if not any(
                item.available_at == event.available_at and item.provider_records == event.provider_records and item.model_dump() == event.model_dump()
                for item in history
            ):
                history.append(event)
                history.sort(key=lambda item: (item.available_at, item.last_updated_at, str(item.provider_records)))

    async def get_event(self, event_id: UUID) -> EconomicEvent | None:
        async with self._lock:
            values = self._events.get(event_id, [])
            return values[-1] if values else None

    async def list_events(
        self, start: datetime | None = None, end: datetime | None = None, as_of: datetime | None = None, limit: int = 500
    ) -> tuple[EconomicEvent, ...]:
        async with self._lock:
            values = []
            for history in self._events.values():
                visible = [item for item in history if as_of is None or item.available_at <= as_of]
                if not visible:
                    continue
                item = visible[-1]
                if start and (item.scheduled_at_utc is None or item.scheduled_at_utc < start):
                    continue
                if end and (item.scheduled_at_utc is None or item.scheduled_at_utc > end):
                    continue
                values.append(item)
            return tuple(sorted(values, key=lambda item: (item.scheduled_at_utc or item.available_at, str(item.event_id)))[:limit])

    async def get_event_at_boundary(self, event_id: UUID, boundary: datetime) -> EconomicEvent | None:
        async with self._lock:
            values = [item for item in self._events.get(event_id, []) if item.available_at <= boundary]
            return values[-1] if values else None

    async def save_revision(self, revision: EconomicEventRevision) -> None:
        async with self._lock:
            values = self._revisions.setdefault(revision.event_id, [])
            if not any(item.revision_id == revision.revision_id for item in values):
                values.append(revision)
                values.sort(key=lambda item: item.revision_number)

    async def list_revisions(self, event_id: UUID, limit: int = 500) -> tuple[EconomicEventRevision, ...]:
        async with self._lock:
            return tuple(self._revisions.get(event_id, [])[-limit:])

    async def save_snapshot(self, snapshot: EconomicCalendarSnapshot) -> None:
        async with self._lock:
            self._snapshots.setdefault(snapshot.snapshot_id, snapshot)

    async def get_snapshot(self, snapshot_id: UUID) -> EconomicCalendarSnapshot | None:
        async with self._lock:
            return self._snapshots.get(snapshot_id)

    async def list_snapshots(self, limit: int = 100) -> tuple[EconomicCalendarSnapshot, ...]:
        async with self._lock:
            return tuple(sorted(self._snapshots.values(), key=lambda item: item.historical_boundary, reverse=True)[:limit])

    async def save_instrument_context(self, context: InstrumentEventContext) -> None:
        async with self._lock:
            values = self._contexts.setdefault(context.symbol, [])
            if not any(item.context_id == context.context_id for item in values):
                values.append(context)
                values.sort(key=lambda item: item.historical_boundary)

    async def get_instrument_context(self, symbol: str, boundary: datetime | None = None) -> InstrumentEventContext | None:
        async with self._lock:
            values = [item for item in self._contexts.get(symbol.upper(), []) if boundary is None or item.historical_boundary <= boundary]
            return values[-1] if values else None

    async def save_sync_state(self, provider: str, state: dict[str, Any]) -> None:
        async with self._lock:
            self._sync[provider] = dict(state)

    async def load_sync_state(self, provider: str) -> dict[str, Any] | None:
        async with self._lock:
            value = self._sync.get(provider)
            return dict(value) if value else None

    async def save_checkpoint(self, checkpoint: EconomicCalendarCheckpoint) -> None:
        if sha256(_checkpoint_bytes(checkpoint.state_payload)).hexdigest() != checkpoint.payload_hash:
            raise ValueError("checkpoint payload hash mismatch")
        async with self._lock:
            self._checkpoint = checkpoint

    async def load_checkpoint(self) -> EconomicCalendarCheckpoint | None:
        async with self._lock:
            checkpoint = self._checkpoint
        if checkpoint and sha256(_checkpoint_bytes(checkpoint.state_payload)).hexdigest() != checkpoint.payload_hash:
            raise ValueError("checkpoint payload integrity validation failed")
        return checkpoint

    async def prune_history(self, keep_events: int, keep_observations: int, keep_snapshots: int) -> dict[str, int]:
        async with self._lock:
            event_keys = sorted(self._events, key=lambda key: self._events[key][-1].available_at)
            removed_events = event_keys[:-keep_events] if keep_events else event_keys
            for key in removed_events:
                self._events.pop(key, None)
            observation_keys = sorted(self._observations, key=lambda key: self._observations[key].available_at)
            for key in observation_keys[:-keep_observations] if keep_observations else observation_keys:
                self._observations.pop(key, None)
            snapshot_keys = sorted(self._snapshots, key=lambda key: self._snapshots[key].historical_boundary)
            for key in snapshot_keys[:-keep_snapshots] if keep_snapshots else snapshot_keys:
                self._snapshots.pop(key, None)
            return {
                "events": len(removed_events),
                "observations": max(0, len(observation_keys) - keep_observations),
                "snapshots": max(0, len(snapshot_keys) - keep_snapshots),
            }


def _checkpoint_bytes(payload: dict[str, Any]) -> bytes:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


class SqlAlchemyEconomicCalendarRepository(EconomicCalendarRepository, ScopedSessionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        ScopedSessionRepository.__init__(self, session_factory)

    @scoped_session
    async def save_provider_observations(self, values: tuple[ProviderEventObservation, ...]) -> int:
        if not values:
            return 0
        rows = [
            dict(
                id=item.observation_id,
                provider_name=item.provider_name,
                provider_event_id=item.provider_event_id,
                available_at=item.available_at,
                ingested_at=item.ingested_at,
                payload_hash=item.payload_hash,
                payload=item.model_dump(mode="json"),
            )
            for item in values
        ]
        try:
            await self.session.execute(insert(EconomicCalendarObservationRecord).values(rows).on_conflict_do_nothing(index_elements=["id"]))
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return len(values)

    @scoped_session
    async def get_provider_observation(self, observation_id: UUID) -> ProviderEventObservation | None:
        record = await self.session.get(EconomicCalendarObservationRecord, observation_id)
        return ProviderEventObservation.model_validate(record.payload) if record else None

    @scoped_session
    async def list_provider_observations(self, event_id: UUID | None = None, limit: int = 100) -> tuple[ProviderEventObservation, ...]:
        query = select(EconomicCalendarObservationRecord).order_by(EconomicCalendarObservationRecord.available_at.desc()).limit(limit)
        values = tuple(ProviderEventObservation.model_validate(item.payload) for item in (await self.session.scalars(query)).all())
        if event_id is None:
            return values
        event = await self.get_event(event_id)
        ids = set(event.provider_records) if event else set()
        return tuple(item for item in values if item.observation_id in ids)

    @scoped_session
    async def save_event(self, event: EconomicEvent) -> None:
        statement = insert(EconomicCalendarEventRecord).values(
            id=event.event_id,
            canonical_name=event.canonical_name,
            scheduled_at=event.scheduled_at_utc,
            available_at=event.available_at,
            country_code=event.country_code,
            currency_codes=list(event.currency_codes),
            category=event.category.value,
            importance=event.importance.value,
            status=event.status.value,
            configuration_version=event.configuration_version,
            payload=event.model_dump(mode="json"),
        )
        try:
            await self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=["id"],
                    set_={name: getattr(statement.excluded, name) for name in ("scheduled_at", "available_at", "category", "importance", "status", "payload")},
                )
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    @scoped_session
    async def get_event(self, event_id: UUID) -> EconomicEvent | None:
        record = await self.session.get(EconomicCalendarEventRecord, event_id)
        return EconomicEvent.model_validate(record.payload) if record else None

    @scoped_session
    async def list_events(
        self, start: datetime | None = None, end: datetime | None = None, as_of: datetime | None = None, limit: int = 500
    ) -> tuple[EconomicEvent, ...]:
        query = select(EconomicCalendarEventRecord)
        if start:
            query = query.where(EconomicCalendarEventRecord.scheduled_at >= start)
        if end:
            query = query.where(EconomicCalendarEventRecord.scheduled_at <= end)
        if as_of:
            query = query.where(EconomicCalendarEventRecord.available_at <= as_of)
        query = query.order_by(EconomicCalendarEventRecord.scheduled_at, EconomicCalendarEventRecord.id).limit(limit)
        return tuple(EconomicEvent.model_validate(item.payload) for item in (await self.session.scalars(query)).all())

    @scoped_session
    async def get_event_at_boundary(self, event_id: UUID, boundary: datetime) -> EconomicEvent | None:
        revisions = await self.list_revisions(event_id)
        event = await self.get_event(event_id)
        visible = [item for item in revisions if item.available_at <= boundary]
        if event is None or not visible:
            return None
        state = event
        for revision in sorted((item for item in revisions if item.available_at > boundary), key=lambda item: item.revision_number, reverse=True):
            state = state.model_copy(
                update={
                    "status": revision.previous_status or state.status,
                    "scheduled_at": revision.previous_scheduled_at or state.scheduled_at,
                    "scheduled_at_utc": revision.previous_scheduled_at or state.scheduled_at_utc,
                    "actual_value": revision.previous_actual_value,
                    "forecast_value": revision.previous_forecast_value,
                    "previous_value": revision.previous_previous_value,
                    "revised_previous_value": revision.previous_revised_previous_value,
                    "is_cancelled": revision.previous_status == EconomicEventStatus.CANCELLED,
                    "is_postponed": revision.previous_status == EconomicEventStatus.POSTPONED,
                    "is_rescheduled": False,
                    "is_revised": False,
                    "is_corrected": False,
                }
            )
        latest = visible[-1]
        return state.model_copy(update={"available_at": latest.available_at, "revision_count": len(visible), "latest_revision_id": latest.revision_id})

    @scoped_session
    async def save_revision(self, revision: EconomicEventRevision) -> None:
        try:
            await self.session.execute(
                insert(EconomicCalendarRevisionRecord)
                .values(
                    id=revision.revision_id,
                    event_id=revision.event_id,
                    revision_number=revision.revision_number,
                    revision_type=revision.revision_type.value,
                    available_at=revision.available_at,
                    payload_hash=revision.payload_hash,
                    payload=revision.model_dump(mode="json"),
                )
                .on_conflict_do_nothing(index_elements=["id"])
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    @scoped_session
    async def list_revisions(self, event_id: UUID, limit: int = 500) -> tuple[EconomicEventRevision, ...]:
        query = (
            select(EconomicCalendarRevisionRecord)
            .where(EconomicCalendarRevisionRecord.event_id == event_id)
            .order_by(EconomicCalendarRevisionRecord.revision_number)
            .limit(limit)
        )
        return tuple(EconomicEventRevision.model_validate(item.payload) for item in (await self.session.scalars(query)).all())

    @scoped_session
    async def save_snapshot(self, snapshot: EconomicCalendarSnapshot) -> None:
        try:
            await self.session.execute(
                insert(EconomicCalendarSnapshotRecord)
                .values(
                    id=snapshot.snapshot_id,
                    analysis_timestamp=snapshot.analysis_timestamp,
                    historical_boundary=snapshot.historical_boundary,
                    configuration_version=snapshot.configuration_version,
                    payload=snapshot.model_dump(mode="json"),
                    created_at=snapshot.created_at,
                )
                .on_conflict_do_nothing(index_elements=["id"])
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    @scoped_session
    async def get_snapshot(self, snapshot_id: UUID) -> EconomicCalendarSnapshot | None:
        record = await self.session.get(EconomicCalendarSnapshotRecord, snapshot_id)
        return EconomicCalendarSnapshot.model_validate(record.payload) if record else None

    @scoped_session
    async def list_snapshots(self, limit: int = 100) -> tuple[EconomicCalendarSnapshot, ...]:
        query = select(EconomicCalendarSnapshotRecord).order_by(EconomicCalendarSnapshotRecord.historical_boundary.desc()).limit(limit)
        return tuple(EconomicCalendarSnapshot.model_validate(item.payload) for item in (await self.session.scalars(query)).all())

    @scoped_session
    async def save_instrument_context(self, context: InstrumentEventContext) -> None:
        try:
            await self.session.execute(
                insert(EconomicCalendarContextRecord)
                .values(id=context.context_id, symbol=context.symbol, historical_boundary=context.historical_boundary, payload=context.model_dump(mode="json"))
                .on_conflict_do_nothing(index_elements=["id"])
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    @scoped_session
    async def get_instrument_context(self, symbol: str, boundary: datetime | None = None) -> InstrumentEventContext | None:
        query = select(EconomicCalendarContextRecord).where(EconomicCalendarContextRecord.symbol == symbol.upper())
        if boundary:
            query = query.where(EconomicCalendarContextRecord.historical_boundary <= boundary)
        record = (await self.session.scalars(query.order_by(EconomicCalendarContextRecord.historical_boundary.desc()).limit(1))).first()
        return InstrumentEventContext.model_validate(record.payload) if record else None

    @scoped_session
    async def save_sync_state(self, provider: str, state: dict[str, Any]) -> None:
        statement = insert(EconomicCalendarSyncStateRecord).values(
            provider_name=provider, state=state, updated_at=datetime.fromisoformat(str(state["updated_at"]))
        )
        try:
            await self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=["provider_name"], set_={"state": statement.excluded.state, "updated_at": statement.excluded.updated_at}
                )
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    @scoped_session
    async def load_sync_state(self, provider: str) -> dict[str, Any] | None:
        record = await self.session.get(EconomicCalendarSyncStateRecord, provider)
        return dict(record.state) if record else None

    @scoped_session
    async def save_checkpoint(self, checkpoint: EconomicCalendarCheckpoint) -> None:
        if sha256(_checkpoint_bytes(checkpoint.state_payload)).hexdigest() != checkpoint.payload_hash:
            raise ValueError("checkpoint payload hash mismatch")
        statement = insert(EconomicCalendarCheckpointRecord).values(
            id=checkpoint.checkpoint_id,
            engine_name=checkpoint.engine_name,
            engine_version=checkpoint.engine_version,
            schema_version=checkpoint.schema_version,
            configuration_version=checkpoint.configuration_version,
            normalization_version=checkpoint.normalization_version,
            payload_hash=checkpoint.payload_hash,
            state_payload=checkpoint.state_payload,
            created_at=checkpoint.created_at,
        )
        try:
            await self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=["engine_name", "configuration_version"],
                    set_={
                        name: getattr(statement.excluded, name)
                        for name in ("engine_version", "schema_version", "normalization_version", "payload_hash", "state_payload", "created_at")
                    },
                )
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    @scoped_session
    async def load_checkpoint(self) -> EconomicCalendarCheckpoint | None:
        record = (
            await self.session.scalars(select(EconomicCalendarCheckpointRecord).order_by(EconomicCalendarCheckpointRecord.created_at.desc()).limit(1))
        ).first()
        if not record:
            return None
        payload = dict(record.state_payload)
        if sha256(_checkpoint_bytes(payload)).hexdigest() != record.payload_hash:
            raise ValueError("checkpoint payload integrity validation failed")
        return EconomicCalendarCheckpoint(
            checkpoint_id=record.id,
            engine_name=record.engine_name,
            engine_version=record.engine_version,
            schema_version=record.schema_version,
            configuration_version=record.configuration_version,
            normalization_version=record.normalization_version,
            last_successful_sync_at=datetime.fromisoformat(payload["last_successful_sync_at"]) if payload.get("last_successful_sync_at") else None,
            last_provider_cursor=payload.get("last_provider_cursor", {}),
            last_provider_update_token=payload.get("last_provider_update_token", {}),
            last_processed_observation=payload.get("last_processed_observation"),
            state_payload=payload,
            payload_hash=record.payload_hash,
            created_at=record.created_at,
        )

    @scoped_session
    async def prune_history(self, keep_events: int, keep_observations: int, keep_snapshots: int) -> dict[str, int]:
        removed: dict[str, int] = {}
        for name, model, order, keep in (
            ("events", EconomicCalendarEventRecord, EconomicCalendarEventRecord.available_at, keep_events),
            ("observations", EconomicCalendarObservationRecord, EconomicCalendarObservationRecord.available_at, keep_observations),
            ("snapshots", EconomicCalendarSnapshotRecord, EconomicCalendarSnapshotRecord.historical_boundary, keep_snapshots),
        ):
            ids = tuple((await self.session.scalars(select(model.id).order_by(order.desc()).offset(keep))).all())
            if ids:
                try:
                    await self.session.execute(delete(model).where(model.id.in_(ids)))
                except Exception:
                    await self.session.rollback()
                    raise
            removed[name] = len(ids)
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return removed
