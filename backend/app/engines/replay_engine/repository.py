from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.storage.models import ReplayCheckpointRecord, ReplayOutputRecord, ReplaySessionRecord, ReplayTraceRecordModel, ReplayTransitionRecord

from .exceptions import ReplayCheckpointError, ReplayConcurrencyError, ReplayPersistenceError
from .models import (
    ReplayCheckpoint,
    ReplayOutputReference,
    ReplaySession,
    ReplayStatus,
    ReplayTraceRecord,
    ReplayTransition,
)


class ReplayRepository(ABC):
    @abstractmethod
    async def create_session(self, session: ReplaySession) -> ReplaySession: ...

    @abstractmethod
    async def get_session(self, replay_id: UUID) -> ReplaySession | None: ...

    @abstractmethod
    async def list_sessions(self, offset: int = 0, limit: int = 100, status: ReplayStatus | None = None) -> tuple[ReplaySession, ...]: ...

    @abstractmethod
    async def save_session(self, session: ReplaySession, expected_version: int) -> ReplaySession: ...

    @abstractmethod
    async def save_transition(self, transition: ReplayTransition) -> None: ...

    @abstractmethod
    async def list_transitions(self, replay_id: UUID, offset: int = 0, limit: int = 200) -> tuple[ReplayTransition, ...]: ...

    @abstractmethod
    async def save_checkpoint(self, checkpoint: ReplayCheckpoint) -> ReplayCheckpoint: ...

    @abstractmethod
    async def latest_checkpoint(self, replay_id: UUID) -> ReplayCheckpoint | None: ...

    @abstractmethod
    async def list_checkpoints(self, replay_id: UUID, offset: int = 0, limit: int = 100) -> tuple[ReplayCheckpoint, ...]: ...

    @abstractmethod
    async def save_trace(self, records: tuple[ReplayTraceRecord, ...]) -> None: ...

    @abstractmethod
    async def list_trace(self, replay_id: UUID, offset: int = 0, limit: int = 200) -> tuple[ReplayTraceRecord, ...]: ...

    @abstractmethod
    async def save_outputs(self, outputs: tuple[ReplayOutputReference, ...]) -> None: ...

    @abstractmethod
    async def list_outputs(self, replay_id: UUID, output_type: str | None = None, offset: int = 0, limit: int = 200) -> tuple[ReplayOutputReference, ...]: ...

    @abstractmethod
    async def acquire_lease(self, replay_id: UUID, worker_id: str, now: datetime, lease_seconds: int, expected_version: int) -> ReplaySession: ...

    @abstractmethod
    async def renew_lease(self, replay_id: UUID, worker_id: str, now: datetime, lease_seconds: int) -> ReplaySession: ...

    @abstractmethod
    async def release_lease(self, replay_id: UUID, worker_id: str) -> ReplaySession: ...

    @abstractmethod
    async def cleanup(self, before: datetime, limit: int) -> int: ...


class InMemoryReplayRepository(ReplayRepository):
    def __init__(self) -> None:
        self._sessions: dict[UUID, ReplaySession] = {}
        self._transitions: dict[UUID, list[ReplayTransition]] = {}
        self._checkpoints: dict[UUID, list[ReplayCheckpoint]] = {}
        self._trace: dict[UUID, list[ReplayTraceRecord]] = {}
        self._outputs: dict[UUID, list[ReplayOutputReference]] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, session: ReplaySession) -> ReplaySession:
        async with self._lock:
            if session.replay_id in self._sessions:
                raise ReplayConcurrencyError("replay session already exists")
            self._sessions[session.replay_id] = session
        return session

    async def get_session(self, replay_id: UUID) -> ReplaySession | None:
        async with self._lock:
            return self._sessions.get(replay_id)

    async def list_sessions(self, offset: int = 0, limit: int = 100, status: ReplayStatus | None = None) -> tuple[ReplaySession, ...]:
        async with self._lock:
            values = [item for item in self._sessions.values() if status is None or item.status == status]
        return tuple(sorted(values, key=lambda item: (item.created_at, str(item.replay_id)), reverse=True)[offset : offset + limit])

    async def save_session(self, session: ReplaySession, expected_version: int) -> ReplaySession:
        async with self._lock:
            current = self._sessions.get(session.replay_id)
            if current is None:
                raise ReplayPersistenceError("replay session does not exist")
            if current.row_version != expected_version:
                raise ReplayConcurrencyError("replay session version conflict")
            if session.row_version != expected_version + 1:
                raise ReplayConcurrencyError("replay session version must increment exactly once")
            self._sessions[session.replay_id] = session
        return session

    async def save_transition(self, transition: ReplayTransition) -> None:
        async with self._lock:
            values = self._transitions.setdefault(transition.replay_id, [])
            if any(item.transition_id == transition.transition_id for item in values):
                return
            values.append(transition)

    async def list_transitions(self, replay_id: UUID, offset: int = 0, limit: int = 200) -> tuple[ReplayTransition, ...]:
        async with self._lock:
            values = tuple(self._transitions.get(replay_id, ()))
        return tuple(sorted(values, key=lambda item: (item.occurred_at, str(item.transition_id)))[offset : offset + limit])

    async def save_checkpoint(self, checkpoint: ReplayCheckpoint) -> ReplayCheckpoint:
        if checkpoint.calculated_state_hash() != checkpoint.state_hash:
            raise ReplayCheckpointError("checkpoint state hash is invalid")
        async with self._lock:
            values = self._checkpoints.setdefault(checkpoint.replay_id, [])
            if any(item.checkpoint_id == checkpoint.checkpoint_id for item in values):
                return checkpoint
            if values and checkpoint.sequence != values[-1].sequence + 1:
                raise ReplayCheckpointError("checkpoint sequence must increase exactly once")
            values.append(checkpoint)
        return checkpoint

    async def latest_checkpoint(self, replay_id: UUID) -> ReplayCheckpoint | None:
        values = await self.list_checkpoints(replay_id)
        return values[-1] if values else None

    async def list_checkpoints(self, replay_id: UUID, offset: int = 0, limit: int = 100) -> tuple[ReplayCheckpoint, ...]:
        async with self._lock:
            values = tuple(self._checkpoints.get(replay_id, ()))
        return tuple(sorted(values, key=lambda item: item.sequence)[offset : offset + limit])

    async def save_trace(self, records: tuple[ReplayTraceRecord, ...]) -> None:
        async with self._lock:
            for record in records:
                values = self._trace.setdefault(record.replay_id, [])
                if not any(item.sequence == record.sequence for item in values):
                    values.append(record)

    async def list_trace(self, replay_id: UUID, offset: int = 0, limit: int = 200) -> tuple[ReplayTraceRecord, ...]:
        async with self._lock:
            values = tuple(self._trace.get(replay_id, ()))
        return tuple(sorted(values, key=lambda item: item.sequence)[offset : offset + limit])

    async def save_outputs(self, outputs: tuple[ReplayOutputReference, ...]) -> None:
        async with self._lock:
            for output in outputs:
                values = self._outputs.setdefault(output.replay_id, [])
                if not any(item.output_id == output.output_id for item in values):
                    values.append(output)

    async def list_outputs(self, replay_id: UUID, output_type: str | None = None, offset: int = 0, limit: int = 200) -> tuple[ReplayOutputReference, ...]:
        async with self._lock:
            values = [item for item in self._outputs.get(replay_id, ()) if output_type is None or item.output_type == output_type]
        return tuple(sorted(values, key=lambda item: (item.as_of, str(item.output_id)))[offset : offset + limit])

    async def acquire_lease(self, replay_id: UUID, worker_id: str, now: datetime, lease_seconds: int, expected_version: int) -> ReplaySession:
        async with self._lock:
            current = self._sessions.get(replay_id)
            if current is None:
                raise ReplayPersistenceError("replay session does not exist")
            if current.row_version != expected_version:
                raise ReplayConcurrencyError("replay session version conflict")
            if current.worker_id is not None and current.worker_id != worker_id and current.lease_expires_at is not None and current.lease_expires_at > now:
                raise ReplayConcurrencyError("replay session has an active worker lease")
            updated = current.model_copy(update={"worker_id": worker_id, "heartbeat_at": now, "lease_expires_at": now + timedelta(seconds=lease_seconds), "row_version": current.row_version + 1})
            self._sessions[replay_id] = updated
            return updated

    async def renew_lease(self, replay_id: UUID, worker_id: str, now: datetime, lease_seconds: int) -> ReplaySession:
        current = await self.get_session(replay_id)
        if current is None or current.worker_id != worker_id or current.lease_expires_at is None or current.lease_expires_at <= now:
            raise ReplayConcurrencyError("replay worker lease is lost")
        updated = current.model_copy(update={"heartbeat_at": now, "lease_expires_at": now + timedelta(seconds=lease_seconds), "row_version": current.row_version + 1})
        return await self.save_session(updated, current.row_version)

    async def release_lease(self, replay_id: UUID, worker_id: str) -> ReplaySession:
        current = await self.get_session(replay_id)
        if current is None or current.worker_id != worker_id:
            raise ReplayConcurrencyError("worker does not own replay lease")
        updated = current.model_copy(update={"worker_id": None, "heartbeat_at": None, "lease_expires_at": None, "row_version": current.row_version + 1})
        return await self.save_session(updated, current.row_version)

    async def cleanup(self, before: datetime, limit: int) -> int:
        terminal = {ReplayStatus.COMPLETED, ReplayStatus.CANCELLED, ReplayStatus.FAILED}
        async with self._lock:
            selected = [item.replay_id for item in sorted(self._sessions.values(), key=lambda item: item.created_at) if item.status in terminal and item.created_at < before][:limit]
            for replay_id in selected:
                self._sessions.pop(replay_id, None)
                self._transitions.pop(replay_id, None)
                self._checkpoints.pop(replay_id, None)
                self._trace.pop(replay_id, None)
                self._outputs.pop(replay_id, None)
        return len(selected)


class SqlAlchemyReplayRepository(ReplayRepository):
    """PostgreSQL adapter. Session payload is canonical; indexed columns coordinate workers."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _session(record: ReplaySessionRecord) -> ReplaySession:
        return ReplaySession.model_validate(record.payload)

    async def create_session(self, session: ReplaySession) -> ReplaySession:
        statement = insert(ReplaySessionRecord).values(self._session_values(session)).on_conflict_do_nothing(index_elements=["id"]).returning(ReplaySessionRecord.id)
        try:
            identifier = (await self.session.execute(statement)).scalar_one_or_none()
            if identifier is None:
                await self.session.rollback()
                raise ReplayConcurrencyError("replay session already exists")
            await self.session.commit()
            return session
        except ReplayConcurrencyError:
            raise
        except Exception as exc:
            await self.session.rollback()
            raise ReplayPersistenceError("replay session persistence failed") from exc

    async def get_session(self, replay_id: UUID) -> ReplaySession | None:
        record = await self.session.get(ReplaySessionRecord, replay_id)
        return self._session(record) if record is not None else None

    async def list_sessions(self, offset: int = 0, limit: int = 100, status: ReplayStatus | None = None) -> tuple[ReplaySession, ...]:
        statement = select(ReplaySessionRecord)
        if status is not None:
            statement = statement.where(ReplaySessionRecord.status == status.value)
        records = list((await self.session.scalars(statement.order_by(ReplaySessionRecord.created_at.desc(), ReplaySessionRecord.id).offset(offset).limit(limit))).all())
        return tuple(self._session(item) for item in records)

    async def save_session(self, session: ReplaySession, expected_version: int) -> ReplaySession:
        statement = update(ReplaySessionRecord).where(ReplaySessionRecord.id == session.replay_id, ReplaySessionRecord.row_version == expected_version).values(**self._session_values(session))
        try:
            result = await self.session.execute(statement)
            if int(getattr(result, "rowcount", 0)) != 1:
                await self.session.rollback()
                raise ReplayConcurrencyError("replay session version conflict")
            await self.session.commit()
            return session
        except ReplayConcurrencyError:
            raise
        except Exception as exc:
            await self.session.rollback()
            raise ReplayPersistenceError("replay session update failed") from exc

    async def save_transition(self, transition: ReplayTransition) -> None:
        statement = insert(ReplayTransitionRecord).values(id=transition.transition_id, replay_id=transition.replay_id, occurred_at=transition.occurred_at, from_status=transition.from_status.value, to_status=transition.to_status.value, reason_code=transition.reason_code, payload=transition.model_dump(mode="json")).on_conflict_do_nothing(index_elements=["id"])
        try:
            await self.session.execute(statement)
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    async def list_transitions(self, replay_id: UUID, offset: int = 0, limit: int = 200) -> tuple[ReplayTransition, ...]:
        records = list((await self.session.scalars(select(ReplayTransitionRecord).where(ReplayTransitionRecord.replay_id == replay_id).order_by(ReplayTransitionRecord.occurred_at, ReplayTransitionRecord.id).offset(offset).limit(limit))).all())
        return tuple(ReplayTransition.model_validate(item.payload) for item in records)

    async def save_checkpoint(self, checkpoint: ReplayCheckpoint) -> ReplayCheckpoint:
        if checkpoint.calculated_state_hash() != checkpoint.state_hash:
            raise ReplayCheckpointError("checkpoint state hash is invalid")
        statement = insert(ReplayCheckpointRecord).values(id=checkpoint.checkpoint_id, replay_id=checkpoint.replay_id, sequence=checkpoint.sequence, cursor_at=checkpoint.cursor_at, state_hash=checkpoint.state_hash, created_at=checkpoint.created_at, payload=checkpoint.model_dump(mode="json")).on_conflict_do_nothing(index_elements=["replay_id", "sequence"]).returning(ReplayCheckpointRecord.id)
        try:
            identifier = (await self.session.execute(statement)).scalar_one_or_none()
            if identifier is None:
                await self.session.rollback()
                raise ReplayCheckpointError("checkpoint sequence conflict")
            await self.session.commit()
            return checkpoint
        except ReplayCheckpointError:
            raise
        except Exception as exc:
            await self.session.rollback()
            raise ReplayPersistenceError("checkpoint persistence failed") from exc

    async def latest_checkpoint(self, replay_id: UUID) -> ReplayCheckpoint | None:
        records = await self.list_checkpoints(replay_id, 0, 1, descending=True)
        return records[0] if records else None

    async def list_checkpoints(self, replay_id: UUID, offset: int = 0, limit: int = 100, *, descending: bool = False) -> tuple[ReplayCheckpoint, ...]:
        order = ReplayCheckpointRecord.sequence.desc() if descending else ReplayCheckpointRecord.sequence
        records = list((await self.session.scalars(select(ReplayCheckpointRecord).where(ReplayCheckpointRecord.replay_id == replay_id).order_by(order).offset(offset).limit(limit))).all())
        return tuple(ReplayCheckpoint.model_validate(item.payload) for item in records)

    async def save_trace(self, records: tuple[ReplayTraceRecord, ...]) -> None:
        if not records:
            return
        values = [{"replay_id": item.replay_id, "sequence": item.sequence, "virtual_time": item.virtual_time, "event_id": item.event_id, "event_type": item.event_type, "payload": item.model_dump(mode="json")} for item in records]
        try:
            await self.session.execute(insert(ReplayTraceRecordModel).values(values).on_conflict_do_nothing(index_elements=["replay_id", "sequence"]))
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    async def list_trace(self, replay_id: UUID, offset: int = 0, limit: int = 200) -> tuple[ReplayTraceRecord, ...]:
        records = list((await self.session.scalars(select(ReplayTraceRecordModel).where(ReplayTraceRecordModel.replay_id == replay_id).order_by(ReplayTraceRecordModel.sequence).offset(offset).limit(limit))).all())
        return tuple(ReplayTraceRecord.model_validate(item.payload) for item in records)

    async def save_outputs(self, outputs: tuple[ReplayOutputReference, ...]) -> None:
        if not outputs:
            return
        values = [{"id": item.output_id, "replay_id": item.replay_id, "output_type": item.output_type, "source_engine": item.source_engine, "as_of": item.as_of, "fingerprint": item.fingerprint, "payload": item.model_dump(mode="json")} for item in outputs]
        try:
            await self.session.execute(insert(ReplayOutputRecord).values(values).on_conflict_do_nothing(index_elements=["id"]))
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    async def list_outputs(self, replay_id: UUID, output_type: str | None = None, offset: int = 0, limit: int = 200) -> tuple[ReplayOutputReference, ...]:
        statement = select(ReplayOutputRecord).where(ReplayOutputRecord.replay_id == replay_id)
        if output_type is not None:
            statement = statement.where(ReplayOutputRecord.output_type == output_type)
        records = list((await self.session.scalars(statement.order_by(ReplayOutputRecord.as_of, ReplayOutputRecord.id).offset(offset).limit(limit))).all())
        return tuple(ReplayOutputReference.model_validate(item.payload) for item in records)

    async def acquire_lease(self, replay_id: UUID, worker_id: str, now: datetime, lease_seconds: int, expected_version: int) -> ReplaySession:
        current = await self.get_session(replay_id)
        if current is None:
            raise ReplayPersistenceError("replay session does not exist")
        if current.worker_id and current.worker_id != worker_id and current.lease_expires_at and current.lease_expires_at > now:
            raise ReplayConcurrencyError("replay session has an active worker lease")
        updated = current.model_copy(update={"worker_id": worker_id, "heartbeat_at": now, "lease_expires_at": now + timedelta(seconds=lease_seconds), "row_version": current.row_version + 1})
        return await self.save_session(updated, expected_version)

    async def renew_lease(self, replay_id: UUID, worker_id: str, now: datetime, lease_seconds: int) -> ReplaySession:
        current = await self.get_session(replay_id)
        if current is None or current.worker_id != worker_id or current.lease_expires_at is None or current.lease_expires_at <= now:
            raise ReplayConcurrencyError("replay worker lease is lost")
        updated = current.model_copy(update={"heartbeat_at": now, "lease_expires_at": now + timedelta(seconds=lease_seconds), "row_version": current.row_version + 1})
        return await self.save_session(updated, current.row_version)

    async def release_lease(self, replay_id: UUID, worker_id: str) -> ReplaySession:
        current = await self.get_session(replay_id)
        if current is None or current.worker_id != worker_id:
            raise ReplayConcurrencyError("worker does not own replay lease")
        updated = current.model_copy(update={"worker_id": None, "heartbeat_at": None, "lease_expires_at": None, "row_version": current.row_version + 1})
        return await self.save_session(updated, current.row_version)

    async def cleanup(self, before: datetime, limit: int) -> int:
        ids = list((await self.session.scalars(select(ReplaySessionRecord.id).where(ReplaySessionRecord.status.in_([ReplayStatus.COMPLETED.value, ReplayStatus.CANCELLED.value, ReplayStatus.FAILED.value]), ReplaySessionRecord.created_at < before).order_by(ReplaySessionRecord.created_at).limit(limit))).all())
        if ids:
            try:
                await self.session.execute(delete(ReplaySessionRecord).where(ReplaySessionRecord.id.in_(ids)))
                await self.session.commit()
            except Exception:
                await self.session.rollback()
                raise
        return len(ids)

    @staticmethod
    def _session_values(session: ReplaySession) -> dict[str, object]:
        return {
            "id": session.replay_id,
            "request_fingerprint": session.request_fingerprint,
            "status": session.status.value,
            "mode": session.request.mode.value,
            "created_at": session.created_at,
            "virtual_cursor_at": session.virtual_cursor_at,
            "dataset_id": session.request.dataset.dataset_id,
            "dataset_version": session.request.dataset.dataset_version,
            "processed_events": session.processed_events,
            "generated_events": session.generated_events,
            "progress_percent": float(session.progress_percent) if session.progress_percent is not None else None,
            "semantic_output_hash": session.semantic_output_hash,
            "worker_id": session.worker_id,
            "lease_expires_at": session.lease_expires_at,
            "heartbeat_at": session.heartbeat_at,
            "row_version": session.row_version,
            "payload": session.model_dump(mode="json"),
        }
