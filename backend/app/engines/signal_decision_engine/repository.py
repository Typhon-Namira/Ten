from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.storage.models import SignalDecisionReasonRecord, SignalDecisionRecord, SignalDecisionRuleRecord

from .exceptions import SignalDecisionPersistenceError
from .models import DecisionDirection, DecisionMode, DecisionState, SignalDecision, stable_id


class SignalDecisionRepository(ABC):
    @abstractmethod
    async def save_decision(self, decision: SignalDecision) -> SignalDecision: ...

    @abstractmethod
    async def get_decision(self, decision_id: UUID) -> SignalDecision | None: ...

    @abstractmethod
    async def find_by_fingerprint(self, fingerprint: str, mode: DecisionMode) -> SignalDecision | None: ...

    @abstractmethod
    async def get_active_decision(self, instrument: str, timeframe: str, at: datetime, direction: DecisionDirection | None = None, state: DecisionState | None = None) -> SignalDecision | None: ...

    @abstractmethod
    async def list_decisions(
        self,
        instrument: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        direction: DecisionDirection | None = None,
        state: DecisionState | None = None,
        policy_version: str | None = None,
        ai_score_policy_version: str | None = None,
        mode: DecisionMode | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[SignalDecision, ...]: ...

    @abstractmethod
    async def find_recent_decisions(self, instrument: str, timeframe: str, at: datetime, limit: int = 20) -> tuple[SignalDecision, ...]: ...

    @abstractmethod
    async def prune(self, before: datetime, mode: DecisionMode, limit: int) -> int: ...


class InMemorySignalDecisionRepository(SignalDecisionRepository):
    def __init__(self) -> None:
        self._decisions: dict[UUID, SignalDecision] = {}
        self._fingerprints: dict[tuple[str, DecisionMode], UUID] = {}
        self._lock = asyncio.Lock()

    async def save_decision(self, decision: SignalDecision) -> SignalDecision:
        key = (decision.input_fingerprint, decision.mode)
        async with self._lock:
            existing = self._fingerprints.get(key)
            if existing is not None:
                return self._decisions[existing]
            self._decisions[decision.decision_id] = decision
            self._fingerprints[key] = decision.decision_id
            return decision

    async def get_decision(self, decision_id: UUID) -> SignalDecision | None:
        async with self._lock:
            return self._decisions.get(decision_id)

    async def find_by_fingerprint(self, fingerprint: str, mode: DecisionMode) -> SignalDecision | None:
        async with self._lock:
            identifier = self._fingerprints.get((fingerprint, mode))
            return self._decisions.get(identifier) if identifier else None

    async def get_active_decision(self, instrument: str, timeframe: str, at: datetime, direction: DecisionDirection | None = None, state: DecisionState | None = None) -> SignalDecision | None:
        values = await self.list_decisions(instrument, timeframe, end=at, direction=direction, state=state, limit=100)
        return next((item for item in values if item.valid_from <= at < item.valid_until), None)

    async def list_decisions(
        self,
        instrument: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        direction: DecisionDirection | None = None,
        state: DecisionState | None = None,
        policy_version: str | None = None,
        ai_score_policy_version: str | None = None,
        mode: DecisionMode | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[SignalDecision, ...]:
        async with self._lock:
            values = [
                item
                for item in self._decisions.values()
                if item.instrument == instrument
                and item.timeframe == timeframe
                and (start is None or item.as_of >= start)
                and (end is None or item.as_of <= end)
                and (direction is None or item.direction == direction)
                and (state is None or item.state == state)
                and (policy_version is None or item.decision_policy_version == policy_version)
                and (ai_score_policy_version is None or item.ai_score_policy_version == ai_score_policy_version)
                and (mode is None or item.mode == mode)
            ]
        values.sort(key=lambda item: (item.as_of, str(item.decision_id)), reverse=True)
        return tuple(values[offset : offset + limit])

    async def find_recent_decisions(self, instrument: str, timeframe: str, at: datetime, limit: int = 20) -> tuple[SignalDecision, ...]:
        return await self.list_decisions(instrument, timeframe, end=at, limit=limit)

    async def prune(self, before: datetime, mode: DecisionMode, limit: int) -> int:
        async with self._lock:
            expired = sorted((item for item in self._decisions.values() if item.mode == mode and item.valid_until < before), key=lambda item: item.valid_until)[:limit]
            for item in expired:
                self._decisions.pop(item.decision_id, None)
                self._fingerprints.pop((item.input_fingerprint, item.mode), None)
            return len(expired)


class SqlAlchemySignalDecisionRepository(SignalDecisionRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save_decision(self, decision: SignalDecision) -> SignalDecision:
        try:
            existing = await self.find_by_fingerprint(decision.input_fingerprint, decision.mode)
            if existing:
                return existing
            insert_result = await self.session.execute(
                insert(SignalDecisionRecord).values(
                    id=decision.decision_id,
                    decision_key=decision.decision_key,
                    input_fingerprint=decision.input_fingerprint,
                    instrument=decision.instrument,
                    timeframe=decision.timeframe,
                    direction=decision.direction.value,
                    state=decision.state.value,
                    status=decision.status.value,
                    mode=decision.mode.value,
                    as_of=decision.as_of,
                    decided_at=decision.decided_at,
                    valid_from=decision.valid_from,
                    valid_until=decision.valid_until,
                    ai_score_snapshot_id=decision.ai_score_snapshot_id,
                    decision_policy_version=decision.decision_policy_version,
                    eligibility_score=decision.eligibility_score,
                    payload=decision.model_dump(mode="json"),
                ).on_conflict_do_nothing(index_elements=["input_fingerprint", "mode"]).returning(SignalDecisionRecord.id)
            )
            if insert_result.scalar_one_or_none() is None:
                await self.session.rollback()
                concurrent = await self.find_by_fingerprint(decision.input_fingerprint, decision.mode)
                if concurrent:
                    return concurrent
                raise SignalDecisionPersistenceError("Signal Decision concurrent insert could not be resolved")
            rules = [
                {
                    "id": stable_id("decision-rule", decision.decision_id, index),
                    "decision_id": decision.decision_id,
                    "rule_id": item.rule_id,
                    "category": item.category.value,
                    "outcome": item.outcome.value,
                    "severity": item.severity.value,
                    "payload": item.model_dump(mode="json"),
                }
                for index, item in enumerate(decision.rules)
            ]
            reasons = [
                {
                    "id": stable_id("decision-reason", decision.decision_id, kind, index),
                    "decision_id": decision.decision_id,
                    "reason_type": kind,
                    "reason_code": item.reason_code,
                    "severity": item.severity.value,
                    "payload": item.model_dump(mode="json"),
                }
                for kind, values in (("blocker", decision.blockers), ("warning", decision.warnings), ("supporting", decision.supporting_reasons))
                for index, item in enumerate(values)
            ]
            if rules:
                await self.session.execute(insert(SignalDecisionRuleRecord).values(rules).on_conflict_do_nothing(index_elements=["id"]))
            if reasons:
                await self.session.execute(insert(SignalDecisionReasonRecord).values(reasons).on_conflict_do_nothing(index_elements=["id"]))
            await self.session.commit()
            return await self.find_by_fingerprint(decision.input_fingerprint, decision.mode) or decision
        except Exception as exc:
            await self.session.rollback()
            raise SignalDecisionPersistenceError("Signal Decision persistence failed") from exc

    async def get_decision(self, decision_id: UUID) -> SignalDecision | None:
        record = await self.session.get(SignalDecisionRecord, decision_id)
        return SignalDecision.model_validate(record.payload) if record else None

    async def find_by_fingerprint(self, fingerprint: str, mode: DecisionMode) -> SignalDecision | None:
        query = select(SignalDecisionRecord).where(SignalDecisionRecord.input_fingerprint == fingerprint, SignalDecisionRecord.mode == mode.value).limit(1)
        record = (await self.session.scalars(query)).first()
        return SignalDecision.model_validate(record.payload) if record else None

    async def get_active_decision(self, instrument: str, timeframe: str, at: datetime, direction: DecisionDirection | None = None, state: DecisionState | None = None) -> SignalDecision | None:
        query = select(SignalDecisionRecord).where(
            SignalDecisionRecord.instrument == instrument,
            SignalDecisionRecord.timeframe == timeframe,
            SignalDecisionRecord.valid_from <= at,
            SignalDecisionRecord.valid_until > at,
        )
        if direction:
            query = query.where(SignalDecisionRecord.direction == direction.value)
        if state:
            query = query.where(SignalDecisionRecord.state == state.value)
        record = (await self.session.scalars(query.order_by(SignalDecisionRecord.as_of.desc()).limit(1))).first()
        return SignalDecision.model_validate(record.payload) if record else None

    async def list_decisions(
        self,
        instrument: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        direction: DecisionDirection | None = None,
        state: DecisionState | None = None,
        policy_version: str | None = None,
        ai_score_policy_version: str | None = None,
        mode: DecisionMode | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[SignalDecision, ...]:
        query = select(SignalDecisionRecord).where(SignalDecisionRecord.instrument == instrument, SignalDecisionRecord.timeframe == timeframe)
        if start:
            query = query.where(SignalDecisionRecord.as_of >= start)
        if end:
            query = query.where(SignalDecisionRecord.as_of <= end)
        if direction:
            query = query.where(SignalDecisionRecord.direction == direction.value)
        if state:
            query = query.where(SignalDecisionRecord.state == state.value)
        if policy_version:
            query = query.where(SignalDecisionRecord.decision_policy_version == policy_version)
        if ai_score_policy_version:
            query = query.where(SignalDecisionRecord.payload["ai_score_policy_version"].astext == ai_score_policy_version)
        if mode:
            query = query.where(SignalDecisionRecord.mode == mode.value)
        records = (await self.session.scalars(query.order_by(SignalDecisionRecord.as_of.desc(), SignalDecisionRecord.id.desc()).offset(offset).limit(limit))).all()
        return tuple(SignalDecision.model_validate(item.payload) for item in records)

    async def find_recent_decisions(self, instrument: str, timeframe: str, at: datetime, limit: int = 20) -> tuple[SignalDecision, ...]:
        return await self.list_decisions(instrument, timeframe, end=at, limit=limit)

    async def prune(self, before: datetime, mode: DecisionMode, limit: int) -> int:
        identifiers = (
            await self.session.scalars(
                select(SignalDecisionRecord.id).where(SignalDecisionRecord.valid_until < before, SignalDecisionRecord.mode == mode.value).order_by(SignalDecisionRecord.valid_until).limit(limit)
            )
        ).all()
        if not identifiers:
            return 0
        await self.session.execute(delete(SignalDecisionRecord).where(SignalDecisionRecord.id.in_(identifiers)))
        await self.session.commit()
        return len(identifiers)
