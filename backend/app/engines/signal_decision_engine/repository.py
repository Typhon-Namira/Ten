from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.batching import bounded_insert_chunks
from backend.app.storage.models import (
    SignalDecisionReasonRecord,
    SignalDecisionRecord,
    SignalDecisionRuleRecord,
    SignalEmailOutboxRecord,
)
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .exceptions import SignalDecisionPersistenceError
from .models import DecisionDirection, DecisionMode, DecisionState, SignalDecision, stable_id


def _notification_expiration(notification: Mapping[str, object] | None) -> datetime | None:
    if notification is None or not notification.get("expires_at"):
        return None
    try:
        return datetime.fromisoformat(
            str(notification["expires_at"]).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=UTC)


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
    async def get_latest_decision(self, instrument: str, timeframe: str, direction: DecisionDirection | None = None, state: DecisionState | None = None) -> SignalDecision | None: ...

    @abstractmethod
    async def find_by_analysis_lineage(
        self,
        instrument: str,
        timeframe: str,
        market_snapshot_id: UUID,
        analysis_id: UUID,
        signal_id: UUID,
    ) -> SignalDecision | None: ...

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

    async def get_latest_decision(self, instrument: str, timeframe: str, direction: DecisionDirection | None = None, state: DecisionState | None = None) -> SignalDecision | None:
        values = await self.list_decisions(instrument, timeframe, direction=direction, state=state, limit=1)
        return values[0] if values else None

    async def find_by_analysis_lineage(
        self,
        instrument: str,
        timeframe: str,
        market_snapshot_id: UUID,
        analysis_id: UUID,
        signal_id: UUID,
    ) -> SignalDecision | None:
        async with self._lock:
            matches = [
                item
                for item in self._decisions.values()
                if item.instrument == instrument
                and item.timeframe == timeframe
                and item.source_lineage is not None
                and item.source_lineage.market_snapshot_id == market_snapshot_id
                and item.source_lineage.current_ai_analysis_id == analysis_id
                and item.source_lineage.current_ai_signal_id == signal_id
            ]
        return max(
            matches,
            key=lambda item: (item.decided_at, str(item.decision_id)),
            default=None,
        )

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


class SqlAlchemySignalDecisionRepository(SignalDecisionRepository, ScopedSessionRepository):
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        signal_email_enabled: bool = False,
        signal_email_recipient: str = "tufannamira@gmail.com",
    ) -> None:
        ScopedSessionRepository.__init__(self, session_factory)
        self.signal_email_enabled = signal_email_enabled
        self.signal_email_recipient = signal_email_recipient

    @scoped_session
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
                for chunk in bounded_insert_chunks(rules):
                    await self.session.execute(insert(SignalDecisionRuleRecord).values(list(chunk)).on_conflict_do_nothing(index_elements=["id"]))
            if reasons:
                for chunk in bounded_insert_chunks(reasons):
                    await self.session.execute(insert(SignalDecisionReasonRecord).values(list(chunk)).on_conflict_do_nothing(index_elements=["id"]))
            notification = decision.notification_context
            notification_expires_at = _notification_expiration(notification)
            if (
                self.signal_email_enabled
                and decision.mode == DecisionMode.LIVE
                and decision.publication_eligible
                and notification is not None
                and notification.get("primary_scenario_id") is not None
                and notification.get("direction") in {"BUY", "SELL"}
                and all(
                    notification.get(field) is not None
                    for field in ("entry", "stop_loss", "take_profit", "risk_reward")
                )
                and float(notification["risk_reward"]) > 0
                and float(notification.get("primary_scenario_score", 100))
                >= float(notification.get("email_threshold", 0))
                and decision.decided_at < decision.valid_until
                and (
                    notification_expires_at is None
                    or decision.decided_at < notification_expires_at
                )
            ):
                now = datetime.now(UTC)
                payload = {
                    **notification,
                    "symbol": decision.instrument,
                    "market_time": decision.as_of.isoformat(),
                    "decision_id": str(decision.decision_id),
                    "guardrail_status": (
                        "APPROVED" if decision.publication_eligible else "REJECTED"
                    ),
                    "publication_status": (
                        "ELIGIBLE" if decision.publication_eligible else "INELIGIBLE"
                    ),
                    "blockers": [
                        item.reason_code
                        for item in (*decision.blockers, *decision.warnings)
                    ],
                }
                deduplication_key = sha256(
                    "|".join(
                        str(value)
                        for value in (
                            decision.instrument,
                            notification.get("market_cutoff", decision.as_of.isoformat()),
                            notification.get("primary_scenario_id", notification["signal_id"]),
                            notification["direction"],
                            notification["entry"],
                            notification["stop_loss"],
                            notification["take_profit"],
                        )
                    ).encode()
                ).hexdigest()
                await self.session.execute(
                    insert(SignalEmailOutboxRecord)
                    .values(
                        id=stable_id("signal-email", notification["signal_id"]),
                        signal_id=UUID(str(notification["signal_id"])),
                        primary_scenario_id=(
                            UUID(str(notification["primary_scenario_id"]))
                            if notification.get("primary_scenario_id")
                            else None
                        ),
                        deduplication_key=deduplication_key,
                        decision_id=decision.decision_id,
                        recipient=self.signal_email_recipient,
                        status="PENDING",
                        payload=payload,
                        attempt_count=0,
                        next_retry_at=now,
                        created_at=now,
                        updated_at=now,
                    )
                    .on_conflict_do_nothing()
                )
            await self.session.commit()
            return await self.find_by_fingerprint(decision.input_fingerprint, decision.mode) or decision
        except Exception as exc:
            await self.session.rollback()
            raise SignalDecisionPersistenceError("Signal Decision persistence failed") from exc

    @scoped_session
    async def get_decision(self, decision_id: UUID) -> SignalDecision | None:
        record = await self.session.get(SignalDecisionRecord, decision_id)
        return SignalDecision.model_validate(record.payload) if record else None

    @scoped_session
    async def find_by_fingerprint(self, fingerprint: str, mode: DecisionMode) -> SignalDecision | None:
        query = select(SignalDecisionRecord).where(SignalDecisionRecord.input_fingerprint == fingerprint, SignalDecisionRecord.mode == mode.value).limit(1)
        record = (await self.session.scalars(query)).first()
        return SignalDecision.model_validate(record.payload) if record else None

    @scoped_session
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

    @scoped_session
    async def get_latest_decision(self, instrument: str, timeframe: str, direction: DecisionDirection | None = None, state: DecisionState | None = None) -> SignalDecision | None:
        query = select(SignalDecisionRecord).where(SignalDecisionRecord.instrument == instrument, SignalDecisionRecord.timeframe == timeframe)
        if direction:
            query = query.where(SignalDecisionRecord.direction == direction.value)
        if state:
            query = query.where(SignalDecisionRecord.state == state.value)
        record = (await self.session.scalars(query.order_by(SignalDecisionRecord.as_of.desc(), SignalDecisionRecord.id.desc()).limit(1))).first()
        return SignalDecision.model_validate(record.payload) if record else None

    @scoped_session
    async def find_by_analysis_lineage(
        self,
        instrument: str,
        timeframe: str,
        market_snapshot_id: UUID,
        analysis_id: UUID,
        signal_id: UUID,
    ) -> SignalDecision | None:
        source_lineage = SignalDecisionRecord.payload["source_lineage"]
        query = (
            select(SignalDecisionRecord)
            .where(
                SignalDecisionRecord.instrument == instrument,
                SignalDecisionRecord.timeframe == timeframe,
                source_lineage["market_snapshot_id"].astext
                == str(market_snapshot_id),
                source_lineage["current_ai_analysis_id"].astext == str(analysis_id),
                source_lineage["current_ai_signal_id"].astext == str(signal_id),
            )
            .order_by(
                SignalDecisionRecord.as_of.desc(),
                SignalDecisionRecord.id.desc(),
            )
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        return SignalDecision.model_validate(record.payload) if record else None

    @scoped_session
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

    @scoped_session
    async def prune(self, before: datetime, mode: DecisionMode, limit: int) -> int:
        identifiers = (
            await self.session.scalars(
                select(SignalDecisionRecord.id).where(SignalDecisionRecord.valid_until < before, SignalDecisionRecord.mode == mode.value).order_by(SignalDecisionRecord.valid_until).limit(limit)
            )
        ).all()
        if not identifiers:
            return 0
        try:
            await self.session.execute(delete(SignalDecisionRecord).where(SignalDecisionRecord.id.in_(identifiers)))
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return len(identifiers)
