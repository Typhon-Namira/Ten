"""Persistence ports for Unified Market State infrastructure."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    EvidenceItemRecord,
    MarketEvidenceFrameRecord,
    UnifiedMarketStateEvidenceLinkRecord,
    UnifiedMarketStateCurrentRecord,
    UnifiedMarketStateRecord,
    UnifiedMarketStateTimeframeRecord,
)
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import EvidenceItem, MarketEvidenceFrame, TimeframeState, UnifiedMarketState


_STATE_RELATIONAL_FIELDS = {"timeframes", "evidence"}
_EVIDENCE_FRAME_FIELDS = {"raw_value", "normalized_value", "provenance"}


class _StateRecordLike(Protocol):
    payload: dict[str, Any]
    market_data_boundary: datetime


class _TimeframeRowLike(Protocol):
    frame_id: Any
    timeframe: str
    source_candle_close_at: datetime
    expected_candle_close_at: datetime
    stale: bool


class _EvidenceRecordLike(Protocol):
    evidence_id: Any
    source_frame_id: Any
    source_engine: str
    payload: dict[str, Any]


def _compact_state_payload(value: UnifiedMarketState) -> dict[str, object]:
    """Store state metadata once; relational rows remain the source of its collections.

    Prior versions embedded the complete engine payload in the frame, every evidence row,
    and the state.  Production measurements showed those three copies consumed 81% of the
    database.  The immutable frame is now the single payload owner.
    """

    return value.model_dump(mode="json", exclude=_STATE_RELATIONAL_FIELDS)


def _compact_evidence_payload(value: EvidenceItem) -> dict[str, object]:
    return value.model_dump(mode="json", exclude=_EVIDENCE_FRAME_FIELDS)


def _reconstruct_compact_state(
    record: _StateRecordLike,
    timeframe_rows: Sequence[_TimeframeRowLike],
    frames: Mapping[Any, MarketEvidenceFrame],
    link_rows: Sequence[tuple[Any, _EvidenceRecordLike]],
) -> UnifiedMarketState:
    """Rehydrate the exact domain object from compact rows and immutable frames."""

    timeframes: list[TimeframeState] = []
    for item in timeframe_rows:
        frame_id = item.frame_id
        frame = frames[frame_id]
        evidence_ids = tuple(
            evidence.evidence_id
            for _, evidence in link_rows
            if evidence.source_frame_id == frame_id
        )
        source_close = item.source_candle_close_at
        timeframes.append(
            TimeframeState(
                timeframe=item.timeframe,
                frame_id=frame_id,
                source_candle_open_at=frame.candle_open_at,
                source_candle_close_at=source_close,
                expected_candle_close_at=item.expected_candle_close_at,
                freshness_seconds=max(
                    0.0,
                    (record.market_data_boundary - source_close).total_seconds(),
                ),
                stale=item.stale,
                evidence_ids=evidence_ids,
            )
        )

    evidence_items: list[EvidenceItem] = []
    for _, evidence_record in link_rows:
        frame = frames[evidence_record.source_frame_id]
        captured = next(
            item
            for item in frame.evidence
            if item.source_engine == evidence_record.source_engine
        )
        payload = dict(evidence_record.payload)
        payload.update(
            raw_value=captured.raw_value,
            normalized_value=captured.normalized_value,
            provenance=captured.provenance,
        )
        evidence_items.append(EvidenceItem.model_validate(payload))

    payload = dict(record.payload)
    payload.update(
        timeframes=[item.model_dump(mode="json") for item in timeframes],
        evidence=[item.model_dump(mode="json") for item in evidence_items],
    )
    return UnifiedMarketState.model_validate(payload)


class UnifiedMarketStateRepository(Protocol):
    async def save_frame(self, value: MarketEvidenceFrame) -> MarketEvidenceFrame: ...
    async def latest_frame(self, instrument: str, timeframe: str, boundary: datetime, knowledge_cutoff: datetime) -> MarketEvidenceFrame | None: ...
    async def save_state(self, value: UnifiedMarketState) -> UnifiedMarketState: ...
    async def get_state(self, state_id: object) -> UnifiedMarketState | None: ...
    async def latest_state(
        self,
        instrument: str,
        trigger_timeframe: str | None = None,
    ) -> UnifiedMarketState | None: ...


class InMemoryUnifiedMarketStateRepository:
    def __init__(self) -> None:
        self._frames: dict[object, MarketEvidenceFrame] = {}
        self._states: dict[object, UnifiedMarketState] = {}
        self._lock = asyncio.Lock()

    async def save_frame(self, value: MarketEvidenceFrame) -> MarketEvidenceFrame:
        async with self._lock:
            self._frames[value.frame_id] = value
        return value

    async def latest_frame(self, instrument: str, timeframe: str, boundary: datetime, knowledge_cutoff: datetime) -> MarketEvidenceFrame | None:
        async with self._lock:
            values = [
                item
                for item in self._frames.values()
                if item.instrument == instrument
                and item.timeframe == timeframe
                and item.candle_close_at <= boundary
                and item.knowledge_cutoff <= knowledge_cutoff
            ]
        return max(values, key=lambda item: (item.candle_close_at, item.knowledge_cutoff, str(item.frame_id)), default=None)

    async def save_state(self, value: UnifiedMarketState) -> UnifiedMarketState:
        async with self._lock:
            self._states[value.state_id] = value
        return value

    async def latest_state(
        self,
        instrument: str,
        trigger_timeframe: str | None = None,
    ) -> UnifiedMarketState | None:
        async with self._lock:
            values = [
                item
                for item in self._states.values()
                if item.instrument == instrument
                and (
                    trigger_timeframe is None
                    or item.trigger_timeframe == trigger_timeframe
                )
            ]
        return max(values, key=lambda item: (item.market_data_boundary, str(item.state_id)), default=None)

    async def get_state(self, state_id: object) -> UnifiedMarketState | None:
        async with self._lock:
            return self._states.get(state_id)


class SqlAlchemyUnifiedMarketStateRepository(ScopedSessionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        super().__init__(session_factory)

    @scoped_session
    async def save_frame(self, value: MarketEvidenceFrame) -> MarketEvidenceFrame:
        try:
            await self.session.execute(
                insert(MarketEvidenceFrameRecord)
                .values(
                    frame_id=value.frame_id,
                    frame_hash=value.frame_hash,
                    instrument=value.instrument,
                    timeframe=value.timeframe,
                    candle_close_at=value.candle_close_at,
                    knowledge_cutoff=value.knowledge_cutoff,
                    payload=value.model_dump(mode="json"),
                    created_at=value.created_at,
                )
                .on_conflict_do_nothing(index_elements=["frame_hash"])
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return value

    @scoped_session
    async def latest_frame(self, instrument: str, timeframe: str, boundary: datetime, knowledge_cutoff: datetime) -> MarketEvidenceFrame | None:
        query = (
            select(MarketEvidenceFrameRecord)
            .where(
                MarketEvidenceFrameRecord.instrument == instrument,
                MarketEvidenceFrameRecord.timeframe == timeframe,
                MarketEvidenceFrameRecord.candle_close_at <= boundary,
                MarketEvidenceFrameRecord.knowledge_cutoff <= knowledge_cutoff,
            )
            .order_by(MarketEvidenceFrameRecord.candle_close_at.desc(), MarketEvidenceFrameRecord.knowledge_cutoff.desc())
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        return MarketEvidenceFrame.model_validate(record.payload) if record else None

    @scoped_session
    async def save_state(self, value: UnifiedMarketState) -> UnifiedMarketState:
        try:
            await self.session.execute(
                insert(UnifiedMarketStateRecord)
                .values(
                    state_id=value.state_id,
                    state_hash=value.state_hash,
                    instrument=value.instrument,
                    trigger_timeframe=value.trigger_timeframe,
                    market_data_boundary=value.market_data_boundary,
                    knowledge_cutoff=value.knowledge_cutoff,
                    status=value.status.value,
                    payload=_compact_state_payload(value),
                    created_at=value.created_at,
                )
                .on_conflict_do_nothing(index_elements=["state_hash"])
            )
            for timeframe_state in value.timeframes:
                await self.session.execute(
                    insert(UnifiedMarketStateTimeframeRecord)
                    .values(
                        state_id=value.state_id,
                        timeframe=timeframe_state.timeframe,
                        frame_id=timeframe_state.frame_id,
                        source_candle_close_at=timeframe_state.source_candle_close_at,
                        expected_candle_close_at=timeframe_state.expected_candle_close_at,
                        stale=timeframe_state.stale,
                    )
                    .on_conflict_do_nothing(index_elements=["state_id", "timeframe"])
                )
            for ordinal, evidence_item in enumerate(value.evidence):
                await self.session.execute(
                    insert(EvidenceItemRecord)
                    .values(
                        evidence_id=evidence_item.evidence_id,
                        source_frame_id=evidence_item.source_frame_id,
                        source_engine=evidence_item.source_engine,
                        source_timeframe=evidence_item.source_timeframe,
                        availability=evidence_item.availability.value,
                        source_candle_close_at=evidence_item.source_candle_close_timestamp,
                        available_at=evidence_item.available_at,
                        payload=_compact_evidence_payload(evidence_item),
                    )
                    .on_conflict_do_nothing(index_elements=["evidence_id"])
                )
                await self.session.execute(
                    insert(UnifiedMarketStateEvidenceLinkRecord)
                    .values(state_id=value.state_id, evidence_id=evidence_item.evidence_id, ordinal=ordinal)
                    .on_conflict_do_nothing(index_elements=["state_id", "ordinal"])
                )
            await self.session.execute(
                insert(UnifiedMarketStateCurrentRecord)
                .values(
                    instrument=value.instrument,
                    state_id=value.state_id,
                    state_hash=value.state_hash,
                    updated_at=value.created_at,
                )
                .on_conflict_do_update(
                    index_elements=["instrument"],
                    set_={
                        "state_id": value.state_id,
                        "state_hash": value.state_hash,
                        "updated_at": value.created_at,
                    },
                    where=UnifiedMarketStateCurrentRecord.updated_at <= value.created_at,
                )
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return value

    @scoped_session
    async def get_state(self, state_id: object) -> UnifiedMarketState | None:
        record = await self.session.get(UnifiedMarketStateRecord, state_id)
        if record is None:
            return None
        if record.payload.get("timeframes") is not None and record.payload.get("evidence") is not None:
            return UnifiedMarketState.model_validate(record.payload)
        timeframe_rows = list(
            (
                await self.session.scalars(
                    select(UnifiedMarketStateTimeframeRecord)
                    .where(UnifiedMarketStateTimeframeRecord.state_id == record.state_id)
                    .order_by(UnifiedMarketStateTimeframeRecord.timeframe)
                )
            ).all()
        )
        frame_ids = [item.frame_id for item in timeframe_rows]
        frame_rows = list(
            (
                await self.session.scalars(
                    select(MarketEvidenceFrameRecord).where(
                        MarketEvidenceFrameRecord.frame_id.in_(frame_ids)
                    )
                )
            ).all()
        )
        frames = {
            item.frame_id: MarketEvidenceFrame.model_validate(item.payload)
            for item in frame_rows
        }
        link_rows = (
            await self.session.execute(
                select(UnifiedMarketStateEvidenceLinkRecord, EvidenceItemRecord)
                .join(
                    EvidenceItemRecord,
                    EvidenceItemRecord.evidence_id
                    == UnifiedMarketStateEvidenceLinkRecord.evidence_id,
                )
                .where(UnifiedMarketStateEvidenceLinkRecord.state_id == record.state_id)
                .order_by(UnifiedMarketStateEvidenceLinkRecord.ordinal)
            )
        ).all()
        typed_links = [(row[0], row[1]) for row in link_rows]
        return _reconstruct_compact_state(record, timeframe_rows, frames, typed_links)

    @scoped_session
    async def latest_state(
        self,
        instrument: str,
        trigger_timeframe: str | None = None,
    ) -> UnifiedMarketState | None:
        query = (
            select(UnifiedMarketStateRecord)
            .outerjoin(
                UnifiedMarketStateCurrentRecord,
                UnifiedMarketStateCurrentRecord.state_id == UnifiedMarketStateRecord.state_id,
            )
            .where(UnifiedMarketStateRecord.instrument == instrument)
            .order_by(
                UnifiedMarketStateCurrentRecord.updated_at.desc().nullslast(),
                UnifiedMarketStateRecord.market_data_boundary.desc(),
            )
            .limit(1)
        )
        if trigger_timeframe is not None:
            query = query.where(
                UnifiedMarketStateRecord.trigger_timeframe == trigger_timeframe
            )
        record = (await self.session.scalars(query)).first()
        if record is None:
            return None
        # Legacy rows remain directly readable during the rolling migration.
        if record.payload.get("timeframes") is not None and record.payload.get("evidence") is not None:
            return UnifiedMarketState.model_validate(record.payload)

        timeframe_rows = list(
            (
                await self.session.scalars(
                    select(UnifiedMarketStateTimeframeRecord)
                    .where(UnifiedMarketStateTimeframeRecord.state_id == record.state_id)
                    .order_by(UnifiedMarketStateTimeframeRecord.timeframe)
                )
            ).all()
        )
        frame_ids = [item.frame_id for item in timeframe_rows]
        frame_rows = list(
            (
                await self.session.scalars(
                    select(MarketEvidenceFrameRecord).where(
                        MarketEvidenceFrameRecord.frame_id.in_(frame_ids)
                    )
                )
            ).all()
        )
        frames = {
            item.frame_id: MarketEvidenceFrame.model_validate(item.payload)
            for item in frame_rows
        }
        link_rows = (
            await self.session.execute(
                select(UnifiedMarketStateEvidenceLinkRecord, EvidenceItemRecord)
                .join(
                    EvidenceItemRecord,
                    EvidenceItemRecord.evidence_id
                    == UnifiedMarketStateEvidenceLinkRecord.evidence_id,
                )
                .where(UnifiedMarketStateEvidenceLinkRecord.state_id == record.state_id)
                .order_by(UnifiedMarketStateEvidenceLinkRecord.ordinal)
            )
        ).all()

        typed_links = [(row[0], row[1]) for row in link_rows]
        return _reconstruct_compact_state(record, timeframe_rows, frames, typed_links)
