"""Persistence ports for Unified Market State infrastructure."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    EvidenceItemRecord,
    MarketEvidenceFrameRecord,
    UnifiedMarketStateEvidenceLinkRecord,
    UnifiedMarketStateRecord,
    UnifiedMarketStateTimeframeRecord,
)
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import MarketEvidenceFrame, UnifiedMarketState


class UnifiedMarketStateRepository(Protocol):
    async def save_frame(self, value: MarketEvidenceFrame) -> MarketEvidenceFrame: ...
    async def latest_frame(self, instrument: str, timeframe: str, boundary: datetime, knowledge_cutoff: datetime) -> MarketEvidenceFrame | None: ...
    async def save_state(self, value: UnifiedMarketState) -> UnifiedMarketState: ...
    async def latest_state(self, instrument: str) -> UnifiedMarketState | None: ...


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

    async def latest_state(self, instrument: str) -> UnifiedMarketState | None:
        async with self._lock:
            values = [item for item in self._states.values() if item.instrument == instrument]
        return max(values, key=lambda item: (item.market_data_boundary, str(item.state_id)), default=None)


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
                    payload=value.model_dump(mode="json"),
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
                        payload=evidence_item.model_dump(mode="json"),
                    )
                    .on_conflict_do_nothing(index_elements=["evidence_id"])
                )
                await self.session.execute(
                    insert(UnifiedMarketStateEvidenceLinkRecord)
                    .values(state_id=value.state_id, evidence_id=evidence_item.evidence_id, ordinal=ordinal)
                    .on_conflict_do_nothing(index_elements=["state_id", "evidence_id"])
                )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return value

    @scoped_session
    async def latest_state(self, instrument: str) -> UnifiedMarketState | None:
        query = (
            select(UnifiedMarketStateRecord)
            .where(UnifiedMarketStateRecord.instrument == instrument)
            .order_by(UnifiedMarketStateRecord.market_data_boundary.desc())
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        return UnifiedMarketState.model_validate(record.payload) if record else None
