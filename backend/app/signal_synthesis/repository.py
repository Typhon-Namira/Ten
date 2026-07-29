"""Idempotent persistence for multi-timeframe analytical synthesis."""

from __future__ import annotations

import asyncio
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    MultiTimeframeSignalSetRecord,
    TimeframeAnalyticalSignalRecord,
)
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import MultiTimeframeSignalSet


class MultiTimeframeSignalRepository(Protocol):
    async def save(self, value: MultiTimeframeSignalSet) -> MultiTimeframeSignalSet: ...
    async def for_state(self, market_state_id: UUID) -> MultiTimeframeSignalSet | None: ...
    async def latest(self, instrument: str) -> MultiTimeframeSignalSet | None: ...


class InMemoryMultiTimeframeSignalRepository:
    def __init__(self) -> None:
        self.values: dict[UUID, MultiTimeframeSignalSet] = {}
        self.by_state: dict[UUID, UUID] = {}
        self.lock = asyncio.Lock()

    async def save(self, value: MultiTimeframeSignalSet) -> MultiTimeframeSignalSet:
        async with self.lock:
            existing = self.by_state.get(value.market_state_id)
            if existing is not None:
                return self.values[existing]
            self.values[value.synthesis_id] = value
            self.by_state[value.market_state_id] = value.synthesis_id
            return value

    async def for_state(self, market_state_id: UUID) -> MultiTimeframeSignalSet | None:
        async with self.lock:
            identifier = self.by_state.get(market_state_id)
            return self.values.get(identifier) if identifier is not None else None

    async def latest(self, instrument: str) -> MultiTimeframeSignalSet | None:
        async with self.lock:
            values = tuple(item for item in self.values.values() if item.instrument == instrument)
        return max(values, key=lambda item: (item.market_timestamp, str(item.synthesis_id)), default=None)


class SqlAlchemyMultiTimeframeSignalRepository(
    ScopedSessionRepository,
):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        super().__init__(session_factory)

    @scoped_session
    async def save(self, value: MultiTimeframeSignalSet) -> MultiTimeframeSignalSet:
        await self.session.execute(
            insert(MultiTimeframeSignalSetRecord)
            .values(
                synthesis_id=value.synthesis_id,
                cycle_id=value.cycle_id,
                market_state_id=value.market_state_id,
                analysis_id=value.analysis_id,
                quantitative_forecast_id=value.quantitative_forecast_id,
                instrument=value.instrument,
                combined_direction=value.combined_signal.analytical_direction.value,
                combined_confidence=value.combined_signal.confidence,
                execution_status=value.combined_signal.execution_status.value,
                schema_version=value.schema_version,
                payload=value.model_dump(mode="json"),
                market_timestamp=value.market_timestamp,
                created_at=value.created_at,
            )
            .on_conflict_do_nothing(index_elements=["market_state_id"])
        )
        for signal in (*value.timeframe_signals, value.combined_signal):
            await self.session.execute(
                insert(TimeframeAnalyticalSignalRecord)
                .values(
                    signal_id=signal.signal_id,
                    synthesis_id=value.synthesis_id,
                    instrument=value.instrument,
                    timeframe=signal.timeframe,
                    analytical_direction=signal.analytical_direction.value,
                    confidence=signal.confidence,
                    strength=signal.strength.value,
                    execution_status=signal.execution_status.value,
                    payload={
                        "schema_version": value.schema_version,
                        "signal_id": str(signal.signal_id),
                        "synthesis_id": str(signal.synthesis_id),
                        "market_state_id": str(signal.market_state_id),
                        "analysis_id": str(signal.analysis_id),
                        "quantitative_forecast_id": str(
                            signal.quantitative_forecast_id
                        ),
                        "bullish_score": signal.bullish_score,
                        "bearish_score": signal.bearish_score,
                        "expected_horizon": signal.expected_horizon,
                        "confidence_decomposition": signal.confidence_decomposition.model_dump(
                            mode="json"
                        ),
                        "directional_thesis": signal.directional_thesis,
                        "invalidation_conditions": signal.invalidation_conditions,
                        "execution_eligibility": signal.execution_eligibility.value,
                        "blocking_reasons": signal.blocking_reasons,
                        "geometry": (
                            signal.geometry.model_dump(mode="json")
                            if signal.geometry is not None
                            else None
                        ),
                        "completed_at": signal.completed_at.isoformat(),
                    },
                    completed_at=signal.completed_at,
                )
                .on_conflict_do_nothing(index_elements=["synthesis_id", "timeframe"])
            )
        await self.session.commit()
        return await self.for_state(value.market_state_id) or value

    @scoped_session
    async def for_state(self, market_state_id: UUID) -> MultiTimeframeSignalSet | None:
        record = (
            await self.session.scalars(
                select(MultiTimeframeSignalSetRecord)
                .where(MultiTimeframeSignalSetRecord.market_state_id == market_state_id)
                .limit(1)
            )
        ).first()
        return MultiTimeframeSignalSet.model_validate(record.payload) if record else None

    @scoped_session
    async def latest(self, instrument: str) -> MultiTimeframeSignalSet | None:
        record = (
            await self.session.scalars(
                select(MultiTimeframeSignalSetRecord)
                .where(MultiTimeframeSignalSetRecord.instrument == instrument)
                .order_by(
                    MultiTimeframeSignalSetRecord.market_timestamp.desc(),
                    MultiTimeframeSignalSetRecord.synthesis_id.desc(),
                )
                .limit(1)
            )
        ).first()
        return MultiTimeframeSignalSet.model_validate(record.payload) if record else None
