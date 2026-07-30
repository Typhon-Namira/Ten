"""Idempotent scenario persistence and point-in-time history queries."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    CombinedForwardScenarioRecord,
    ForwardMarketScenarioRecord,
    ScenarioOutcomeRecord,
)
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import CombinedForwardScenario, ForwardMarketScenario, ScenarioOutcome


class ScenarioForecastRepository(Protocol):
    async def save_scenario(self, value: ForwardMarketScenario) -> ForwardMarketScenario: ...
    async def save_combined(self, value: CombinedForwardScenario) -> CombinedForwardScenario: ...
    async def save_outcome(self, value: ScenarioOutcome) -> ScenarioOutcome: ...
    async def latest_scenario(
        self, instrument: str, timeframe: str, *, at_or_before: datetime | None = None
    ) -> ForwardMarketScenario | None: ...
    async def latest_combined(self, instrument: str) -> CombinedForwardScenario | None: ...
    async def pending_before(
        self, instrument: str, timeframe: str, expiry: datetime
    ) -> tuple[ForwardMarketScenario, ...]: ...
    async def completed_history(
        self, instrument: str, *, limit: int = 100
    ) -> tuple[tuple[ForwardMarketScenario, ScenarioOutcome], ...]: ...


class InMemoryScenarioForecastRepository:
    def __init__(self) -> None:
        self.scenarios: dict[UUID, ForwardMarketScenario] = {}
        self.combined: dict[UUID, CombinedForwardScenario] = {}
        self.outcomes: dict[UUID, ScenarioOutcome] = {}
        self._scope: dict[tuple[str, str, datetime], UUID] = {}
        self._lock = asyncio.Lock()

    async def save_scenario(self, value: ForwardMarketScenario) -> ForwardMarketScenario:
        async with self._lock:
            key = (
                value.instrument,
                value.timeframe,
                value.market_cutoff_time,
            )
            existing = self._scope.get(key)
            if existing is not None:
                return self.scenarios[existing]
            self.scenarios[value.scenario_id] = value
            self._scope[key] = value.scenario_id
            return value

    async def save_combined(self, value: CombinedForwardScenario) -> CombinedForwardScenario:
        async with self._lock:
            existing = next(
                (
                    item
                    for item in self.combined.values()
                    if item.market_state_id == value.market_state_id
                ),
                None,
            )
            if existing is not None:
                return existing
            self.combined[value.combined_scenario_id] = value
            return value

    async def save_outcome(self, value: ScenarioOutcome) -> ScenarioOutcome:
        async with self._lock:
            existing = self.outcomes.get(value.scenario_id)
            if existing is not None:
                return existing
            self.outcomes[value.scenario_id] = value
            return value

    async def latest_scenario(
        self, instrument: str, timeframe: str, *, at_or_before: datetime | None = None
    ) -> ForwardMarketScenario | None:
        async with self._lock:
            values = tuple(
                item
                for item in self.scenarios.values()
                if item.instrument == instrument
                and item.timeframe == timeframe
                and (at_or_before is None or item.market_cutoff_time <= at_or_before)
            )
        return max(
            values,
            key=lambda item: (item.market_cutoff_time, str(item.scenario_id)),
            default=None,
        )

    async def latest_combined(self, instrument: str) -> CombinedForwardScenario | None:
        async with self._lock:
            values = tuple(
                item for item in self.combined.values() if item.instrument == instrument
            )
        return max(
            values,
            key=lambda item: (item.market_cutoff_time, str(item.combined_scenario_id)),
            default=None,
        )

    async def pending_before(
        self, instrument: str, timeframe: str, expiry: datetime
    ) -> tuple[ForwardMarketScenario, ...]:
        async with self._lock:
            return tuple(
                item
                for item in self.scenarios.values()
                if item.instrument == instrument
                and item.timeframe == timeframe
                and item.expiry <= expiry
                and item.scenario_id not in self.outcomes
            )

    async def completed_history(
        self, instrument: str, *, limit: int = 100
    ) -> tuple[tuple[ForwardMarketScenario, ScenarioOutcome], ...]:
        async with self._lock:
            values = [
                (self.scenarios[scenario_id], outcome)
                for scenario_id, outcome in self.outcomes.items()
                if self.scenarios[scenario_id].instrument == instrument
            ]
        return tuple(
            sorted(values, key=lambda item: item[1].completed_at, reverse=True)[:limit]
        )


class SqlAlchemyScenarioForecastRepository(ScopedSessionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        super().__init__(session_factory)

    @scoped_session
    async def save_scenario(self, value: ForwardMarketScenario) -> ForwardMarketScenario:
        await self.session.execute(
            insert(ForwardMarketScenarioRecord)
            .values(
                scenario_id=value.scenario_id,
                cycle_id=value.cycle_id,
                market_state_id=value.market_state_id,
                synthesis_id=value.synthesis_id,
                analysis_id=value.analysis_id,
                quantitative_forecast_id=value.quantitative_forecast_id,
                instrument=value.instrument,
                timeframe=value.timeframe,
                primary_direction=value.primary_direction.value,
                scenario_validity=value.scenario_validity.value,
                execution_geometry_validity=value.execution_geometry_validity.value,
                market_cutoff_time=value.market_cutoff_time,
                expiry=value.expiry,
                payload=value.model_dump(mode="json"),
                created_at=value.created_at,
            )
            .on_conflict_do_nothing(
                index_elements=["instrument", "timeframe", "market_cutoff_time"]
            )
        )
        await self.session.commit()
        return (
            await self.latest_scenario(
                value.instrument,
                value.timeframe,
                at_or_before=value.market_cutoff_time,
            )
        ) or value

    @scoped_session
    async def save_combined(self, value: CombinedForwardScenario) -> CombinedForwardScenario:
        await self.session.execute(
            insert(CombinedForwardScenarioRecord)
            .values(
                combined_scenario_id=value.combined_scenario_id,
                cycle_id=value.cycle_id,
                market_state_id=value.market_state_id,
                m5_scenario_id=value.m5_scenario_id,
                m15_scenario_id=value.m15_scenario_id,
                instrument=value.instrument,
                agreement=value.agreement.value,
                combined_direction=value.combined_direction.value,
                execution_geometry_validity=value.execution_geometry_validity.value,
                market_cutoff_time=value.market_cutoff_time,
                expiry=value.expiry,
                payload=value.model_dump(mode="json"),
                created_at=value.created_at,
            )
            .on_conflict_do_nothing(index_elements=["market_state_id"])
        )
        await self.session.commit()
        record = (
            await self.session.scalars(
                select(CombinedForwardScenarioRecord)
                .where(CombinedForwardScenarioRecord.market_state_id == value.market_state_id)
                .limit(1)
            )
        ).first()
        return CombinedForwardScenario.model_validate(record.payload) if record else value

    @scoped_session
    async def save_outcome(self, value: ScenarioOutcome) -> ScenarioOutcome:
        await self.session.execute(
            insert(ScenarioOutcomeRecord)
            .values(
                outcome_id=value.outcome_id,
                scenario_id=value.scenario_id,
                status=value.status.value,
                calibration_bucket=value.calibration_bucket,
                directional_accuracy=value.directional_accuracy,
                payload=value.model_dump(mode="json"),
                evaluated_at=value.evaluated_at,
                completed_at=value.completed_at,
            )
            .on_conflict_do_nothing(index_elements=["scenario_id"])
        )
        await self.session.commit()
        record = (
            await self.session.scalars(
                select(ScenarioOutcomeRecord)
                .where(ScenarioOutcomeRecord.scenario_id == value.scenario_id)
                .limit(1)
            )
        ).first()
        return ScenarioOutcome.model_validate(record.payload) if record else value

    @scoped_session
    async def _for_state(
        self, market_state_id: UUID, timeframe: str
    ) -> ForwardMarketScenario | None:
        record = (
            await self.session.scalars(
                select(ForwardMarketScenarioRecord)
                .where(
                    ForwardMarketScenarioRecord.market_state_id == market_state_id,
                    ForwardMarketScenarioRecord.timeframe == timeframe,
                )
                .limit(1)
            )
        ).first()
        return ForwardMarketScenario.model_validate(record.payload) if record else None

    @scoped_session
    async def latest_scenario(
        self, instrument: str, timeframe: str, *, at_or_before: datetime | None = None
    ) -> ForwardMarketScenario | None:
        query = select(ForwardMarketScenarioRecord).where(
            ForwardMarketScenarioRecord.instrument == instrument,
            ForwardMarketScenarioRecord.timeframe == timeframe,
        )
        if at_or_before is not None:
            query = query.where(
                ForwardMarketScenarioRecord.market_cutoff_time <= at_or_before
            )
        record = (
            await self.session.scalars(
                query.order_by(
                    ForwardMarketScenarioRecord.market_cutoff_time.desc(),
                    ForwardMarketScenarioRecord.scenario_id.desc(),
                ).limit(1)
            )
        ).first()
        return ForwardMarketScenario.model_validate(record.payload) if record else None

    @scoped_session
    async def latest_combined(self, instrument: str) -> CombinedForwardScenario | None:
        record = (
            await self.session.scalars(
                select(CombinedForwardScenarioRecord)
                .where(CombinedForwardScenarioRecord.instrument == instrument)
                .order_by(
                    CombinedForwardScenarioRecord.market_cutoff_time.desc(),
                    CombinedForwardScenarioRecord.combined_scenario_id.desc(),
                )
                .limit(1)
            )
        ).first()
        return CombinedForwardScenario.model_validate(record.payload) if record else None

    @scoped_session
    async def pending_before(
        self, instrument: str, timeframe: str, expiry: datetime
    ) -> tuple[ForwardMarketScenario, ...]:
        records = (
            await self.session.scalars(
                select(ForwardMarketScenarioRecord)
                .outerjoin(
                    ScenarioOutcomeRecord,
                    ScenarioOutcomeRecord.scenario_id
                    == ForwardMarketScenarioRecord.scenario_id,
                )
                .where(
                    ForwardMarketScenarioRecord.instrument == instrument,
                    ForwardMarketScenarioRecord.timeframe == timeframe,
                    ForwardMarketScenarioRecord.expiry <= expiry,
                    ScenarioOutcomeRecord.scenario_id.is_(None),
                )
                .order_by(ForwardMarketScenarioRecord.expiry)
            )
        ).all()
        return tuple(ForwardMarketScenario.model_validate(item.payload) for item in records)

    @scoped_session
    async def completed_history(
        self, instrument: str, *, limit: int = 100
    ) -> tuple[tuple[ForwardMarketScenario, ScenarioOutcome], ...]:
        rows = (
            await self.session.execute(
                select(ForwardMarketScenarioRecord, ScenarioOutcomeRecord)
                .join(
                    ScenarioOutcomeRecord,
                    ScenarioOutcomeRecord.scenario_id
                    == ForwardMarketScenarioRecord.scenario_id,
                )
                .where(ForwardMarketScenarioRecord.instrument == instrument)
                .order_by(ScenarioOutcomeRecord.completed_at.desc())
                .limit(limit)
            )
        ).all()
        return tuple(
            (
                ForwardMarketScenario.model_validate(scenario.payload),
                ScenarioOutcome.model_validate(outcome.payload),
            )
            for scenario, outcome in rows
        )
