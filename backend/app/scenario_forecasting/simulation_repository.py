"""Idempotent normalized persistence for market simulations and selections."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    CandidateScenarioOutcomeRecord,
    CandidateMarketScenarioRecord,
    MarketSimulationCycleRecord,
    PrimaryScenarioGeometryRecord,
    PrimaryScenarioSelectionRecord,
    ScenarioLifecycleTransitionRecord,
    ScenarioPathStageRecord,
    ScenarioScoreComponentRecord,
)
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .simulation_models import (
    CandidateMarketScenario,
    CandidateScenarioOutcome,
    MarketSimulationCycle,
    PrimaryScenarioSelection,
)


class MarketSimulationRepository(Protocol):
    async def save(
        self,
        simulation: MarketSimulationCycle,
        selection: PrimaryScenarioSelection,
    ) -> PrimaryScenarioSelection: ...

    async def for_state(self, market_state_id: UUID) -> PrimaryScenarioSelection | None: ...

    async def latest(self, instrument: str) -> PrimaryScenarioSelection | None: ...

    async def candidates(
        self, simulation_cycle_id: UUID
    ) -> tuple[CandidateMarketScenario, ...]: ...

    async def pending_primary_before(
        self, instrument: str, cutoff: datetime
    ) -> tuple[tuple[PrimaryScenarioSelection, CandidateMarketScenario], ...]: ...

    async def save_outcome(
        self, outcome: CandidateScenarioOutcome
    ) -> CandidateScenarioOutcome: ...


class InMemoryMarketSimulationRepository:
    def __init__(self) -> None:
        self.simulations: dict[UUID, MarketSimulationCycle] = {}
        self.selections: dict[UUID, PrimaryScenarioSelection] = {}
        self._state: dict[UUID, UUID] = {}
        self.outcomes: dict[UUID, CandidateScenarioOutcome] = {}
        self._lock = asyncio.Lock()

    async def save(
        self,
        simulation: MarketSimulationCycle,
        selection: PrimaryScenarioSelection,
    ) -> PrimaryScenarioSelection:
        async with self._lock:
            existing_id = self._state.get(simulation.market_state_id)
            if existing_id is not None:
                return self.selections[existing_id]
            self.simulations[simulation.simulation_cycle_id] = simulation
            self.selections[selection.selection_id] = selection
            self._state[simulation.market_state_id] = selection.selection_id
            return selection

    async def for_state(self, market_state_id: UUID) -> PrimaryScenarioSelection | None:
        async with self._lock:
            selection_id = self._state.get(market_state_id)
            return self.selections.get(selection_id) if selection_id else None

    async def latest(self, instrument: str) -> PrimaryScenarioSelection | None:
        async with self._lock:
            values = [
                item for item in self.selections.values() if item.instrument == instrument
            ]
        return max(values, key=lambda item: item.market_cutoff, default=None)

    async def candidates(
        self, simulation_cycle_id: UUID
    ) -> tuple[CandidateMarketScenario, ...]:
        async with self._lock:
            simulation = self.simulations.get(simulation_cycle_id)
            return simulation.candidates if simulation else ()

    async def pending_primary_before(
        self, instrument: str, cutoff: datetime
    ) -> tuple[tuple[PrimaryScenarioSelection, CandidateMarketScenario], ...]:
        async with self._lock:
            return tuple(
                (selection, selection.primary)
                for selection in self.selections.values()
                if selection.instrument == instrument
                and selection.primary is not None
                and selection.primary.expiry <= cutoff
                and selection.primary.candidate_id not in self.outcomes
            )

    async def save_outcome(
        self, outcome: CandidateScenarioOutcome
    ) -> CandidateScenarioOutcome:
        async with self._lock:
            return self.outcomes.setdefault(outcome.candidate_id, outcome)


class SqlAlchemyMarketSimulationRepository(ScopedSessionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        super().__init__(session_factory)

    @scoped_session
    async def save(
        self,
        simulation: MarketSimulationCycle,
        selection: PrimaryScenarioSelection,
    ) -> PrimaryScenarioSelection:
        await self.session.execute(
            insert(MarketSimulationCycleRecord)
            .values(
                simulation_cycle_id=simulation.simulation_cycle_id,
                cycle_id=simulation.cycle_id,
                market_state_id=simulation.market_state_id,
                synthesis_id=simulation.synthesis_id,
                analysis_id=simulation.analysis_id,
                quantitative_forecast_id=simulation.quantitative_forecast_id,
                instrument=simulation.instrument,
                market_cutoff=simulation.market_cutoff,
                candidate_count=simulation.candidate_count,
                engine_version=simulation.engine_version,
                configuration_version=simulation.configuration_version,
                created_at=simulation.created_at,
            )
            .on_conflict_do_nothing()
        )
        for candidate in simulation.candidates:
            await self.session.execute(
                insert(CandidateMarketScenarioRecord)
                .values(
                    candidate_id=candidate.candidate_id,
                    simulation_cycle_id=candidate.simulation_cycle_id,
                    direction=candidate.direction.value,
                    scenario_type=candidate.scenario_type,
                    rank=candidate.rank,
                    final_scenario_score=candidate.final_scenario_score,
                    scenario_validity=candidate.scenario_validity.value,
                    geometry_validity=candidate.geometry_validity.value,
                    diversity_key=candidate.diversity_key,
                    expiry=candidate.expiry,
                    payload=candidate.model_dump(mode="json"),
                )
                .on_conflict_do_nothing()
            )
            for stage in candidate.path_sequence:
                await self.session.execute(
                    insert(ScenarioPathStageRecord)
                    .values(
                        stage_id=stage.stage_id,
                        candidate_id=candidate.candidate_id,
                        sequence=stage.sequence,
                        payload=stage.model_dump(mode="json"),
                    )
                    .on_conflict_do_nothing()
                )
            for component in candidate.score_components:
                await self.session.execute(
                    insert(ScenarioScoreComponentRecord)
                    .values(
                        candidate_id=candidate.candidate_id,
                        name=component.name,
                        contribution=component.contribution,
                        payload=component.model_dump(mode="json"),
                    )
                    .on_conflict_do_nothing(index_elements=["candidate_id", "name"])
                )
        compact = selection.model_dump(
            mode="json", exclude={"primary", "alternative"}
        )
        await self.session.execute(
            insert(PrimaryScenarioSelectionRecord)
            .values(
                selection_id=selection.selection_id,
                simulation_cycle_id=selection.simulation_cycle_id,
                market_state_id=selection.market_state_id,
                primary_candidate_id=selection.primary_candidate_id,
                alternative_candidate_id=selection.alternative_candidate_id,
                instrument=selection.instrument,
                status=selection.status.value,
                authoritative_action=selection.authoritative_action.value,
                signal_eligible=selection.signal_eligible,
                market_cutoff=selection.market_cutoff,
                payload=compact,
                selected_at=selection.selected_at,
            )
            .on_conflict_do_nothing()
        )
        if selection.primary is not None and selection.primary.geometry is not None:
            await self.session.execute(
                insert(PrimaryScenarioGeometryRecord)
                .values(
                    selection_id=selection.selection_id,
                    candidate_id=selection.primary.candidate_id,
                    entry_type=selection.primary.entry_type.value,
                    payload=selection.primary.geometry.model_dump(mode="json"),
                )
                .on_conflict_do_nothing()
            )
        transition_id = uuid5(
            NAMESPACE_URL, f"ten:scenario-lifecycle:{selection.selection_id}:GENERATED"
        )
        await self.session.execute(
            insert(ScenarioLifecycleTransitionRecord)
            .values(
                transition_id=transition_id,
                selection_id=selection.selection_id,
                previous_status=None,
                new_status="GENERATED",
                reason="authoritative_m15_simulation_completed",
                transitioned_at=selection.selected_at,
            )
            .on_conflict_do_nothing(index_elements=["transition_id"])
        )
        await self.session.commit()
        return await self.for_state(selection.market_state_id) or selection

    @scoped_session
    async def for_state(self, market_state_id: UUID) -> PrimaryScenarioSelection | None:
        record = (
            await self.session.scalars(
                select(PrimaryScenarioSelectionRecord)
                .where(PrimaryScenarioSelectionRecord.market_state_id == market_state_id)
                .limit(1)
            )
        ).first()
        return await self._hydrate(record) if record else None

    @scoped_session
    async def latest(self, instrument: str) -> PrimaryScenarioSelection | None:
        record = (
            await self.session.scalars(
                select(PrimaryScenarioSelectionRecord)
                .where(PrimaryScenarioSelectionRecord.instrument == instrument)
                .order_by(
                    PrimaryScenarioSelectionRecord.market_cutoff.desc(),
                    PrimaryScenarioSelectionRecord.selection_id.desc(),
                )
                .limit(1)
            )
        ).first()
        return await self._hydrate(record) if record else None

    @scoped_session
    async def candidates(
        self, simulation_cycle_id: UUID
    ) -> tuple[CandidateMarketScenario, ...]:
        records = (
            await self.session.scalars(
                select(CandidateMarketScenarioRecord)
                .where(
                    CandidateMarketScenarioRecord.simulation_cycle_id
                    == simulation_cycle_id
                )
                .order_by(CandidateMarketScenarioRecord.rank)
            )
        ).all()
        return tuple(
            CandidateMarketScenario.model_validate(item.payload) for item in records
        )

    @scoped_session
    async def pending_primary_before(
        self, instrument: str, cutoff: datetime
    ) -> tuple[tuple[PrimaryScenarioSelection, CandidateMarketScenario], ...]:
        rows = (
            await self.session.execute(
                select(PrimaryScenarioSelectionRecord, CandidateMarketScenarioRecord)
                .join(
                    CandidateMarketScenarioRecord,
                    CandidateMarketScenarioRecord.candidate_id
                    == PrimaryScenarioSelectionRecord.primary_candidate_id,
                )
                .outerjoin(
                    CandidateScenarioOutcomeRecord,
                    CandidateScenarioOutcomeRecord.candidate_id
                    == CandidateMarketScenarioRecord.candidate_id,
                )
                .where(
                    PrimaryScenarioSelectionRecord.instrument == instrument,
                    CandidateMarketScenarioRecord.expiry <= cutoff,
                    CandidateScenarioOutcomeRecord.candidate_id.is_(None),
                )
            )
        ).all()
        values = []
        for selection_record, candidate_record in rows:
            selection = await self._hydrate(selection_record)
            candidate = CandidateMarketScenario.model_validate(candidate_record.payload)
            values.append((selection, candidate))
        return tuple(values)

    @scoped_session
    async def save_outcome(
        self, outcome: CandidateScenarioOutcome
    ) -> CandidateScenarioOutcome:
        await self.session.execute(
            insert(CandidateScenarioOutcomeRecord)
            .values(
                outcome_id=outcome.outcome_id,
                candidate_id=outcome.candidate_id,
                selection_id=outcome.selection_id,
                status=outcome.status,
                directional_accuracy=outcome.directional_accuracy,
                payload=outcome.model_dump(mode="json"),
                completed_at=outcome.completed_at,
            )
            .on_conflict_do_nothing(index_elements=["candidate_id"])
        )
        await self.session.commit()
        record = (
            await self.session.scalars(
                select(CandidateScenarioOutcomeRecord)
                .where(
                    CandidateScenarioOutcomeRecord.candidate_id
                    == outcome.candidate_id
                )
                .limit(1)
            )
        ).first()
        return CandidateScenarioOutcome.model_validate(record.payload) if record else outcome

    async def _hydrate(
        self, record: PrimaryScenarioSelectionRecord
    ) -> PrimaryScenarioSelection:
        identifiers = tuple(
            item
            for item in (record.primary_candidate_id, record.alternative_candidate_id)
            if item is not None
        )
        rows = (
            await self.session.scalars(
                select(CandidateMarketScenarioRecord).where(
                    CandidateMarketScenarioRecord.candidate_id.in_(identifiers)
                )
            )
        ).all() if identifiers else ()
        candidates = {
            item.candidate_id: CandidateMarketScenario.model_validate(item.payload)
            for item in rows
        }
        primary = (
            candidates.get(record.primary_candidate_id)
            if record.primary_candidate_id is not None
            else None
        )
        alternative = (
            candidates.get(record.alternative_candidate_id)
            if record.alternative_candidate_id is not None
            else None
        )
        return PrimaryScenarioSelection.model_validate(
            record.payload
            | {
                "primary": primary,
                "alternative": alternative,
            }
        )
