"""Idempotent normalized persistence for market simulations and selections."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    CandidateScenarioOutcomeRecord,
    CandidateMarketScenarioRecord,
    AuthoritativeSimulationAttemptRecord,
    MarketSimulationCycleRecord,
    PrimaryScenarioGeometryRecord,
    PrimaryScenarioSelectionRecord,
    ScenarioLifecycleTransitionRecord,
    ScenarioPathStageRecord,
    ScenarioScoreComponentRecord,
)
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .simulation_models import (
    AuthoritativeSimulationAttempt,
    CandidateMarketScenario,
    CandidateScenarioOutcome,
    MarketSimulationCycle,
    PrimaryScenarioSelection,
    SimulationAttemptStatus,
)


class MarketSimulationRepository(Protocol):
    async def save_attempt(
        self, attempt: AuthoritativeSimulationAttempt
    ) -> AuthoritativeSimulationAttempt: ...

    async def claim_attempt(
        self,
        attempt: AuthoritativeSimulationAttempt,
        *,
        started_at: datetime,
    ) -> tuple[AuthoritativeSimulationAttempt, bool]: ...

    async def attempt_at_cutoff(
        self, instrument: str, market_cutoff: datetime
    ) -> AuthoritativeSimulationAttempt | None: ...

    async def latest_attempt(
        self, instrument: str
    ) -> AuthoritativeSimulationAttempt | None: ...

    async def recent_attempts(
        self, instrument: str, limit: int = 16
    ) -> tuple[AuthoritativeSimulationAttempt, ...]: ...

    async def at_cutoff(
        self, instrument: str, market_cutoff: datetime
    ) -> PrimaryScenarioSelection | None: ...

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
        self.attempts: dict[UUID, AuthoritativeSimulationAttempt] = {}
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

    async def save_attempt(
        self, attempt: AuthoritativeSimulationAttempt
    ) -> AuthoritativeSimulationAttempt:
        async with self._lock:
            existing = self.attempts.get(attempt.attempt_id)
            if (
                existing is not None
                and existing.status.terminal
                and not (
                    existing.status == SimulationAttemptStatus.BLOCKED
                    and existing.failure_type
                    in {
                        "AI_ANALYSIS_MISSING",
                        "AI_ANALYSIS_PENDING",
                        "SIMULATION_NOT_INVOKED",
                    }
                )
            ):
                return existing
            self.attempts[attempt.attempt_id] = attempt
            return attempt

    async def claim_attempt(
        self,
        attempt: AuthoritativeSimulationAttempt,
        *,
        started_at: datetime,
    ) -> tuple[AuthoritativeSimulationAttempt, bool]:
        async with self._lock:
            existing = self.attempts.get(attempt.attempt_id)
            if existing is not None and (
                existing.status == SimulationAttemptStatus.RUNNING
                or (
                    existing.status.terminal
                    and not (
                        existing.status == SimulationAttemptStatus.BLOCKED
                        and existing.failure_type
                        in {
                            "AI_ANALYSIS_MISSING",
                            "AI_ANALYSIS_PENDING",
                            "SIMULATION_NOT_INVOKED",
                        }
                    )
                )
            ):
                return existing, False
            source = existing or attempt
            running = attempt.model_copy(
                update={
                    "status": SimulationAttemptStatus.RUNNING,
                    "started_at": started_at,
                    "retry_count": source.retry_count,
                    "completed_at": None,
                    "failure_stage": None,
                    "failure_type": None,
                    "failure_message": None,
                }
            )
            self.attempts[attempt.attempt_id] = running
            return running, True

    async def attempt_at_cutoff(
        self, instrument: str, market_cutoff: datetime
    ) -> AuthoritativeSimulationAttempt | None:
        async with self._lock:
            return next(
                (
                    item
                    for item in self.attempts.values()
                    if item.instrument == instrument
                    and item.market_cutoff == market_cutoff
                ),
                None,
            )

    async def latest_attempt(
        self, instrument: str
    ) -> AuthoritativeSimulationAttempt | None:
        values = await self.recent_attempts(instrument, 1)
        return values[0] if values else None

    async def recent_attempts(
        self, instrument: str, limit: int = 16
    ) -> tuple[AuthoritativeSimulationAttempt, ...]:
        async with self._lock:
            values = sorted(
                (
                    item
                    for item in self.attempts.values()
                    if item.instrument == instrument
                ),
                key=lambda item: (item.market_cutoff, item.server_time),
                reverse=True,
            )
            return tuple(values[:limit])

    async def at_cutoff(
        self, instrument: str, market_cutoff: datetime
    ) -> PrimaryScenarioSelection | None:
        async with self._lock:
            return next(
                (
                    item
                    for item in self.selections.values()
                    if item.instrument == instrument
                    and item.market_cutoff == market_cutoff
                ),
                None,
            )

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
    async def save_attempt(
        self, attempt: AuthoritativeSimulationAttempt
    ) -> AuthoritativeSimulationAttempt:
        values = {
            "attempt_id": attempt.attempt_id,
            "correlation_id": attempt.correlation_id,
            "instrument": attempt.instrument,
            "timeframe": attempt.timeframe,
            "market_cutoff": attempt.market_cutoff,
            "simulation_version": attempt.simulation_version,
            "status": attempt.status.value,
            "payload": attempt.model_dump(mode="json"),
            "scheduled_at": attempt.scheduled_at,
            "started_at": attempt.started_at,
            "completed_at": attempt.completed_at,
            "candidate_count": attempt.candidate_count,
            "simulation_cycle_id": attempt.simulation_cycle_id,
            "primary_scenario_id": attempt.primary_scenario_id,
            "alternative_scenario_id": attempt.alternative_scenario_id,
            "failure_stage": attempt.failure_stage,
            "failure_type": attempt.failure_type,
            "failure_message": attempt.failure_message,
            "skip_reason": attempt.skip_reason,
            "retry_count": attempt.retry_count,
            "market_state_id": attempt.market_state_id,
            "snapshot_id": attempt.snapshot_id,
            "quantitative_forecast_id": attempt.quantitative_forecast_id,
            "ai_analysis_id": attempt.ai_analysis_id,
            "ai_analysis_cutoff": attempt.ai_analysis_cutoff,
            "ai_analysis_committed_at": attempt.ai_analysis_committed_at,
            "dependency_lookup_result": attempt.dependency_lookup_result,
        }
        await self.session.execute(
            insert(AuthoritativeSimulationAttemptRecord)
            .values(**values)
            .on_conflict_do_update(
                index_elements=[
                    "instrument",
                    "timeframe",
                    "market_cutoff",
                    "simulation_version",
                ],
                set_={
                    key: value
                    for key, value in values.items()
                    if key
                    not in {
                        "attempt_id",
                        "instrument",
                        "timeframe",
                        "market_cutoff",
                        "simulation_version",
                        "scheduled_at",
                    }
                },
                where=AuthoritativeSimulationAttemptRecord.status.in_(
                    (
                        "SCHEDULED",
                        "WAITING_FOR_AI_ANALYSIS",
                        "RUNNING",
                        "FAILED",
                    )
                ),
            )
        )
        await self.session.commit()
        record = await self.session.get(
            AuthoritativeSimulationAttemptRecord, attempt.attempt_id
        )
        return (
            AuthoritativeSimulationAttempt.model_validate(record.payload)
            if record is not None
            else attempt
        )

    @scoped_session
    async def claim_attempt(
        self,
        attempt: AuthoritativeSimulationAttempt,
        *,
        started_at: datetime,
    ) -> tuple[AuthoritativeSimulationAttempt, bool]:
        await self.session.execute(
            insert(AuthoritativeSimulationAttemptRecord)
            .values(
                attempt_id=attempt.attempt_id,
                correlation_id=attempt.correlation_id,
                instrument=attempt.instrument,
                timeframe=attempt.timeframe,
                market_cutoff=attempt.market_cutoff,
                simulation_version=attempt.simulation_version,
                status=attempt.status.value,
                payload=attempt.model_dump(mode="json"),
                scheduled_at=attempt.scheduled_at,
                candidate_count=0,
                retry_count=attempt.retry_count,
                market_state_id=attempt.market_state_id,
                snapshot_id=attempt.snapshot_id,
                quantitative_forecast_id=attempt.quantitative_forecast_id,
                ai_analysis_id=attempt.ai_analysis_id,
                ai_analysis_cutoff=attempt.ai_analysis_cutoff,
                ai_analysis_committed_at=attempt.ai_analysis_committed_at,
                dependency_lookup_result=attempt.dependency_lookup_result,
            )
            .on_conflict_do_nothing()
        )
        running = attempt.model_copy(
            update={
                "status": SimulationAttemptStatus.RUNNING,
                "started_at": started_at,
                "completed_at": None,
                "failure_stage": None,
                "failure_type": None,
                "failure_message": None,
            }
        )
        claimed = (
            await self.session.execute(
                update(AuthoritativeSimulationAttemptRecord)
                .where(
                    AuthoritativeSimulationAttemptRecord.attempt_id
                    == attempt.attempt_id,
                    or_(
                        AuthoritativeSimulationAttemptRecord.status.in_(
                            (
                                "SCHEDULED",
                                "WAITING_FOR_AI_ANALYSIS",
                                "FAILED",
                            )
                        ),
                        (
                            AuthoritativeSimulationAttemptRecord.status
                            == "BLOCKED"
                        )
                        & AuthoritativeSimulationAttemptRecord.failure_type.in_(
                            (
                                "AI_ANALYSIS_MISSING",
                                "AI_ANALYSIS_PENDING",
                                "SIMULATION_NOT_INVOKED",
                            )
                        ),
                    ),
                )
                .values(
                    status=SimulationAttemptStatus.RUNNING.value,
                    payload=running.model_dump(mode="json"),
                    started_at=started_at,
                    completed_at=None,
                    failure_stage=None,
                    failure_type=None,
                    failure_message=None,
                    correlation_id=running.correlation_id,
                    market_state_id=running.market_state_id,
                    snapshot_id=running.snapshot_id,
                    quantitative_forecast_id=running.quantitative_forecast_id,
                    ai_analysis_id=running.ai_analysis_id,
                    ai_analysis_cutoff=running.ai_analysis_cutoff,
                    ai_analysis_committed_at=running.ai_analysis_committed_at,
                    dependency_lookup_result=running.dependency_lookup_result,
                )
                .returning(AuthoritativeSimulationAttemptRecord.attempt_id)
            )
        ).scalar_one_or_none()
        await self.session.commit()
        record = await self.session.get(
            AuthoritativeSimulationAttemptRecord, attempt.attempt_id
        )
        current = (
            AuthoritativeSimulationAttempt.model_validate(record.payload)
            if record is not None
            else running
        )
        return current, claimed is not None

    @scoped_session
    async def attempt_at_cutoff(
        self, instrument: str, market_cutoff: datetime
    ) -> AuthoritativeSimulationAttempt | None:
        record = (
            await self.session.scalars(
                select(AuthoritativeSimulationAttemptRecord)
                .where(
                    AuthoritativeSimulationAttemptRecord.instrument == instrument,
                    AuthoritativeSimulationAttemptRecord.market_cutoff
                    == market_cutoff,
                )
                .order_by(
                    AuthoritativeSimulationAttemptRecord.scheduled_at.desc()
                )
                .limit(1)
            )
        ).first()
        return (
            AuthoritativeSimulationAttempt.model_validate(record.payload)
            if record is not None
            else None
        )

    @scoped_session
    async def latest_attempt(
        self, instrument: str
    ) -> AuthoritativeSimulationAttempt | None:
        values = await self.recent_attempts(instrument, 1)
        return values[0] if values else None

    @scoped_session
    async def recent_attempts(
        self, instrument: str, limit: int = 16
    ) -> tuple[AuthoritativeSimulationAttempt, ...]:
        records = (
            await self.session.scalars(
                select(AuthoritativeSimulationAttemptRecord)
                .where(AuthoritativeSimulationAttemptRecord.instrument == instrument)
                .order_by(
                    AuthoritativeSimulationAttemptRecord.market_cutoff.desc(),
                    AuthoritativeSimulationAttemptRecord.scheduled_at.desc(),
                )
                .limit(limit)
            )
        ).all()
        return tuple(
            AuthoritativeSimulationAttempt.model_validate(item.payload)
            for item in records
        )

    @scoped_session
    async def at_cutoff(
        self, instrument: str, market_cutoff: datetime
    ) -> PrimaryScenarioSelection | None:
        record = (
            await self.session.scalars(
                select(PrimaryScenarioSelectionRecord)
                .where(
                    PrimaryScenarioSelectionRecord.instrument == instrument,
                    PrimaryScenarioSelectionRecord.market_cutoff == market_cutoff,
                )
                .limit(1)
            )
        ).first()
        return await self._hydrate(record) if record else None

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
