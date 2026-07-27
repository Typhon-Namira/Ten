"""Persistence ports for AI reasoning, memory, and managed signals."""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    AIForecastEvidenceLinkRecord,
    AIForecastScenarioRecord,
    AIMarketForecastRecord,
    AIReasoningRequestRecord,
    AIReasoningCycleLockRecord,
    AISetupFamilyVersionRecord,
    AISignalProposalRecord,
    LLMStructuredOutputFailureRecord,
    ManagedSignalRecord,
    ManagedSignalOutcomeRecord,
    MarketMemoryEntryRecord,
    SignalLevelRevisionRecord,
    SignalMonitoringEvaluationRecord,
    SignalStateTransitionRecord,
)
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .llm_context import build_llm_analysis_context
from .models import (
    AIMarketForecast,
    AIReasoningRequest,
    AISignalProposal,
    LLMStructuredOutputFailure,
    ManagedSignal,
    MarketMemoryEntry,
    SignalLevelRevision,
    SignalMonitoringEvaluation,
    SignalOutcome,
    SignalStateTransition,
)
from .setup_families import SetupFamilyDefinition
from .request_persistence import (
    PersistedAIReasoningRequest,
    decode_persisted_request,
    persisted_request_from_domain,
    persisted_request_payload,
)

logger = logging.getLogger(__name__)


class AIReasoningRepository(Protocol):
    async def claim_reasoning_cycle(
        self,
        idempotency_key: str,
        instrument: str,
        ums_boundary: datetime,
        cycle_version: str,
        provider_contract_version: str,
        claimed_at: datetime,
    ) -> bool: ...
    async def complete_reasoning_cycle(
        self,
        idempotency_key: str,
        request_id: object,
        forecast_id: object,
        status: str,
        completed_at: datetime,
    ) -> None: ...
    async def result_for_reasoning_cycle(
        self,
        idempotency_key: str,
    ) -> tuple[AIMarketForecast, AISignalProposal | None] | None: ...
    async def save_setup_family(self, value: SetupFamilyDefinition, registry_version: str) -> SetupFamilyDefinition: ...
    async def save_request(self, value: AIReasoningRequest) -> AIReasoningRequest: ...
    async def save_failure(self, value: LLMStructuredOutputFailure) -> LLMStructuredOutputFailure: ...
    async def save_forecast(self, value: AIMarketForecast) -> AIMarketForecast: ...
    async def save_proposal(self, value: AISignalProposal) -> AISignalProposal: ...
    async def latest_forecast(self, instrument: str) -> AIMarketForecast | None: ...
    async def latest_proposal(self) -> AISignalProposal | None: ...
    async def request_for_state(self, market_state_id: object) -> PersistedAIReasoningRequest | None: ...
    async def forecast_for_state(self, market_state_id: object) -> AIMarketForecast | None: ...
    async def proposal_for_state(self, market_state_id: object) -> AISignalProposal | None: ...
    async def signal_by_opportunity(self, opportunity_key: str) -> ManagedSignal | None: ...
    async def active_signals(self, instrument: str) -> tuple[ManagedSignal, ...]: ...
    async def save_signal(self, value: ManagedSignal) -> ManagedSignal: ...
    async def save_transition(self, value: SignalStateTransition) -> SignalStateTransition: ...
    async def save_revision(self, value: SignalLevelRevision) -> SignalLevelRevision: ...
    async def save_monitoring(self, value: SignalMonitoringEvaluation) -> SignalMonitoringEvaluation: ...
    async def save_signal_outcome(self, value: SignalOutcome) -> SignalOutcome: ...
    async def append_memory(self, value: MarketMemoryEntry) -> MarketMemoryEntry: ...
    async def recent_memory(self, instrument: str, limit: int) -> tuple[MarketMemoryEntry, ...]: ...
    async def signal_history(self, signal_id: object) -> dict[str, tuple[object, ...]]: ...


class InMemoryAIReasoningRepository:
    def __init__(self) -> None:
        self.setup_families: dict[tuple[str, str], SetupFamilyDefinition] = {}
        self.requests: dict[object, AIReasoningRequest] = {}
        self.failures: dict[object, LLMStructuredOutputFailure] = {}
        self.forecasts: dict[object, AIMarketForecast] = {}
        self.proposals: dict[object, AISignalProposal] = {}
        self.signals: dict[object, ManagedSignal] = {}
        self.transitions: dict[object, SignalStateTransition] = {}
        self.revisions: dict[object, SignalLevelRevision] = {}
        self.monitoring: dict[object, SignalMonitoringEvaluation] = {}
        self.outcomes: dict[object, SignalOutcome] = {}
        self.memory: dict[object, MarketMemoryEntry] = {}
        self.reasoning_cycles: dict[str, dict[str, object]] = {}
        self._lock = asyncio.Lock()

    async def claim_reasoning_cycle(
        self,
        idempotency_key: str,
        instrument: str,
        ums_boundary: datetime,
        cycle_version: str,
        provider_contract_version: str,
        claimed_at: datetime,
    ) -> bool:
        async with self._lock:
            if idempotency_key in self.reasoning_cycles:
                return False
            self.reasoning_cycles[idempotency_key] = {
                "instrument": instrument,
                "ums_boundary": ums_boundary,
                "cycle_version": cycle_version,
                "provider_contract_version": provider_contract_version,
                "status": "claimed",
                "claimed_at": claimed_at,
            }
            return True

    async def complete_reasoning_cycle(
        self,
        idempotency_key: str,
        request_id: object,
        forecast_id: object,
        status: str,
        completed_at: datetime,
    ) -> None:
        async with self._lock:
            cycle = self.reasoning_cycles[idempotency_key]
            cycle.update(
                request_id=request_id,
                forecast_id=forecast_id,
                status=status,
                completed_at=completed_at,
            )

    async def result_for_reasoning_cycle(
        self,
        idempotency_key: str,
    ) -> tuple[AIMarketForecast, AISignalProposal | None] | None:
        async with self._lock:
            cycle = self.reasoning_cycles.get(idempotency_key)
            if cycle is None or cycle.get("status") not in {"completed", "failed"}:
                return None
            request_id = cycle.get("request_id")
            forecast = self.forecasts.get(request_id)
            if forecast is None:
                return None
            proposal = next(
                (
                    item
                    for item in self.proposals.values()
                    if item.forecast_id == forecast.forecast_id
                ),
                None,
            )
            return forecast, proposal

    async def save_setup_family(self, value: SetupFamilyDefinition, registry_version: str) -> SetupFamilyDefinition:
        async with self._lock:
            self.setup_families[(value.setup_family_id, value.version)] = value
        return value

    async def save_request(self, value: AIReasoningRequest) -> AIReasoningRequest:
        async with self._lock:
            self.requests[value.request_id] = value
        return value

    async def save_failure(self, value: LLMStructuredOutputFailure) -> LLMStructuredOutputFailure:
        async with self._lock:
            self.failures[value.failure_id] = value
        return value

    async def save_forecast(self, value: AIMarketForecast) -> AIMarketForecast:
        async with self._lock:
            existing = self.forecasts.get(value.request_id)
            if existing is not None:
                return existing
            self.forecasts[value.request_id] = value
            return value

    async def save_proposal(self, value: AISignalProposal) -> AISignalProposal:
        async with self._lock:
            self.proposals[value.proposal_id] = value
        return value

    async def latest_forecast(self, instrument: str) -> AIMarketForecast | None:
        async with self._lock:
            values = list(self.forecasts.values())
        return max(values, key=lambda item: (item.generated_at, str(item.forecast_id)), default=None)

    async def latest_proposal(self) -> AISignalProposal | None:
        async with self._lock:
            values = list(self.proposals.values())
        return max(values, key=lambda item: (item.created_at, str(item.proposal_id)), default=None)

    async def request_for_state(self, market_state_id: object) -> PersistedAIReasoningRequest | None:
        async with self._lock:
            values = [item for item in self.requests.values() if item.market_state_id == market_state_id]
        request = max(values, key=lambda item: (item.created_at, str(item.request_id)), default=None)
        if request is None:
            return None
        return persisted_request_from_domain(request)

    async def forecast_for_state(self, market_state_id: object) -> AIMarketForecast | None:
        async with self._lock:
            values = [item for item in self.forecasts.values() if item.market_state_id == market_state_id]
        return max(values, key=lambda item: (item.generated_at, str(item.forecast_id)), default=None)

    async def proposal_for_state(self, market_state_id: object) -> AISignalProposal | None:
        async with self._lock:
            values = [item for item in self.proposals.values() if item.market_state_id == market_state_id]
        return max(values, key=lambda item: (item.created_at, str(item.proposal_id)), default=None)

    async def signal_by_opportunity(self, opportunity_key: str) -> ManagedSignal | None:
        async with self._lock:
            return next((item for item in self.signals.values() if item.structural_opportunity_key == opportunity_key), None)

    async def active_signals(self, instrument: str) -> tuple[ManagedSignal, ...]:
        terminal = {"closed", "cancelled", "invalidated", "expired", "stopped"}
        async with self._lock:
            values = [item for item in self.signals.values() if item.instrument == instrument and item.state.value not in terminal]
        return tuple(sorted(values, key=lambda item: item.updated_at, reverse=True))

    async def save_signal(self, value: ManagedSignal) -> ManagedSignal:
        async with self._lock:
            existing_key = next(
                (key for key, item in self.signals.items() if item.structural_opportunity_key == value.structural_opportunity_key),
                None,
            )
            if existing_key is not None:
                existing = self.signals[existing_key]
                if existing.signal_id != value.signal_id:
                    return existing
                self.signals[existing_key] = value
                return value
            self.signals[value.structural_opportunity_key] = value
            return value

    async def save_transition(self, value: SignalStateTransition) -> SignalStateTransition:
        async with self._lock:
            self.transitions[value.transition_id] = value
        return value

    async def save_revision(self, value: SignalLevelRevision) -> SignalLevelRevision:
        async with self._lock:
            self.revisions[value.revision_id] = value
        return value

    async def save_monitoring(self, value: SignalMonitoringEvaluation) -> SignalMonitoringEvaluation:
        async with self._lock:
            self.monitoring[value.evaluation_id] = value
        return value

    async def save_signal_outcome(self, value: SignalOutcome) -> SignalOutcome:
        async with self._lock:
            self.outcomes[value.signal_id] = value
        return value

    async def append_memory(self, value: MarketMemoryEntry) -> MarketMemoryEntry:
        async with self._lock:
            self.memory[value.entry_id] = value
        return value

    async def recent_memory(self, instrument: str, limit: int) -> tuple[MarketMemoryEntry, ...]:
        async with self._lock:
            values = [item for item in self.memory.values() if item.instrument == instrument]
        return tuple(sorted(values, key=lambda item: item.occurred_at, reverse=True)[:limit])

    async def signal_history(self, signal_id: object) -> dict[str, tuple[object, ...]]:
        async with self._lock:
            transitions = tuple(item for item in self.transitions.values() if item.signal_id == signal_id)
            revisions = tuple(item for item in self.revisions.values() if item.signal_id == signal_id)
            monitoring = tuple(item for item in self.monitoring.values() if item.signal_id == signal_id)
            outcomes = tuple(item for item in self.outcomes.values() if item.signal_id == signal_id)
        return {"transitions": transitions, "revisions": revisions, "monitoring": monitoring, "outcomes": outcomes}


class SqlAlchemyAIReasoningRepository(ScopedSessionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        super().__init__(session_factory)

    @scoped_session
    async def claim_reasoning_cycle(
        self,
        idempotency_key: str,
        instrument: str,
        ums_boundary: datetime,
        cycle_version: str,
        provider_contract_version: str,
        claimed_at: datetime,
    ) -> bool:
        statement = (
            insert(AIReasoningCycleLockRecord)
            .values(
                idempotency_key=idempotency_key,
                instrument=instrument,
                ums_boundary=ums_boundary,
                cycle_version=cycle_version,
                provider_contract_version=provider_contract_version,
                status="claimed",
                claimed_at=claimed_at,
            )
            .on_conflict_do_nothing(index_elements=["idempotency_key"])
            .returning(AIReasoningCycleLockRecord.idempotency_key)
        )
        claimed = (await self.session.execute(statement)).scalar_one_or_none() is not None
        await self.session.commit()
        return claimed

    @scoped_session
    async def complete_reasoning_cycle(
        self,
        idempotency_key: str,
        request_id: object,
        forecast_id: object,
        status: str,
        completed_at: datetime,
    ) -> None:
        await self.session.execute(
            update(AIReasoningCycleLockRecord)
            .where(AIReasoningCycleLockRecord.idempotency_key == idempotency_key)
            .values(
                request_id=request_id,
                forecast_id=forecast_id,
                status=status,
                completed_at=completed_at,
            )
        )
        await self.session.commit()

    @scoped_session
    async def result_for_reasoning_cycle(
        self,
        idempotency_key: str,
    ) -> tuple[AIMarketForecast, AISignalProposal | None] | None:
        window = (
            await self.session.scalars(
                select(AIReasoningCycleLockRecord)
                .where(
                    AIReasoningCycleLockRecord.idempotency_key == idempotency_key,
                    AIReasoningCycleLockRecord.status.in_(("completed", "failed")),
                )
                .limit(1)
            )
        ).first()
        if window is None or window.forecast_id is None:
            return None
        forecast_record = (
            await self.session.scalars(
                select(AIMarketForecastRecord)
                .where(AIMarketForecastRecord.forecast_id == window.forecast_id)
                .limit(1)
            )
        ).first()
        if forecast_record is None:
            return None
        proposal_record = (
            await self.session.scalars(
                select(AISignalProposalRecord)
                .where(AISignalProposalRecord.forecast_id == window.forecast_id)
                .order_by(AISignalProposalRecord.created_at.desc())
                .limit(1)
            )
        ).first()
        return (
            AIMarketForecast.model_validate(forecast_record.payload),
            AISignalProposal.model_validate(proposal_record.payload)
            if proposal_record is not None
            else None,
        )

    @scoped_session
    async def save_setup_family(self, value: SetupFamilyDefinition, registry_version: str) -> SetupFamilyDefinition:
        await self.session.execute(
            insert(AISetupFamilyVersionRecord)
            .values(setup_family_id=value.setup_family_id, version=value.version, registry_version=registry_version, payload=value.model_dump(mode="json"))
            .on_conflict_do_nothing(index_elements=["setup_family_id", "version"])
        )
        await self.session.commit()
        return value

    @scoped_session
    async def save_request(self, value: AIReasoningRequest) -> AIReasoningRequest:
        context = build_llm_analysis_context(value)
        await self.session.execute(
            insert(AIReasoningRequestRecord)
            .values(
                request_id=value.request_id,
                cycle_id=value.cycle_id,
                market_state_id=value.market_state_id,
                quantitative_forecast_id=value.quantitative_forecast_id,
                instrument=value.instrument,
                analysis_timestamp=value.analysis_timestamp,
                prompt_version=value.prompt_version,
                model_identifier=value.model_identifier,
                payload=persisted_request_payload(value, context),
                created_at=value.created_at,
            )
            .on_conflict_do_nothing(index_elements=["request_id"])
        )
        await self.session.commit()
        return value

    @scoped_session
    async def save_failure(self, value: LLMStructuredOutputFailure) -> LLMStructuredOutputFailure:
        await self.session.execute(
            insert(LLMStructuredOutputFailureRecord)
            .values(
                failure_id=value.failure_id,
                request_id=value.request_id,
                attempt=value.attempt,
                model_identifier=value.model_identifier,
                failure_state=value.failure_state,
                payload=value.model_dump(mode="json"),
                created_at=value.created_at,
            )
            .on_conflict_do_nothing(index_elements=["failure_id"])
        )
        await self.session.commit()
        return value

    @scoped_session
    async def save_forecast(self, value: AIMarketForecast) -> AIMarketForecast:
        statement = (
            insert(AIMarketForecastRecord)
            .values(
                forecast_id=value.forecast_id,
                request_id=value.request_id,
                market_state_id=value.market_state_id,
                quantitative_forecast_id=value.quantitative_forecast_id,
                status=value.status.value,
                dominant_direction=value.dominant_direction.value if value.dominant_direction else None,
                selected_setup_family=value.selected_setup_family,
                payload=value.model_dump(mode="json"),
                generated_at=value.generated_at,
            )
            .on_conflict_do_nothing(index_elements=["request_id"])
            .returning(AIMarketForecastRecord.forecast_id)
        )
        forecast_id = (await self.session.execute(statement)).scalar_one_or_none()
        persisted = value
        if forecast_id is None:
            record = (
                await self.session.scalars(
                    select(AIMarketForecastRecord).where(AIMarketForecastRecord.request_id == value.request_id).limit(1)
                )
            ).first()
            if record is None:
                await self.session.rollback()
                raise RuntimeError("AI forecast conflict did not resolve to a persisted row")
            forecast_id = record.forecast_id
            persisted = AIMarketForecast.model_validate(record.payload)
        for ordinal, scenario in enumerate(persisted.alternative_scenarios):
            await self.session.execute(
                insert(AIForecastScenarioRecord)
                .values(forecast_id=forecast_id, ordinal=ordinal, scenario_name=scenario.name, payload=scenario.model_dump(mode="json"))
                .on_conflict_do_nothing(index_elements=["forecast_id", "ordinal"])
            )
        for role, evidence_ids in (("supporting", persisted.supporting_evidence_ids), ("contradicting", persisted.contradicting_evidence_ids)):
            for evidence_id in evidence_ids:
                await self.session.execute(
                    insert(AIForecastEvidenceLinkRecord)
                    .values(forecast_id=forecast_id, evidence_id=evidence_id, role=role)
                    .on_conflict_do_nothing(index_elements=["forecast_id", "evidence_id", "role"])
                )
        await self.session.commit()
        return persisted

    @scoped_session
    async def save_proposal(self, value: AISignalProposal) -> AISignalProposal:
        await self.session.execute(
            insert(AISignalProposalRecord)
            .values(
                proposal_id=value.proposal_id,
                forecast_id=value.forecast_id,
                market_state_id=value.market_state_id,
                structural_opportunity_key=value.structural_opportunity_key,
                recommended_action=value.recommended_action.value,
                payload=value.model_dump(mode="json"),
                created_at=value.created_at,
            )
            .on_conflict_do_nothing(index_elements=["proposal_id"])
        )
        await self.session.commit()
        return value

    @scoped_session
    async def latest_forecast(self, instrument: str) -> AIMarketForecast | None:
        query = (
            select(AIMarketForecastRecord)
            .join(AIReasoningRequestRecord, AIMarketForecastRecord.request_id == AIReasoningRequestRecord.request_id)
            .where(AIReasoningRequestRecord.instrument == instrument)
            .order_by(AIMarketForecastRecord.generated_at.desc())
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        return AIMarketForecast.model_validate(record.payload) if record else None

    @scoped_session
    async def latest_proposal(self) -> AISignalProposal | None:
        record = (await self.session.scalars(select(AISignalProposalRecord).order_by(AISignalProposalRecord.created_at.desc()).limit(1))).first()
        return AISignalProposal.model_validate(record.payload) if record else None

    @scoped_session
    async def request_for_state(self, market_state_id: object) -> PersistedAIReasoningRequest | None:
        query = (
            select(AIReasoningRequestRecord)
            .where(AIReasoningRequestRecord.market_state_id == market_state_id)
            .order_by(AIReasoningRequestRecord.created_at.desc())
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        if record is None:
            return None
        decoded = decode_persisted_request(record)
        if decoded.compatibility_status == "incompatible":
            logger.warning(
                "ai_reasoning.request_payload.incompatible",
                extra={
                    "request_id": str(record.request_id),
                    "cycle_id": str(record.cycle_id),
                    "market_state_id": str(record.market_state_id),
                    "payload_format": decoded.payload_format,
                    "payload_schema_version": decoded.payload_schema_version,
                    "reason": decoded.compatibility_reason,
                    "payload_keys": sorted(record.payload),
                },
            )
        return decoded

    @scoped_session
    async def forecast_for_state(self, market_state_id: object) -> AIMarketForecast | None:
        query = (
            select(AIMarketForecastRecord)
            .where(AIMarketForecastRecord.market_state_id == market_state_id)
            .order_by(AIMarketForecastRecord.generated_at.desc(), AIMarketForecastRecord.forecast_id.desc())
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        return AIMarketForecast.model_validate(record.payload) if record else None

    @scoped_session
    async def proposal_for_state(self, market_state_id: object) -> AISignalProposal | None:
        query = (
            select(AISignalProposalRecord)
            .where(AISignalProposalRecord.market_state_id == market_state_id)
            .order_by(AISignalProposalRecord.created_at.desc(), AISignalProposalRecord.proposal_id.desc())
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        return AISignalProposal.model_validate(record.payload) if record else None

    @scoped_session
    async def signal_by_opportunity(self, opportunity_key: str) -> ManagedSignal | None:
        record = (await self.session.scalars(select(ManagedSignalRecord).where(ManagedSignalRecord.structural_opportunity_key == opportunity_key).limit(1))).first()
        return ManagedSignal.model_validate(record.payload) if record else None

    @scoped_session
    async def active_signals(self, instrument: str) -> tuple[ManagedSignal, ...]:
        terminal = ("closed", "cancelled", "invalidated", "expired", "stopped")
        records = (await self.session.scalars(select(ManagedSignalRecord).where(ManagedSignalRecord.instrument == instrument, ManagedSignalRecord.state.not_in(terminal)).order_by(ManagedSignalRecord.updated_at.desc()))).all()
        return tuple(ManagedSignal.model_validate(record.payload) for record in records)

    @scoped_session
    async def save_signal(self, value: ManagedSignal) -> ManagedSignal:
        payload = value.model_dump(mode="json")
        statement = (
            insert(ManagedSignalRecord)
            .values(
                signal_id=value.signal_id,
                instrument=value.instrument,
                structural_opportunity_key=value.structural_opportunity_key,
                setup_family=value.setup_family,
                direction=value.direction.value,
                state=value.state.value,
                current_proposal_id=value.current_proposal_id,
                payload=payload,
                updated_at=value.updated_at,
            )
            .on_conflict_do_nothing(index_elements=["structural_opportunity_key"])
            .returning(ManagedSignalRecord.payload)
        )
        persisted_payload = (await self.session.execute(statement)).scalar_one_or_none()
        if persisted_payload is None:
            record = (
                await self.session.scalars(
                    select(ManagedSignalRecord)
                    .where(ManagedSignalRecord.structural_opportunity_key == value.structural_opportunity_key)
                    .limit(1)
                )
            ).first()
            if record is None:
                await self.session.rollback()
                raise RuntimeError("managed-signal conflict did not resolve to a persisted row")
            if record.signal_id != value.signal_id:
                await self.session.commit()
                return ManagedSignal.model_validate(record.payload)
            await self.session.execute(
                update(ManagedSignalRecord)
                .where(ManagedSignalRecord.signal_id == value.signal_id)
                .values(
                    state=value.state.value,
                    current_proposal_id=value.current_proposal_id,
                    payload=payload,
                    updated_at=value.updated_at,
                )
            )
            persisted_payload = payload
        await self.session.commit()
        return ManagedSignal.model_validate(persisted_payload)

    async def _save_simple(self, record_type: type, values: dict[str, object], conflict: str) -> None:
        await self.session.execute(insert(record_type).values(**values).on_conflict_do_nothing(index_elements=[conflict]))
        await self.session.commit()

    @scoped_session
    async def save_transition(self, value: SignalStateTransition) -> SignalStateTransition:
        await self._save_simple(SignalStateTransitionRecord, {"transition_id": value.transition_id, "signal_id": value.signal_id, "previous_state": value.previous_state.value, "new_state": value.new_state.value, "payload": value.model_dump(mode="json"), "created_at": value.created_at}, "transition_id")
        return value

    @scoped_session
    async def save_revision(self, value: SignalLevelRevision) -> SignalLevelRevision:
        await self._save_simple(SignalLevelRevisionRecord, {"revision_id": value.revision_id, "signal_id": value.signal_id, "level_type": value.level_type, "payload": value.model_dump(mode="json"), "created_at": value.created_at}, "revision_id")
        return value

    @scoped_session
    async def save_monitoring(self, value: SignalMonitoringEvaluation) -> SignalMonitoringEvaluation:
        await self._save_simple(SignalMonitoringEvaluationRecord, {"evaluation_id": value.evaluation_id, "signal_id": value.signal_id, "forecast_id": value.forecast_id, "thesis_valid": value.thesis_valid, "recommended_action": value.recommended_action.value, "payload": value.model_dump(mode="json"), "evaluated_at": value.evaluated_at}, "evaluation_id")
        return value

    @scoped_session
    async def save_signal_outcome(self, value: SignalOutcome) -> SignalOutcome:
        await self._save_simple(ManagedSignalOutcomeRecord, {"outcome_id": value.outcome_id, "signal_id": value.signal_id, "final_state": value.final_state.value, "payload": value.model_dump(mode="json"), "closed_at": value.closed_at}, "signal_id")
        return value

    @scoped_session
    async def append_memory(self, value: MarketMemoryEntry) -> MarketMemoryEntry:
        await self._save_simple(MarketMemoryEntryRecord, {"entry_id": value.entry_id, "instrument": value.instrument, "cycle_id": value.cycle_id, "market_state_id": value.market_state_id, "category": value.category, "opportunity_key": value.opportunity_key, "signal_id": value.signal_id, "payload": value.model_dump(mode="json"), "occurred_at": value.occurred_at}, "entry_id")
        return value

    @scoped_session
    async def recent_memory(self, instrument: str, limit: int) -> tuple[MarketMemoryEntry, ...]:
        records = (await self.session.scalars(select(MarketMemoryEntryRecord).where(MarketMemoryEntryRecord.instrument == instrument).order_by(MarketMemoryEntryRecord.occurred_at.desc()).limit(limit))).all()
        return tuple(MarketMemoryEntry.model_validate(record.payload) for record in records)

    @scoped_session
    async def signal_history(self, signal_id: object) -> dict[str, tuple[object, ...]]:
        transitions = (await self.session.scalars(select(SignalStateTransitionRecord).where(SignalStateTransitionRecord.signal_id == signal_id).order_by(SignalStateTransitionRecord.created_at))).all()
        revisions = (await self.session.scalars(select(SignalLevelRevisionRecord).where(SignalLevelRevisionRecord.signal_id == signal_id).order_by(SignalLevelRevisionRecord.created_at))).all()
        monitoring = (await self.session.scalars(select(SignalMonitoringEvaluationRecord).where(SignalMonitoringEvaluationRecord.signal_id == signal_id).order_by(SignalMonitoringEvaluationRecord.evaluated_at))).all()
        outcomes = (await self.session.scalars(select(ManagedSignalOutcomeRecord).where(ManagedSignalOutcomeRecord.signal_id == signal_id).order_by(ManagedSignalOutcomeRecord.closed_at))).all()
        return {
            "transitions": tuple(SignalStateTransition.model_validate(item.payload) for item in transitions),
            "revisions": tuple(SignalLevelRevision.model_validate(item.payload) for item in revisions),
            "monitoring": tuple(SignalMonitoringEvaluation.model_validate(item.payload) for item in monitoring),
            "outcomes": tuple(SignalOutcome.model_validate(item.payload) for item in outcomes),
        }
