"""Persistence ports for AI reasoning, memory, and managed signals."""

from __future__ import annotations

import asyncio
from datetime import datetime
from hashlib import sha256
import json
import logging
from typing import Protocol

from pydantic import BaseModel
from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    AIMarketAnalysisRecord,
    AIAnalysisSignalOutcomeRecord,
    AIAnalysisSignalRecord,
    AIMarketForecastRecord,
    AIReasoningRequestRecord,
    AIReasoningCycleLockRecord,
    AIReasoningGateDecisionRecord,
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
from .gate import AIReasoningGateDecision
from .analysis import (
    AIAnalysisSignal,
    AIAnalysisSignalOutcome,
    AIMarketAnalysis,
    AnalysisStatus,
)
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


class AIArtifactConflictError(RuntimeError):
    """The same logical identity was presented with conflicting canonical data."""


def _canonical_hash(value: object, *, excluded: frozenset[str] = frozenset()) -> str:
    payload: object
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", exclude=set(excluded))
    else:
        payload = value
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def analysis_payload_hash(value: AIMarketAnalysis) -> str:
    """Exclude transport timing while preserving all analytical semantics."""

    payload = value.model_dump(
        mode="json",
        exclude={"created_at"},
    )
    metadata = dict(payload["provider_metadata"])
    metadata.pop("latency_ms", None)
    metadata.pop("token_usage", None)
    payload["provider_metadata"] = metadata
    return _canonical_hash(payload)


def analysis_signal_payload_hash(value: AIAnalysisSignal) -> str:
    return _canonical_hash(value, excluded=frozenset({"generated_at"}))


class AIReasoningRepository(Protocol):
    async def save_gate_decision(
        self, value: AIReasoningGateDecision
    ) -> AIReasoningGateDecision: ...

    async def latest_gate_decision(
        self,
        instrument: str,
        attempted_cutoff: datetime | None = None,
    ) -> AIReasoningGateDecision | None: ...

    async def claim_reasoning_cycle(
        self,
        idempotency_key: str,
        instrument: str,
        ums_boundary: datetime,
        cycle_version: str,
        provider_contract_version: str,
        claimed_at: datetime,
        *,
        analysis_timeframe: str | None = None,
        five_minute_window_start: datetime | None = None,
        market_state_hash: str | None = None,
        analysis_contract_version: str | None = None,
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
    async def analysis_for_reasoning_cycle(self, idempotency_key: str) -> AIMarketAnalysis | None: ...
    async def complete_analysis_cycle(
        self,
        idempotency_key: str,
        request_id: object | None,
        analysis_id: object | None,
        status: str,
        completed_at: datetime,
    ) -> None: ...
    async def analysis_for_market_state_hash(
        self,
        instrument: str,
        market_state_hash: str,
        analysis_contract_version: str,
    ) -> AIMarketAnalysis | None: ...
    async def save_analysis(self, value: AIMarketAnalysis) -> AIMarketAnalysis: ...
    async def save_analysis_signal(self, value: AIAnalysisSignal) -> AIAnalysisSignal: ...
    async def save_analysis_signal_outcome(
        self, value: AIAnalysisSignalOutcome
    ) -> AIAnalysisSignalOutcome: ...
    async def analysis_signal_outcome(
        self, signal_id: object
    ) -> AIAnalysisSignalOutcome | None: ...
    async def count_analysis_signal_outcomes(
        self, instrument: str
    ) -> tuple[int, int]: ...
    async def signal_for_analysis(self, analysis_id: object) -> AIAnalysisSignal | None: ...
    async def latest_analysis_signal(
        self,
        instrument: str,
        timeframe: str | None = None,
    ) -> AIAnalysisSignal | None: ...
    async def get_analysis_signal(self, signal_id: object) -> AIAnalysisSignal | None: ...
    async def latest_completed_analysis_cycle(
        self,
        instrument: str,
        timeframe: str | None = None,
    ) -> tuple[AIMarketAnalysis, AIAnalysisSignal] | None: ...
    async def list_analysis_signals(
        self,
        instrument: str,
        timeframe: str | None,
        start: datetime | None,
        end: datetime | None,
        direction: str | None,
        minimum_confidence: int | None,
        strength: str | None,
        offset: int,
        limit: int,
    ) -> tuple[AIAnalysisSignal, ...]: ...
    async def count_analysis_signals(
        self,
        instrument: str,
        timeframe: str | None = None,
    ) -> int: ...
    async def latest_analysis(self, instrument: str, timeframe: str | None = None) -> AIMarketAnalysis | None: ...
    async def analyses_before(
        self,
        instrument: str,
        timeframe: str,
        at: datetime,
        limit: int,
    ) -> tuple[AIMarketAnalysis, ...]: ...
    async def get_analysis(self, analysis_id: object) -> AIMarketAnalysis | None: ...
    async def analysis_for_state(self, market_state_id: object) -> AIMarketAnalysis | None: ...
    async def list_analyses(
        self,
        instrument: str,
        timeframe: str | None,
        start: datetime | None,
        end: datetime | None,
        status: AnalysisStatus | None,
        provider: str | None,
        offset: int,
        limit: int,
    ) -> tuple[AIMarketAnalysis, ...]: ...
    async def save_setup_family(self, value: SetupFamilyDefinition, registry_version: str) -> SetupFamilyDefinition: ...
    async def save_request(self, value: AIReasoningRequest) -> AIReasoningRequest: ...
    async def save_failure(self, value: LLMStructuredOutputFailure) -> LLMStructuredOutputFailure: ...
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
        self.analyses: dict[object, AIMarketAnalysis] = {}
        self.analysis_signals: dict[object, AIAnalysisSignal] = {}
        self.analysis_signal_outcomes: dict[object, AIAnalysisSignalOutcome] = {}
        self.proposals: dict[object, AISignalProposal] = {}
        self.signals: dict[object, ManagedSignal] = {}
        self.transitions: dict[object, SignalStateTransition] = {}
        self.revisions: dict[object, SignalLevelRevision] = {}
        self.monitoring: dict[object, SignalMonitoringEvaluation] = {}
        self.outcomes: dict[object, SignalOutcome] = {}
        self.memory: dict[object, MarketMemoryEntry] = {}
        self.reasoning_cycles: dict[str, dict[str, object]] = {}
        self.gate_decisions: dict[object, AIReasoningGateDecision] = {}
        self._lock = asyncio.Lock()

    async def save_gate_decision(
        self, value: AIReasoningGateDecision
    ) -> AIReasoningGateDecision:
        async with self._lock:
            self.gate_decisions[value.decision_id] = value
        return value

    async def latest_gate_decision(
        self,
        instrument: str,
        attempted_cutoff: datetime | None = None,
    ) -> AIReasoningGateDecision | None:
        async with self._lock:
            values = [
                item
                for item in self.gate_decisions.values()
                if item.instrument == instrument
                and (
                    attempted_cutoff is None
                    or item.attempted_cutoff == attempted_cutoff
                )
            ]
        return max(values, key=lambda item: item.created_at, default=None)

    async def claim_reasoning_cycle(
        self,
        idempotency_key: str,
        instrument: str,
        ums_boundary: datetime,
        cycle_version: str,
        provider_contract_version: str,
        claimed_at: datetime,
        *,
        analysis_timeframe: str | None = None,
        five_minute_window_start: datetime | None = None,
        market_state_hash: str | None = None,
        analysis_contract_version: str | None = None,
    ) -> bool:
        async with self._lock:
            duplicate = any(
                item.get("instrument") == instrument
                and (
                    item.get("idempotency_key") == idempotency_key
                    or (
                        analysis_timeframe is not None
                        and item.get("analysis_timeframe") == analysis_timeframe
                        and item.get("five_minute_window_start")
                        == five_minute_window_start
                        and item.get("analysis_contract_version")
                        == analysis_contract_version
                    )
                    or (
                        market_state_hash is not None
                        and item.get("market_state_hash") == market_state_hash
                        and item.get("analysis_contract_version")
                        == analysis_contract_version
                    )
                )
                for item in self.reasoning_cycles.values()
            )
            if idempotency_key in self.reasoning_cycles or duplicate:
                return False
            self.reasoning_cycles[idempotency_key] = {
                "idempotency_key": idempotency_key,
                "instrument": instrument,
                "ums_boundary": ums_boundary,
                "cycle_version": cycle_version,
                "provider_contract_version": provider_contract_version,
                "analysis_timeframe": analysis_timeframe,
                "five_minute_window_start": five_minute_window_start,
                "market_state_hash": market_state_hash,
                "analysis_contract_version": analysis_contract_version,
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
            if cycle is None or cycle.get("status") not in {
                "completed",
                "failed",
                "COMPLETED",
                "FAILED_PROVIDER",
                "FAILED_SCHEMA",
                "FAILED_PERSISTENCE",
                "TIMED_OUT",
            }:
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

    async def analysis_for_reasoning_cycle(self, idempotency_key: str) -> AIMarketAnalysis | None:
        async with self._lock:
            cycle = self.reasoning_cycles.get(idempotency_key)
            if cycle is None or cycle.get("status") not in {
                "completed",
                "failed",
                "COMPLETED",
                "FAILED_PROVIDER",
                "FAILED_SCHEMA",
                "FAILED_PERSISTENCE",
                "TIMED_OUT",
            }:
                return None
            return self.analyses.get(cycle.get("analysis_id"))

    async def complete_analysis_cycle(
        self,
        idempotency_key: str,
        request_id: object | None,
        analysis_id: object | None,
        status: str,
        completed_at: datetime,
    ) -> None:
        async with self._lock:
            self.reasoning_cycles[idempotency_key].update(
                request_id=request_id,
                analysis_id=analysis_id,
                status=status,
                completed_at=completed_at,
            )

    async def analysis_for_market_state_hash(
        self,
        instrument: str,
        market_state_hash: str,
        analysis_contract_version: str,
    ) -> AIMarketAnalysis | None:
        async with self._lock:
            cycle = next(
                (
                    item
                    for item in self.reasoning_cycles.values()
                    if item.get("instrument") == instrument
                    and item.get("market_state_hash") == market_state_hash
                    and item.get("analysis_contract_version")
                    == analysis_contract_version
                ),
                None,
            )
            return (
                self.analyses.get(cycle.get("analysis_id"))
                if cycle is not None
                else None
            )

    async def save_analysis(self, value: AIMarketAnalysis) -> AIMarketAnalysis:
        async with self._lock:
            existing = next(
                (
                    item
                    for item in self.analyses.values()
                    if item.request_id == value.request_id
                    or (
                        item.symbol == value.symbol
                        and item.timeframe == value.timeframe
                        and item.cycle_id == value.cycle_id
                        and item.schema_version == value.schema_version
                    )
                ),
                None,
            )
            if existing is not None:
                if analysis_payload_hash(existing) != analysis_payload_hash(value):
                    raise AIArtifactConflictError(
                        "conflicting AI analysis payload for stable logical identity"
                    )
                return existing
            self.analyses[value.analysis_id] = value
        return value

    async def save_analysis_signal(self, value: AIAnalysisSignal) -> AIAnalysisSignal:
        async with self._lock:
            existing = next(
                (
                    item
                    for item in self.analysis_signals.values()
                    if item.analysis_id == value.analysis_id
                    or (
                        item.instrument == value.instrument
                        and item.timeframe == value.timeframe
                        and item.cycle_id == value.cycle_id
                        and item.schema_version == value.schema_version
                    )
                ),
                None,
            )
            if existing is not None:
                if analysis_signal_payload_hash(existing) != analysis_signal_payload_hash(value):
                    raise AIArtifactConflictError(
                        "conflicting analysis signal for stable logical identity"
                    )
                return existing
            self.analysis_signals[value.signal_id] = value
            return value

    async def save_analysis_signal_outcome(
        self,
        value: AIAnalysisSignalOutcome,
    ) -> AIAnalysisSignalOutcome:
        async with self._lock:
            self.analysis_signal_outcomes[value.signal_id] = value
            return value

    async def analysis_signal_outcome(
        self,
        signal_id: object,
    ) -> AIAnalysisSignalOutcome | None:
        async with self._lock:
            return self.analysis_signal_outcomes.get(signal_id)

    async def count_analysis_signal_outcomes(
        self,
        instrument: str,
    ) -> tuple[int, int]:
        async with self._lock:
            signal_ids = {
                item.signal_id
                for item in self.analysis_signals.values()
                if item.instrument == instrument
            }
            values = [
                item
                for signal_id, item in self.analysis_signal_outcomes.items()
                if signal_id in signal_ids
            ]
        complete = sum(item.completed_at is not None for item in values)
        return len(values), complete

    async def signal_for_analysis(self, analysis_id: object) -> AIAnalysisSignal | None:
        async with self._lock:
            return next(
                (
                    item
                    for item in self.analysis_signals.values()
                    if item.analysis_id == analysis_id
                ),
                None,
            )

    async def latest_analysis_signal(
        self,
        instrument: str,
        timeframe: str | None = None,
    ) -> AIAnalysisSignal | None:
        async with self._lock:
            values = [
                item
                for item in self.analysis_signals.values()
                if item.instrument == instrument
                and (timeframe is None or item.timeframe == timeframe)
            ]
        return max(values, key=lambda item: item.generated_at, default=None)

    async def get_analysis_signal(self, signal_id: object) -> AIAnalysisSignal | None:
        async with self._lock:
            return self.analysis_signals.get(signal_id)

    async def latest_completed_analysis_cycle(
        self,
        instrument: str,
        timeframe: str | None = None,
    ) -> tuple[AIMarketAnalysis, AIAnalysisSignal] | None:
        async with self._lock:
            pairs = [
                (analysis, signal)
                for signal in self.analysis_signals.values()
                if signal.instrument == instrument
                and (timeframe is None or signal.timeframe == timeframe)
                and (analysis := self.analyses.get(signal.analysis_id)) is not None
                and analysis.status == AnalysisStatus.AVAILABLE
                and analysis.validation_passed
            ]
        return max(
            pairs,
            key=lambda pair: (
                pair[1].generated_at,
                pair[0].analysis_timestamp,
                str(pair[1].signal_id),
            ),
            default=None,
        )

    async def list_analysis_signals(
        self,
        instrument: str,
        timeframe: str | None,
        start: datetime | None,
        end: datetime | None,
        direction: str | None,
        minimum_confidence: int | None,
        strength: str | None,
        offset: int,
        limit: int,
    ) -> tuple[AIAnalysisSignal, ...]:
        async with self._lock:
            values = [
                item
                for item in self.analysis_signals.values()
                if item.instrument == instrument
                and (timeframe is None or item.timeframe == timeframe)
                and (start is None or item.generated_at >= start)
                and (end is None or item.generated_at <= end)
                and (direction is None or item.signal.value == direction)
                and (
                    minimum_confidence is None
                    or item.confidence >= minimum_confidence
                )
                and (strength is None or item.strength.value == strength)
            ]
        ordered = sorted(
            values,
            key=lambda item: (item.generated_at, str(item.signal_id)),
            reverse=True,
        )
        return tuple(ordered[offset : offset + limit])

    async def count_analysis_signals(
        self,
        instrument: str,
        timeframe: str | None = None,
    ) -> int:
        async with self._lock:
            return sum(
                1
                for item in self.analysis_signals.values()
                if item.instrument == instrument
                and (timeframe is None or item.timeframe == timeframe)
            )

    async def latest_analysis(
        self,
        instrument: str,
        timeframe: str | None = None,
    ) -> AIMarketAnalysis | None:
        async with self._lock:
            values = [
                item
                for item in self.analyses.values()
                if item.symbol == instrument
                and (timeframe is None or item.timeframe == timeframe)
                and item.status == AnalysisStatus.AVAILABLE
            ]
        return max(values, key=lambda item: item.analysis_timestamp, default=None)

    async def analyses_before(
        self,
        instrument: str,
        timeframe: str,
        at: datetime,
        limit: int,
    ) -> tuple[AIMarketAnalysis, ...]:
        async with self._lock:
            values = [
                item
                for item in self.analyses.values()
                if item.symbol == instrument
                and item.timeframe == timeframe
                and item.analysis_timestamp < at
                and item.status == AnalysisStatus.AVAILABLE
            ]
        return tuple(sorted(values, key=lambda item: item.analysis_timestamp, reverse=True)[:limit])

    async def get_analysis(self, analysis_id: object) -> AIMarketAnalysis | None:
        async with self._lock:
            return self.analyses.get(analysis_id)

    async def analysis_for_state(self, market_state_id: object) -> AIMarketAnalysis | None:
        async with self._lock:
            values = [
                item
                for item in self.analyses.values()
                if item.market_snapshot_id == market_state_id
            ]
        return max(values, key=lambda item: item.analysis_timestamp, default=None)

    async def list_analyses(
        self,
        instrument: str,
        timeframe: str | None,
        start: datetime | None,
        end: datetime | None,
        status: AnalysisStatus | None,
        provider: str | None,
        offset: int,
        limit: int,
    ) -> tuple[AIMarketAnalysis, ...]:
        async with self._lock:
            values = [
                item
                for item in self.analyses.values()
                if item.symbol == instrument
                and (timeframe is None or item.timeframe == timeframe)
                and (start is None or item.analysis_timestamp >= start)
                and (end is None or item.analysis_timestamp <= end)
                and (status is None or item.status == status)
                and (provider is None or item.provider_metadata.provider == provider)
            ]
        ordered = sorted(
            values,
            key=lambda item: (item.analysis_timestamp, str(item.analysis_id)),
            reverse=True,
        )
        return tuple(ordered[offset : offset + limit])

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
    async def save_gate_decision(
        self, value: AIReasoningGateDecision
    ) -> AIReasoningGateDecision:
        await self.session.execute(
            insert(AIReasoningGateDecisionRecord)
            .values(
                decision_id=value.decision_id,
                instrument=value.instrument,
                trigger_timeframe=value.trigger_timeframe,
                attempted_cutoff=value.attempted_cutoff,
                analysis_lookup_cutoff=value.analysis_lookup_cutoff,
                market_state_id=value.market_state_id,
                snapshot_id=value.snapshot_id,
                gate_decision=value.gate_decision,
                gate_skip_reason=value.gate_skip_reason,
                existing_analysis_id=value.existing_analysis_id,
                analysis_created_at=value.analysis_created_at,
                analysis_market_cutoff=value.analysis_market_cutoff,
                payload=value.model_dump(mode="json"),
                created_at=value.created_at,
            )
            .on_conflict_do_update(
                index_elements=["decision_id"],
                set_={
                    "gate_decision": value.gate_decision,
                    "gate_skip_reason": value.gate_skip_reason,
                    "existing_analysis_id": value.existing_analysis_id,
                    "analysis_created_at": value.analysis_created_at,
                    "analysis_market_cutoff": value.analysis_market_cutoff,
                    "payload": value.model_dump(mode="json"),
                    "created_at": value.created_at,
                },
            )
        )
        await self.session.commit()
        return value

    @scoped_session
    async def latest_gate_decision(
        self,
        instrument: str,
        attempted_cutoff: datetime | None = None,
    ) -> AIReasoningGateDecision | None:
        query = select(AIReasoningGateDecisionRecord).where(
            AIReasoningGateDecisionRecord.instrument == instrument
        )
        if attempted_cutoff is not None:
            query = query.where(
                AIReasoningGateDecisionRecord.attempted_cutoff
                == attempted_cutoff
            )
        record = (
            await self.session.scalars(
                query.order_by(AIReasoningGateDecisionRecord.created_at.desc())
                .limit(1)
            )
        ).first()
        return (
            AIReasoningGateDecision.model_validate(record.payload)
            if record is not None
            else None
        )

    @scoped_session
    async def claim_reasoning_cycle(
        self,
        idempotency_key: str,
        instrument: str,
        ums_boundary: datetime,
        cycle_version: str,
        provider_contract_version: str,
        claimed_at: datetime,
        *,
        analysis_timeframe: str | None = None,
        five_minute_window_start: datetime | None = None,
        market_state_hash: str | None = None,
        analysis_contract_version: str | None = None,
    ) -> bool:
        statement = (
            insert(AIReasoningCycleLockRecord)
            .values(
                idempotency_key=idempotency_key,
                instrument=instrument,
                ums_boundary=ums_boundary,
                cycle_version=cycle_version,
                provider_contract_version=provider_contract_version,
                analysis_timeframe=analysis_timeframe,
                five_minute_window_start=five_minute_window_start,
                market_state_hash=market_state_hash,
                analysis_contract_version=analysis_contract_version,
                status="claimed",
                claimed_at=claimed_at,
            )
            .on_conflict_do_nothing()
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
                    AIReasoningCycleLockRecord.status.in_(
                        (
                            "completed",
                            "failed",
                            "COMPLETED",
                            "FAILED_PROVIDER",
                            "FAILED_SCHEMA",
                            "FAILED_PERSISTENCE",
                            "TIMED_OUT",
                        )
                    ),
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
    async def analysis_for_reasoning_cycle(self, idempotency_key: str) -> AIMarketAnalysis | None:
        lock = (
            await self.session.scalars(
                select(AIReasoningCycleLockRecord)
                .where(
                    AIReasoningCycleLockRecord.idempotency_key == idempotency_key,
                    AIReasoningCycleLockRecord.status.in_(
                        (
                            "completed",
                            "failed",
                            "COMPLETED",
                            "FAILED_PROVIDER",
                            "FAILED_SCHEMA",
                            "FAILED_PERSISTENCE",
                            "TIMED_OUT",
                        )
                    ),
                )
                .limit(1)
            )
        ).first()
        if lock is None or lock.analysis_id is None:
            return None
        record = await self.session.get(AIMarketAnalysisRecord, lock.analysis_id)
        return AIMarketAnalysis.model_validate(record.payload) if record is not None else None

    @scoped_session
    async def complete_analysis_cycle(
        self,
        idempotency_key: str,
        request_id: object | None,
        analysis_id: object | None,
        status: str,
        completed_at: datetime,
    ) -> None:
        await self.session.execute(
            update(AIReasoningCycleLockRecord)
            .where(AIReasoningCycleLockRecord.idempotency_key == idempotency_key)
            .values(
                request_id=request_id,
                analysis_id=analysis_id,
                status=status,
                completed_at=completed_at,
            )
        )
        await self.session.commit()

    @scoped_session
    async def analysis_for_market_state_hash(
        self,
        instrument: str,
        market_state_hash: str,
        analysis_contract_version: str,
    ) -> AIMarketAnalysis | None:
        lock = (
            await self.session.scalars(
                select(AIReasoningCycleLockRecord)
                .where(
                    AIReasoningCycleLockRecord.instrument == instrument,
                    AIReasoningCycleLockRecord.market_state_hash
                    == market_state_hash,
                    AIReasoningCycleLockRecord.analysis_contract_version
                    == analysis_contract_version,
                )
                .order_by(AIReasoningCycleLockRecord.claimed_at.desc())
                .limit(1)
            )
        ).first()
        if lock is None or lock.analysis_id is None:
            return None
        record = await self.session.get(AIMarketAnalysisRecord, lock.analysis_id)
        return AIMarketAnalysis.model_validate(record.payload) if record else None

    @scoped_session
    async def save_analysis(self, value: AIMarketAnalysis) -> AIMarketAnalysis:
        statement = (
            insert(AIMarketAnalysisRecord)
            .values(
                analysis_id=value.analysis_id,
                request_id=value.request_id,
                cycle_id=value.cycle_id,
                market_snapshot_id=value.market_snapshot_id,
                quantitative_forecast_id=value.quantitative_forecast_id,
                symbol=value.symbol,
                timeframe=value.timeframe,
                analysis_timestamp=value.analysis_timestamp,
                status=value.status.value,
                schema_version=value.schema_version,
                provider=value.provider_metadata.provider,
                validation_passed=value.validation_passed,
                payload=value.model_dump(mode="json"),
                created_at=value.created_at,
            )
            .on_conflict_do_nothing()
            .returning(AIMarketAnalysisRecord.analysis_id)
        )
        identifier = (await self.session.execute(statement)).scalar_one_or_none()
        if identifier is None:
            record = (
                await self.session.scalars(
                    select(AIMarketAnalysisRecord)
                    .where(
                        or_(
                            AIMarketAnalysisRecord.request_id == value.request_id,
                            (
                                (AIMarketAnalysisRecord.symbol == value.symbol)
                                & (AIMarketAnalysisRecord.timeframe == value.timeframe)
                                & (AIMarketAnalysisRecord.cycle_id == value.cycle_id)
                                & (
                                    AIMarketAnalysisRecord.schema_version
                                    == value.schema_version
                                )
                            ),
                        )
                    )
                    .limit(1)
                )
            ).first()
            if record is None:
                await self.session.rollback()
                raise RuntimeError("AI analysis conflict did not resolve to a persisted row")
            existing = AIMarketAnalysis.model_validate(record.payload)
            existing_hash = analysis_payload_hash(existing)
            incoming_hash = analysis_payload_hash(value)
            if existing_hash != incoming_hash:
                logger.error(
                    "analytical.persistence.non_deterministic_duplicate",
                    extra={
                        "artifact_type": "ai_market_analysis",
                        "analysis_id": str(value.analysis_id),
                        "cycle_id": str(value.cycle_id),
                        "snapshot_id": str(value.market_snapshot_id),
                        "existing_payload_hash": existing_hash,
                        "incoming_payload_hash": incoming_hash,
                    },
                )
                await self.session.rollback()
                raise AIArtifactConflictError(
                    "conflicting AI analysis payload for stable logical identity"
                )
            await self.session.commit()
            return existing
        await self.session.commit()
        return value

    @scoped_session
    async def save_analysis_signal(self, value: AIAnalysisSignal) -> AIAnalysisSignal:
        payload_hash = analysis_signal_payload_hash(value)
        identifier = (
            await self.session.execute(
                insert(AIAnalysisSignalRecord)
                .values(
                    signal_id=value.signal_id,
                    analysis_id=value.analysis_id,
                    cycle_id=value.cycle_id,
                    snapshot_id=value.snapshot_id,
                    instrument=value.instrument,
                    timeframe=value.timeframe,
                    signal=value.signal.value,
                    confidence=value.confidence,
                    strength=value.strength.value,
                    schema_version=value.schema_version,
                    payload_hash=payload_hash,
                    payload=value.model_dump(mode="json"),
                    generated_at=value.generated_at,
                )
                .on_conflict_do_nothing()
                .returning(AIAnalysisSignalRecord.signal_id)
            )
        ).scalar_one_or_none()
        if identifier is None:
            record = (
                await self.session.scalars(
                    select(AIAnalysisSignalRecord)
                    .where(
                        or_(
                            AIAnalysisSignalRecord.analysis_id == value.analysis_id,
                            (
                                (AIAnalysisSignalRecord.instrument == value.instrument)
                                & (AIAnalysisSignalRecord.timeframe == value.timeframe)
                                & (AIAnalysisSignalRecord.cycle_id == value.cycle_id)
                                & (AIAnalysisSignalRecord.schema_version == value.schema_version)
                            ),
                        )
                    )
                    .limit(1)
                )
            ).first()
            if record is None:
                await self.session.rollback()
                raise RuntimeError("analysis-signal conflict did not resolve")
            if record.payload_hash != payload_hash:
                logger.error(
                    "analytical.persistence.non_deterministic_duplicate",
                    extra={
                        "artifact_type": "ai_analysis_signal",
                        "analysis_id": str(value.analysis_id),
                        "signal_id": str(value.signal_id),
                        "cycle_id": str(value.cycle_id),
                        "snapshot_id": str(value.snapshot_id),
                        "existing_payload_hash": record.payload_hash,
                        "incoming_payload_hash": payload_hash,
                    },
                )
                await self.session.rollback()
                raise AIArtifactConflictError(
                    "conflicting analysis signal for stable logical identity"
                )
            await self.session.commit()
            return AIAnalysisSignal.model_validate(record.payload)
        await self.session.commit()
        return value

    @scoped_session
    async def save_analysis_signal_outcome(
        self,
        value: AIAnalysisSignalOutcome,
    ) -> AIAnalysisSignalOutcome:
        await self.session.execute(
            insert(AIAnalysisSignalOutcomeRecord)
            .values(
                outcome_id=value.outcome_id,
                signal_id=value.signal_id,
                status=value.status.value,
                entry_reached=value.entry_reached,
                payload=value.model_dump(mode="json"),
                evaluated_at=value.evaluated_at,
                completed_at=value.completed_at,
            )
            .on_conflict_do_update(
                index_elements=["signal_id"],
                set_={
                    "status": value.status.value,
                    "entry_reached": value.entry_reached,
                    "payload": value.model_dump(mode="json"),
                    "evaluated_at": value.evaluated_at,
                    "completed_at": value.completed_at,
                },
            )
        )
        await self.session.commit()
        return value

    @scoped_session
    async def analysis_signal_outcome(
        self,
        signal_id: object,
    ) -> AIAnalysisSignalOutcome | None:
        record = (
            await self.session.scalars(
                select(AIAnalysisSignalOutcomeRecord)
                .where(AIAnalysisSignalOutcomeRecord.signal_id == signal_id)
                .limit(1)
            )
        ).first()
        return (
            AIAnalysisSignalOutcome.model_validate(record.payload)
            if record is not None
            else None
        )

    @scoped_session
    async def count_analysis_signal_outcomes(
        self,
        instrument: str,
    ) -> tuple[int, int]:
        base = (
            select(AIAnalysisSignalOutcomeRecord.completed_at)
            .join(
                AIAnalysisSignalRecord,
                AIAnalysisSignalRecord.signal_id
                == AIAnalysisSignalOutcomeRecord.signal_id,
            )
            .where(AIAnalysisSignalRecord.instrument == instrument)
        )
        completed_values = list((await self.session.scalars(base)).all())
        return (
            len(completed_values),
            sum(value is not None for value in completed_values),
        )

    @scoped_session
    async def signal_for_analysis(self, analysis_id: object) -> AIAnalysisSignal | None:
        record = (
            await self.session.scalars(
                select(AIAnalysisSignalRecord)
                .where(AIAnalysisSignalRecord.analysis_id == analysis_id)
                .limit(1)
            )
        ).first()
        return AIAnalysisSignal.model_validate(record.payload) if record else None

    @scoped_session
    async def latest_analysis_signal(
        self,
        instrument: str,
        timeframe: str | None = None,
    ) -> AIAnalysisSignal | None:
        query = select(AIAnalysisSignalRecord).where(
            AIAnalysisSignalRecord.instrument == instrument
        )
        if timeframe is not None:
            query = query.where(AIAnalysisSignalRecord.timeframe == timeframe)
        record = (
            await self.session.scalars(
                query.order_by(
                    AIAnalysisSignalRecord.generated_at.desc(),
                    AIAnalysisSignalRecord.signal_id.desc(),
                ).limit(1)
            )
        ).first()
        return AIAnalysisSignal.model_validate(record.payload) if record else None

    @scoped_session
    async def get_analysis_signal(self, signal_id: object) -> AIAnalysisSignal | None:
        record = await self.session.get(AIAnalysisSignalRecord, signal_id)
        return AIAnalysisSignal.model_validate(record.payload) if record else None

    @scoped_session
    async def latest_completed_analysis_cycle(
        self,
        instrument: str,
        timeframe: str | None = None,
    ) -> tuple[AIMarketAnalysis, AIAnalysisSignal] | None:
        query = (
            select(AIMarketAnalysisRecord, AIAnalysisSignalRecord)
            .join(
                AIAnalysisSignalRecord,
                AIAnalysisSignalRecord.analysis_id
                == AIMarketAnalysisRecord.analysis_id,
            )
            .where(
                AIMarketAnalysisRecord.symbol == instrument,
                AIMarketAnalysisRecord.status == AnalysisStatus.AVAILABLE.value,
                AIMarketAnalysisRecord.validation_passed.is_(True),
            )
        )
        if timeframe is not None:
            query = query.where(AIAnalysisSignalRecord.timeframe == timeframe)
        row = (
            await self.session.execute(
                query.order_by(
                    AIAnalysisSignalRecord.generated_at.desc(),
                    AIMarketAnalysisRecord.analysis_timestamp.desc(),
                    AIAnalysisSignalRecord.signal_id.desc(),
                ).limit(1)
            )
        ).first()
        if row is None:
            return None
        return (
            AIMarketAnalysis.model_validate(row[0].payload),
            AIAnalysisSignal.model_validate(row[1].payload),
        )

    @scoped_session
    async def list_analysis_signals(
        self,
        instrument: str,
        timeframe: str | None,
        start: datetime | None,
        end: datetime | None,
        direction: str | None,
        minimum_confidence: int | None,
        strength: str | None,
        offset: int,
        limit: int,
    ) -> tuple[AIAnalysisSignal, ...]:
        query = select(AIAnalysisSignalRecord).where(
            AIAnalysisSignalRecord.instrument == instrument
        )
        if timeframe is not None:
            query = query.where(AIAnalysisSignalRecord.timeframe == timeframe)
        if start is not None:
            query = query.where(AIAnalysisSignalRecord.generated_at >= start)
        if end is not None:
            query = query.where(AIAnalysisSignalRecord.generated_at <= end)
        if direction is not None:
            query = query.where(AIAnalysisSignalRecord.signal == direction)
        if minimum_confidence is not None:
            query = query.where(
                AIAnalysisSignalRecord.confidence >= minimum_confidence
            )
        if strength is not None:
            query = query.where(AIAnalysisSignalRecord.strength == strength)
        records = (
            await self.session.scalars(
                query.order_by(
                    AIAnalysisSignalRecord.generated_at.desc(),
                    AIAnalysisSignalRecord.signal_id.desc(),
                )
                .offset(offset)
                .limit(limit)
            )
        ).all()
        return tuple(
            AIAnalysisSignal.model_validate(item.payload) for item in records
        )

    @scoped_session
    async def count_analysis_signals(
        self,
        instrument: str,
        timeframe: str | None = None,
    ) -> int:
        query = select(func.count(AIAnalysisSignalRecord.signal_id)).where(
            AIAnalysisSignalRecord.instrument == instrument
        )
        if timeframe is not None:
            query = query.where(AIAnalysisSignalRecord.timeframe == timeframe)
        return int(await self.session.scalar(query) or 0)

    @scoped_session
    async def latest_analysis(
        self,
        instrument: str,
        timeframe: str | None = None,
    ) -> AIMarketAnalysis | None:
        query = select(AIMarketAnalysisRecord).where(
            AIMarketAnalysisRecord.symbol == instrument,
            AIMarketAnalysisRecord.status == AnalysisStatus.AVAILABLE.value,
        )
        if timeframe is not None:
            query = query.where(AIMarketAnalysisRecord.timeframe == timeframe)
        record = (
            await self.session.scalars(
                query.order_by(
                    AIMarketAnalysisRecord.analysis_timestamp.desc(),
                    AIMarketAnalysisRecord.analysis_id.desc(),
                ).limit(1)
            )
        ).first()
        return AIMarketAnalysis.model_validate(record.payload) if record is not None else None

    @scoped_session
    async def analyses_before(
        self,
        instrument: str,
        timeframe: str,
        at: datetime,
        limit: int,
    ) -> tuple[AIMarketAnalysis, ...]:
        records = (
            await self.session.scalars(
                select(AIMarketAnalysisRecord)
                .where(
                    AIMarketAnalysisRecord.symbol == instrument,
                    AIMarketAnalysisRecord.timeframe == timeframe,
                    AIMarketAnalysisRecord.analysis_timestamp < at,
                    AIMarketAnalysisRecord.status == AnalysisStatus.AVAILABLE.value,
                )
                .order_by(AIMarketAnalysisRecord.analysis_timestamp.desc())
                .limit(limit)
            )
        ).all()
        return tuple(AIMarketAnalysis.model_validate(item.payload) for item in records)

    @scoped_session
    async def get_analysis(self, analysis_id: object) -> AIMarketAnalysis | None:
        record = await self.session.get(AIMarketAnalysisRecord, analysis_id)
        return AIMarketAnalysis.model_validate(record.payload) if record is not None else None

    @scoped_session
    async def analysis_for_state(self, market_state_id: object) -> AIMarketAnalysis | None:
        record = (
            await self.session.scalars(
                select(AIMarketAnalysisRecord)
                .where(AIMarketAnalysisRecord.market_snapshot_id == market_state_id)
                .order_by(AIMarketAnalysisRecord.analysis_timestamp.desc())
                .limit(1)
            )
        ).first()
        return AIMarketAnalysis.model_validate(record.payload) if record is not None else None

    @scoped_session
    async def list_analyses(
        self,
        instrument: str,
        timeframe: str | None,
        start: datetime | None,
        end: datetime | None,
        status: AnalysisStatus | None,
        provider: str | None,
        offset: int,
        limit: int,
    ) -> tuple[AIMarketAnalysis, ...]:
        query = select(AIMarketAnalysisRecord).where(
            AIMarketAnalysisRecord.symbol == instrument
        )
        if timeframe is not None:
            query = query.where(AIMarketAnalysisRecord.timeframe == timeframe)
        if start is not None:
            query = query.where(AIMarketAnalysisRecord.analysis_timestamp >= start)
        if end is not None:
            query = query.where(AIMarketAnalysisRecord.analysis_timestamp <= end)
        if status is not None:
            query = query.where(AIMarketAnalysisRecord.status == status.value)
        if provider is not None:
            query = query.where(AIMarketAnalysisRecord.provider == provider)
        records = (
            await self.session.scalars(
                query.order_by(
                    AIMarketAnalysisRecord.analysis_timestamp.desc(),
                    AIMarketAnalysisRecord.analysis_id.desc(),
                )
                .offset(offset)
                .limit(limit)
            )
        ).all()
        return tuple(AIMarketAnalysis.model_validate(item.payload) for item in records)

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
