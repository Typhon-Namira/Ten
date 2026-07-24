"""Strict Phase 3/4 AI reasoning, proposal, memory, and lifecycle contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ImmutableAIModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AIResultStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    FAILED = "failed"
    NON_ACTIONABLE = "non_actionable"


class Direction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class ProposalAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"
    HOLD_EXISTING_SIGNAL = "HOLD_EXISTING_SIGNAL"
    CANCEL_SETUP = "CANCEL_SETUP"
    INVALIDATE_SIGNAL = "INVALIDATE_SIGNAL"
    ADJUST_ENTRY = "ADJUST_ENTRY"
    REDUCE_RISK = "REDUCE_RISK"
    TAKE_PARTIAL_PROFIT = "TAKE_PARTIAL_PROFIT"
    CLOSE_SIGNAL = "CLOSE_SIGNAL"


class ManagedSignalState(StrEnum):
    DETECTED = "detected"
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    WAITING_FOR_ENTRY = "waiting_for_entry"
    ACTIVE = "active"
    PARTIALLY_REALIZED = "partially_realized"
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    CLOSED = "closed"
    TEMPORARILY_BLOCKED = "temporarily_blocked"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    STOPPED = "stopped"


class SetupReadiness(StrEnum):
    NOT_READY = "not_ready"
    DEVELOPING = "developing"
    READY = "ready"
    ACTIVE = "active"


class EntryZone(ImmutableAIModel):
    low: float = Field(gt=0)
    high: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> EntryZone:
        if self.low > self.high:
            raise ValueError("entry zone must be ordered")
        return self


class MarketMemorySummary(ImmutableAIModel):
    schema_version: str = "1.0"
    entry_count: int = Field(ge=0)
    window_started_at: datetime | None = None
    window_ended_at: datetime | None = None
    regime_transitions: tuple[str, ...] = ()
    structure_changes: tuple[str, ...] = ()
    liquidity_events: tuple[str, ...] = ()
    forecast_changes: tuple[str, ...] = ()
    evidence_changes: tuple[str, ...] = ()
    signal_state_changes: tuple[str, ...] = ()
    completed_outcomes: tuple[str, ...] = ()
    repeated_model_mistakes: tuple[str, ...] = ()
    active_opportunity_key: str | None = None
    active_signal_state: str | None = None
    previous_levels: dict[str, Any] = Field(default_factory=dict)
    session_context: str | None = None

    @field_validator("window_started_at", "window_ended_at")
    @classmethod
    def aware_optional(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("memory timestamps must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None


class AIReasoningRequest(ImmutableAIModel):
    schema_version: str = "1.0"
    request_id: UUID
    cycle_id: UUID
    market_state_id: UUID
    quantitative_forecast_id: UUID
    instrument: str
    analysis_timestamp: datetime
    knowledge_cutoff: datetime
    trigger_timeframe: str
    current_price: float = Field(gt=0)
    supported_timeframe_states: tuple[dict[str, Any], ...]
    market_regime: tuple[dict[str, Any], ...] = ()
    trend_evidence: tuple[dict[str, Any], ...] = ()
    volatility_evidence: tuple[dict[str, Any], ...] = ()
    momentum_evidence: tuple[dict[str, Any], ...] = ()
    structure_evidence: tuple[dict[str, Any], ...] = ()
    smc_evidence: tuple[dict[str, Any], ...] = ()
    bos_choch_mss_evidence: tuple[dict[str, Any], ...] = ()
    liquidity_pools: tuple[dict[str, Any], ...] = ()
    liquidity_sweeps_and_raids: tuple[dict[str, Any], ...] = ()
    order_blocks: tuple[dict[str, Any], ...] = ()
    fair_value_gaps: tuple[dict[str, Any], ...] = ()
    displacement_evidence: tuple[dict[str, Any], ...] = ()
    volume_profile_evidence: tuple[dict[str, Any], ...] = ()
    poc_hvn_lvn_evidence: tuple[dict[str, Any], ...] = ()
    institutional_flow_evidence: tuple[dict[str, Any], ...] = ()
    session_context: tuple[dict[str, Any], ...] = ()
    spread: float | None = Field(default=None, ge=0)
    economic_event_context: tuple[dict[str, Any], ...] = ()
    data_quality_summary: dict[str, Any]
    missing_evidence: tuple[UUID, ...] = ()
    degraded_evidence: tuple[UUID, ...] = ()
    stale_evidence: tuple[UUID, ...] = ()
    quantitative_probabilities: dict[str, float | None]
    expected_return: float | None = None
    expected_movement: dict[str, float | None]
    expected_volatility: float | None = Field(default=None, ge=0)
    expected_favorable_excursion: float | None = Field(default=None, ge=0)
    expected_adverse_excursion: float | None = Field(default=None, ge=0)
    tp_probabilities: dict[str, float | None]
    sl_before_tp_probability: float | None = Field(default=None, ge=0, le=1)
    market_memory: MarketMemorySummary
    existing_signal_state: dict[str, Any] | None = None
    previous_ai_forecast: dict[str, Any] | None = None
    previous_ai_proposal: dict[str, Any] | None = None
    prompt_version: str
    reasoning_policy_version: str
    setup_family_registry_version: str
    model_identifier: str
    quantitative_model_version: str
    feature_schema_version: str
    market_state_schema_version: str
    created_at: datetime

    @field_validator("analysis_timestamp", "knowledge_cutoff", "created_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("AI reasoning timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def point_in_time_integrity(self) -> AIReasoningRequest:
        if self.analysis_timestamp > self.knowledge_cutoff or self.created_at < self.analysis_timestamp:
            raise ValueError("AI request violates its point-in-time boundary")
        probabilities = [value for value in self.quantitative_probabilities.values() if value is not None]
        if probabilities and (any(value < 0 or value > 1 for value in probabilities) or abs(sum(probabilities) - 1) > 1e-8):
            raise ValueError("quantitative probabilities must be valid and sum to one")
        return self


class AlternativeScenario(ImmutableAIModel):
    name: str
    probability: float = Field(ge=0, le=1)
    direction: Direction
    confirmation_conditions: tuple[str, ...] = ()


class AIMarketForecast(ImmutableAIModel):
    schema_version: str = "1.0"
    forecast_id: UUID
    request_id: UUID
    market_state_id: UUID
    quantitative_forecast_id: UUID
    cycle_id: UUID
    status: AIResultStatus
    dominant_direction: Direction | None = None
    buy_probability: float | None = Field(default=None, ge=0, le=1)
    sell_probability: float | None = Field(default=None, ge=0, le=1)
    neutral_probability: float | None = Field(default=None, ge=0, le=1)
    expected_horizon: str | None = None
    expected_minimum_move: float | None = Field(default=None, ge=0)
    expected_base_move: float | None = Field(default=None, ge=0)
    expected_maximum_move: float | None = Field(default=None, ge=0)
    expected_volatility: float | None = Field(default=None, ge=0)
    dominant_scenario: str | None = None
    dominant_scenario_probability: float | None = Field(default=None, ge=0, le=1)
    alternative_scenarios: tuple[AlternativeScenario, ...] = ()
    selected_setup_family: str | None = None
    setup_family_candidates: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[UUID, ...] = ()
    contradicting_evidence_ids: tuple[UUID, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    evidence_completeness: float | None = Field(default=None, ge=0, le=1)
    evidence_agreement: float | None = Field(default=None, ge=0, le=1)
    forecast_confidence: float | None = Field(default=None, ge=0, le=1)
    execution_confidence: float | None = Field(default=None, ge=0, le=1)
    risk_quality: float | None = Field(default=None, ge=0, le=1)
    setup_readiness: SetupReadiness | None = None
    uncertainty: float | None = Field(default=None, ge=0, le=1)
    reasoning_summary: str | None = None
    monitoring_conditions: tuple[str, ...] = ()
    model_provider: str
    model_identifier: str
    prompt_version: str
    reasoning_policy_version: str
    setup_family_registry_version: str
    quantitative_model_version: str
    feature_schema_version: str
    market_state_schema_version: str
    latency_ms: float | None = Field(default=None, ge=0)
    validation_passed: bool
    retry_count: int = Field(ge=0)
    token_usage: dict[str, int] | None = None
    failure_state: str | None = None
    failure_phase: str | None = None
    provider_http_status: int | None = None
    provider_error_code: str | None = None
    provider_error_message: str | None = None
    provider_metadata_error_type: str | None = None
    provider_metadata_provider_code: str | None = None
    fallback_state: str | None = None
    shadow_only: bool = True
    awaiting_guardrail_validation: bool = True
    approved_for_publication: bool = False
    generated_at: datetime

    @field_validator("generated_at")
    @classmethod
    def generated_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("AI forecast timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def status_and_probability_invariants(self) -> AIMarketForecast:
        if not self.shadow_only or not self.awaiting_guardrail_validation or self.approved_for_publication:
            raise ValueError("Phase 3/4 AI forecasts must remain unpublished and guardrail-pending")
        values = (self.buy_probability, self.sell_probability, self.neutral_probability)
        if self.status in {AIResultStatus.AVAILABLE, AIResultStatus.NON_ACTIONABLE}:
            if any(value is None for value in values):
                raise ValueError("available AI forecast requires direction probabilities")
            if abs(sum(value for value in values if value is not None) - 1) > 1e-8:
                raise ValueError("AI forecast probabilities must sum to one")
            if self.dominant_direction is None or self.dominant_scenario is None:
                raise ValueError("available AI forecast requires a dominant scenario and direction")
        elif any(value is not None for value in values):
            raise ValueError("unavailable AI forecast cannot fabricate probabilities")
        if all(value is not None for value in (self.expected_minimum_move, self.expected_base_move, self.expected_maximum_move)):
            assert self.expected_minimum_move is not None and self.expected_base_move is not None and self.expected_maximum_move is not None
            if not self.expected_minimum_move <= self.expected_base_move <= self.expected_maximum_move:
                raise ValueError("AI movement bounds are inconsistent")
        return self


class AISignalProposal(ImmutableAIModel):
    schema_version: str = "1.0"
    proposal_id: UUID
    forecast_id: UUID
    market_state_id: UUID
    structural_opportunity_key: str
    recommended_action: ProposalAction
    direction: Direction
    entry_type: str | None = None
    entry_zone: EntryZone | None = None
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit_levels: tuple[float, ...] = ()
    expected_risk_to_reward: float | None = Field(default=None, gt=0)
    invalidation_price: float | None = Field(default=None, gt=0)
    invalidation_conditions: tuple[str, ...] = ()
    expires_at: datetime | None = None
    setup_readiness: SetupReadiness
    proposal_confidence: float = Field(ge=0, le=1)
    risk_notes: tuple[str, ...] = ()
    execution_notes: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[UUID, ...] = ()
    contradicting_evidence_ids: tuple[UUID, ...] = ()
    monitoring_conditions: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    model_identifier: str
    policy_version: str
    shadow_only: bool = True
    awaiting_guardrail_validation: bool = True
    approved_for_publication: bool = False
    created_at: datetime

    @field_validator("expires_at", "created_at")
    @classmethod
    def aware_optional(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("proposal timestamps must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def proposal_geometry(self) -> AISignalProposal:
        if not self.shadow_only or not self.awaiting_guardrail_validation or self.approved_for_publication:
            raise ValueError("AI proposals must remain unpublished and guardrail-pending")
        actionable = self.recommended_action in {ProposalAction.BUY, ProposalAction.SELL, ProposalAction.ADJUST_ENTRY}
        if actionable:
            if self.entry_zone is None or self.stop_loss is None or not self.take_profit_levels:
                raise ValueError("actionable proposal requires entry, stop, and targets")
            if self.direction == Direction.BUY:
                if self.stop_loss >= self.entry_zone.low or any(level <= self.entry_zone.high for level in self.take_profit_levels):
                    raise ValueError("BUY proposal price geometry is invalid")
            elif self.direction == Direction.SELL:
                if self.stop_loss <= self.entry_zone.high or any(level >= self.entry_zone.low for level in self.take_profit_levels):
                    raise ValueError("SELL proposal price geometry is invalid")
            else:
                raise ValueError("actionable proposal cannot be neutral")
        if self.recommended_action == ProposalAction.WAIT and (
            any(value is not None for value in (self.entry_zone, self.stop_loss, self.expected_risk_to_reward, self.invalidation_price))
            or self.take_profit_levels
        ):
            raise ValueError("WAIT proposal cannot fabricate execution levels")
        return self


class MarketMemoryEntry(ImmutableAIModel):
    entry_id: UUID
    instrument: str
    cycle_id: UUID
    market_state_id: UUID
    category: str
    summary: str
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: tuple[UUID, ...] = ()
    opportunity_key: str | None = None
    signal_id: UUID | None = None
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("memory timestamp must be timezone-aware")
        return value.astimezone(UTC)


class ManagedSignal(ImmutableAIModel):
    signal_id: UUID
    instrument: str
    structural_opportunity_key: str
    setup_family: str
    direction: Direction
    state: ManagedSignalState
    current_proposal_id: UUID
    originating_market_state_id: UUID
    latest_market_state_id: UUID
    entry_zone: EntryZone | None = None
    stop_loss: float | None = None
    take_profit_levels: tuple[float, ...] = ()
    invalidation_price: float | None = None
    expires_at: datetime | None = None
    shadow_only: bool = True
    awaiting_guardrail_validation: bool = True
    created_at: datetime
    updated_at: datetime

    @field_validator("expires_at", "created_at", "updated_at")
    @classmethod
    def signal_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("signal timestamps must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None


class SignalStateTransition(ImmutableAIModel):
    transition_id: UUID
    signal_id: UUID
    previous_state: ManagedSignalState
    new_state: ManagedSignalState
    reason: str
    supporting_evidence_ids: tuple[UUID, ...]
    ai_forecast_id: UUID
    ai_proposal_id: UUID
    market_state_id: UUID
    policy_version: str
    model_version: str
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def transition_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("transition timestamp must be timezone-aware")
        return value.astimezone(UTC)


class SignalLevelRevision(ImmutableAIModel):
    revision_id: UUID
    signal_id: UUID
    level_type: str
    old_value: Any
    new_value: Any
    reason: str
    evidence_ids: tuple[UUID, ...]
    model_version: str
    policy_version: str
    approved_rule: str | None = None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def revision_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("revision timestamp must be timezone-aware")
        return value.astimezone(UTC)


class SignalMonitoringEvaluation(ImmutableAIModel):
    evaluation_id: UUID
    signal_id: UUID
    forecast_id: UUID
    proposal_id: UUID | None = None
    market_state_id: UUID
    thesis_valid: bool
    scenario_probability_change: float | None = None
    changed_evidence_ids: tuple[UUID, ...] = ()
    reason_codes: tuple[str, ...] = ()
    recommended_action: ProposalAction
    evaluated_at: datetime

    @field_validator("evaluated_at")
    @classmethod
    def monitoring_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("monitoring timestamp must be timezone-aware")
        return value.astimezone(UTC)


class SignalOutcome(ImmutableAIModel):
    outcome_id: UUID
    signal_id: UUID
    final_state: ManagedSignalState
    realized_return: float | None = None
    reason: str
    closed_at: datetime

    @field_validator("closed_at")
    @classmethod
    def outcome_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("outcome timestamp must be timezone-aware")
        return value.astimezone(UTC)


class LLMStructuredOutputFailure(ImmutableAIModel):
    failure_id: UUID
    request_id: UUID
    attempt: int = Field(ge=0)
    model_identifier: str
    prompt_version: str
    raw_output: dict[str, Any] | None = None
    validation_errors: tuple[str, ...]
    failure_state: str
    provider_failure: dict[str, Any] | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def failure_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("failure timestamp must be timezone-aware")
        return value.astimezone(UTC)
