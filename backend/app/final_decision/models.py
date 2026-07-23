"""Immutable Phase 5/6 contracts owned by deterministic system policy."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.ai_reasoning.models import Direction, EntryZone, ManagedSignalState


class ImmutableDecisionModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FinalAction(StrEnum):
    APPROVED = "approved"
    APPROVED_REDUCED_RISK = "approved_with_reduced_risk"
    POSTPONED = "postponed"
    TEMPORARILY_BLOCKED = "temporarily_blocked"
    REJECTED = "rejected"
    PUBLISHED = "published"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"
    CLOSED = "closed"


class ApprovalState(StrEnum):
    APPROVED = "approved"
    MODIFIED = "modified"
    POSTPONED = "postponed"
    BLOCKED = "blocked"
    REJECTED = "rejected"


class PublicationState(StrEnum):
    NOT_REQUESTED = "not_requested"
    DISABLED = "disabled"
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"


class GateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class RiskClassification(StrEnum):
    STANDARD = "standard"
    REDUCED = "reduced"
    UNAVAILABLE = "unavailable"
    BLOCKED = "blocked"


class OperationMode(StrEnum):
    SAFE_TEST = "safe_test"
    SHADOW = "shadow"
    ANALYTICAL_LIVE = "analytical_live"


class ReplayLLMMode(StrEnum):
    RECORDED_RESPONSE = "recorded_response"
    DETERMINISTIC_BASELINE = "deterministic_baseline"
    FRESH_MODEL = "fresh_model"


class ExecutionContext(ImmutableDecisionModel):
    context_id: UUID
    instrument: str
    evaluated_at: datetime
    operation_mode: OperationMode = OperationMode.SHADOW
    analytical_only: bool = True
    broker_execution_available: bool = False
    market_open: bool | None = None
    current_price: float | None = Field(default=None, gt=0)
    spread: float | None = Field(default=None, ge=0)
    session: str = "unknown"
    publication_service_available: bool = True
    persistence_available: bool = True
    economic_context_available: bool = False
    prohibited_economic_event_window: bool | None = None
    authoritative_account_risk_available: bool = False
    risk_per_signal: float | None = None
    simultaneous_exposure: float | None = None
    daily_loss: float | None = None
    aggregate_drawdown: float | None = None
    conflicting_active_exposure: bool | None = None
    position_size_valid: bool | None = None
    active_opportunity_keys: tuple[str, ...] = ()
    active_signal_id: UUID | None = None

    @field_validator("evaluated_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("execution context timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def no_fabricated_broker_state(self) -> ExecutionContext:
        authoritative = (
            self.risk_per_signal,
            self.simultaneous_exposure,
            self.daily_loss,
            self.aggregate_drawdown,
            self.conflicting_active_exposure,
            self.position_size_valid,
        )
        if not self.authoritative_account_risk_available and any(value is not None for value in authoritative):
            raise ValueError("account-risk values require an authoritative account-risk service")
        if self.broker_execution_available and self.analytical_only:
            raise ValueError("analytical-only context cannot claim broker execution")
        return self


class HardGateDefinition(ImmutableDecisionModel):
    gate_id: str
    gate_version: str
    category: str
    applicable_actions: tuple[str, ...]
    applicable_setup_families: tuple[str, ...] = ("*",)
    required_inputs: tuple[str, ...]
    evaluator: str
    severity: str
    block_behavior: str
    reason_codes: tuple[str, ...]
    configuration_source: str


class GateEvaluation(ImmutableDecisionModel):
    evaluation_id: UUID
    final_action_id: UUID
    gate_id: str
    gate_version: str
    category: str
    status: GateStatus
    severity: str
    block_behavior: str
    reason_codes: tuple[str, ...] = ()
    audit_payload: dict[str, Any] = Field(default_factory=dict)
    configuration_source: str
    evaluated_at: datetime

    @field_validator("evaluated_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("gate timestamp must be timezone-aware")
        return value.astimezone(UTC)


class ProposalModification(ImmutableDecisionModel):
    field_name: str
    original_value: Any
    final_value: Any
    modifying_gate_or_policy: str
    exact_reason: str


class FinalSystemAction(ImmutableDecisionModel):
    schema_version: str = "1.0"
    final_action_id: UUID
    ai_proposal_id: UUID
    managed_signal_id: UUID
    market_state_id: UUID
    quantitative_forecast_id: UUID
    ai_forecast_id: UUID
    action: FinalAction
    approval_state: ApprovalState
    publication_state: PublicationState
    final_direction: Direction
    final_entry: EntryZone | None
    final_stop_loss: float | None
    final_take_profits: tuple[float, ...]
    final_risk_to_reward: float | None
    final_expiry: datetime | None
    final_risk_classification: RiskClassification
    gate_evaluations: tuple[GateEvaluation, ...]
    modifications: tuple[ProposalModification, ...] = ()
    modification_reasons: tuple[str, ...] = ()
    policy_versions: dict[str, str]
    original_proposal_hash: str
    analytical_only: bool = True
    broker_execution_performed: bool = False
    created_at: datetime

    @field_validator("final_expiry", "created_at")
    @classmethod
    def aware_optional(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("final action timestamps must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def analytical_boundary(self) -> FinalSystemAction:
        if self.broker_execution_performed:
            raise ValueError("TEN has no approved broker execution path")
        if self.action == FinalAction.PUBLISHED and self.publication_state != PublicationState.PUBLISHED:
            raise ValueError("published action requires a persisted publication")
        return self


class PublishedAnalyticalSignal(ImmutableDecisionModel):
    schema_version: str = "1.0"
    publication_id: UUID
    signal_id: UUID
    final_action_id: UUID
    proposal_id: UUID
    instrument: str
    direction: Direction
    setup_family: str
    entry_zone: EntryZone
    stop_loss: float
    take_profit_levels: tuple[float, ...]
    invalidation_price: float | None
    invalidation_conditions: tuple[str, ...]
    expires_at: datetime | None
    expected_horizon: str
    buy_probability: float
    sell_probability: float
    neutral_probability: float
    forecast_confidence: float | None
    uncertainty: float | None
    proposal_confidence: float
    final_risk_classification: RiskClassification
    dominant_scenario: str
    supporting_evidence_summary: tuple[str, ...]
    contradicting_evidence_summary: tuple[str, ...]
    lifecycle_state: ManagedSignalState
    model_versions: dict[str, str]
    policy_versions: dict[str, str]
    analytical_only: bool = True
    broker_execution: bool = False
    published_at: datetime

    @field_validator("expires_at", "published_at")
    @classmethod
    def aware_optional(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("publication timestamps must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def probabilities_and_scope(self) -> PublishedAnalyticalSignal:
        if abs(self.buy_probability + self.sell_probability + self.neutral_probability - 1) > 1e-8:
            raise ValueError("published probabilities must sum to one")
        if self.broker_execution or not self.analytical_only:
            raise ValueError("publication is analytical and does not execute at a broker")
        return self


class LLMUsageMetric(ImmutableDecisionModel):
    metric_id: UUID
    usage_date: str
    request_hash: str
    market_state_hash: str
    model_identifier: str
    prompt_version: str
    generation_parameters: dict[str, Any]
    request_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    success: bool
    failure_state: str | None = None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("usage timestamp must be timezone-aware")
        return value.astimezone(UTC)


class DetailedSignalOutcome(ImmutableDecisionModel):
    outcome_id: UUID
    signal_id: UUID
    status: str
    evaluation_horizon_complete: bool
    realized_direction: Direction | None = None
    realized_return: float | None = None
    maximum_favorable_excursion: float | None = None
    maximum_adverse_excursion: float | None = None
    tp1_result: str | None = None
    tp2_result: str | None = None
    stop_loss_result: str | None = None
    tp_sl_ordering: str | None = None
    time_to_entry_seconds: float | None = None
    time_to_tp_seconds: float | None = None
    time_to_sl_seconds: float | None = None
    expiry_outcome: str | None = None
    spread_adjusted_result: float | None = None
    slippage_adjusted_result: float | None = None
    realized_risk_to_reward: float | None = None
    signal_lifetime_seconds: float | None = None
    cancellation_outcome: str | None = None
    invalidation_outcome: str | None = None
    evaluated_at: datetime
    reason_codes: tuple[str, ...] = ()

    @field_validator("evaluated_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("outcome timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def unresolved_has_no_result(self) -> DetailedSignalOutcome:
        if not self.evaluation_horizon_complete and any(
            value is not None
            for value in (
                self.realized_direction,
                self.realized_return,
                self.tp1_result,
                self.stop_loss_result,
                self.realized_risk_to_reward,
            )
        ):
            raise ValueError("unresolved outcomes cannot be labeled as wins or losses")
        return self


class PerformanceReport(ImmutableDecisionModel):
    report_id: UUID
    period_start: datetime
    period_end: datetime
    comparison: dict[str, dict[str, float | int | None]]
    dimensions: dict[str, dict[str, dict[str, float | int | None]]]
    sample_count: int = Field(ge=0)
    generated_at: datetime

    @field_validator("period_start", "period_end", "generated_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("report timestamps must be timezone-aware")
        return value.astimezone(UTC)


class ProbabilityCalibrationReport(ImmutableDecisionModel):
    report_id: UUID
    status: str
    sample_count: int = Field(ge=0)
    brier_score: float | None
    log_loss: float | None
    expected_calibration_error: float | None
    reliability_buckets: tuple[dict[str, Any], ...]
    dimensions: dict[str, dict[str, float | int | None]]
    generated_at: datetime

    @field_validator("generated_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("calibration timestamp must be timezone-aware")
        return value.astimezone(UTC)


class ProductionReadinessReport(ImmutableDecisionModel):
    report_id: UUID
    status: str
    measured_checks: dict[str, dict[str, Any]]
    sample_count: int = Field(ge=0)
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    generated_at: datetime

    @field_validator("generated_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("readiness timestamp must be timezone-aware")
        return value.astimezone(UTC)
