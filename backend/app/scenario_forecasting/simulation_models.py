"""Authoritative multi-path simulation and Primary Scenario contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import GeometryValidity, PriceZone, ScenarioGeometry, ScenarioValidity


class SimulationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CandidateDirection(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGE = "RANGE"
    INCONCLUSIVE = "INCONCLUSIVE"


class EntryType(StrEnum):
    CURRENT_PRICE = "CURRENT_PRICE"
    PULLBACK = "PULLBACK"
    RETEST = "RETEST"
    BREAKOUT_CONFIRMATION = "BREAKOUT_CONFIRMATION"
    LIQUIDITY_SWEEP_REVERSAL = "LIQUIDITY_SWEEP_REVERSAL"
    NONE = "NONE"


class SelectionStatus(StrEnum):
    SELECTED = "SELECTED"
    INSUFFICIENT_CONFIDENCE = "INSUFFICIENT_CONFIDENCE"
    NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"
    BLOCKED = "BLOCKED"


class ScenarioSignalAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SimulationAttemptStatus(StrEnum):
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    NO_SIGNAL = "NO_SIGNAL"
    ANALYTICAL_ONLY = "ANALYTICAL_ONLY"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

    @property
    def terminal(self) -> bool:
        return self not in {self.SCHEDULED, self.RUNNING}


class AuthoritativeSimulationAttempt(SimulationModel):
    attempt_id: UUID
    instrument: str
    timeframe: str = "M15"
    market_cutoff: datetime
    simulation_version: str
    status: SimulationAttemptStatus
    provider_timestamp: datetime | None = None
    candle_open_time: datetime | None = None
    candle_close_time: datetime | None = None
    resolved_market_cutoff: datetime
    server_time: datetime
    timezone: str = "UTC"
    eligibility_result: bool
    eligibility_reason: str
    m5_cutoff: datetime | None = None
    cutoff_difference_seconds: float | None = Field(default=None, ge=0)
    synchronization_status: str
    scheduled_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    candidate_count: int = Field(default=0, ge=0, le=10)
    simulation_cycle_id: UUID | None = None
    primary_scenario_id: UUID | None = None
    alternative_scenario_id: UUID | None = None
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    skip_reason: str | None = None
    retry_count: int = Field(default=0, ge=0)

    @field_validator(
        "market_cutoff",
        "provider_timestamp",
        "candle_open_time",
        "candle_close_time",
        "resolved_market_cutoff",
        "server_time",
        "m5_cutoff",
        "scheduled_at",
        "started_at",
        "completed_at",
    )
    @classmethod
    def attempt_times_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("simulation attempt timestamps must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def lifecycle_is_coherent(self) -> AuthoritativeSimulationAttempt:
        if self.timeframe != "M15":
            raise ValueError("authoritative simulation attempts are M15 only")
        if self.status == SimulationAttemptStatus.RUNNING and self.started_at is None:
            raise ValueError("running attempt requires started_at")
        if self.status.terminal and self.completed_at is None:
            raise ValueError("terminal attempt requires completed_at")
        if self.status == SimulationAttemptStatus.SUCCESS and self.primary_scenario_id is None:
            raise ValueError("successful attempt requires a Primary Scenario")
        if self.status == SimulationAttemptStatus.FAILED and not self.failure_message:
            raise ValueError("failed attempt requires a failure message")
        if self.status == SimulationAttemptStatus.SKIPPED and not self.skip_reason:
            raise ValueError("skipped attempt requires a reason")
        return self


class QuantMoveConversion(SimulationModel):
    raw_expected_move: float = Field(ge=0)
    raw_expected_move_unit: str
    converted_expected_move: float = Field(gt=0)
    converted_expected_move_unit: str = "price_points"
    conversion_method: str
    reference_price: float = Field(gt=0)


class ScenarioPathStage(SimulationModel):
    stage_id: UUID
    sequence: int = Field(ge=1)
    label: str
    expected_price_area: PriceZone
    supporting_evidence_ids: tuple[str, ...]
    invalidation_condition: str
    timing_seconds: int | None = Field(default=None, gt=0)


class ScenarioScoreComponent(SimulationModel):
    name: str
    raw_value: float
    weight: float = Field(ge=0, le=1)
    contribution: float = Field(ge=-100, le=100)
    reason: str
    evidence_ids: tuple[str, ...] = ()


class CandidateMarketScenario(SimulationModel):
    candidate_id: UUID
    simulation_cycle_id: UUID
    cycle_id: UUID
    instrument: str
    market_cutoff: datetime
    reference_price: float = Field(gt=0)
    forecast_horizon_seconds: int = Field(gt=0)
    direction: CandidateDirection
    scenario_type: str
    path_sequence: tuple[ScenarioPathStage, ...] = Field(min_length=2)
    ai_proposed_path: tuple[str, ...] = Field(min_length=2)
    deterministically_validated_path: tuple[str, ...] = Field(min_length=2)
    expected_low: float = Field(gt=0)
    expected_high: float = Field(gt=0)
    likely_close_low: float = Field(gt=0)
    likely_close_high: float = Field(gt=0)
    expected_move: float = Field(gt=0)
    expected_move_unit: str = "price_points"
    quant_move_conversion: QuantMoveConversion
    trigger_condition: str
    entry_type: EntryType
    entry_zone: PriceZone | None = None
    invalidation_level: float | None = Field(default=None, gt=0)
    protective_stop: float | None = Field(default=None, gt=0)
    primary_target: float | None = Field(default=None, gt=0)
    secondary_target: float | None = Field(default=None, gt=0)
    expiry: datetime
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    score_components: tuple[ScenarioScoreComponent, ...]
    raw_model_score: float
    normalized_confidence: float = Field(ge=0, le=100)
    calibration_adjustment: float = Field(ge=-20, le=20)
    calibrated_probability: float | None = Field(default=None, ge=0, le=1)
    calibration_sample_size: int = Field(default=0, ge=0)
    final_scenario_score: float = Field(ge=0, le=100)
    rank: int = Field(ge=0)
    scenario_validity: ScenarioValidity
    geometry_validity: GeometryValidity
    geometry: ScenarioGeometry | None = None
    rejection_reason: str | None = None
    diversity_key: str

    @field_validator("market_cutoff", "expiry")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("candidate timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def coherent(self) -> CandidateMarketScenario:
        if self.expiry <= self.market_cutoff:
            raise ValueError("candidate must expire after cutoff")
        if self.expected_low > self.reference_price or self.expected_high < self.reference_price:
            raise ValueError("candidate range must contain reference price")
        if self.likely_close_low > self.likely_close_high:
            raise ValueError("likely close zone must be ordered")
        if self.geometry_validity == GeometryValidity.VALID:
            if self.geometry is None or self.rejection_reason is not None:
                raise ValueError("valid geometry requires geometry and no rejection")
        elif self.geometry is not None or not self.rejection_reason:
            raise ValueError("invalid geometry requires an exact rejection reason")
        if self.direction in {CandidateDirection.RANGE, CandidateDirection.INCONCLUSIVE}:
            if self.geometry is not None or self.entry_type != EntryType.NONE:
                raise ValueError("non-directional candidates cannot carry trade geometry")
        if tuple(stage.sequence for stage in self.path_sequence) != tuple(
            range(1, len(self.path_sequence) + 1)
        ):
            raise ValueError("path stages must be consecutively ordered")
        return self


class MarketSimulationCycle(SimulationModel):
    schema_version: str = "1.0"
    simulation_cycle_id: UUID
    cycle_id: UUID
    market_state_id: UUID
    synthesis_id: UUID
    analysis_id: UUID
    quantitative_forecast_id: UUID
    instrument: str
    market_cutoff: datetime
    m5_source_cycle_id: UUID
    m15_source_cycle_id: UUID
    candidate_count: int = Field(ge=5, le=10)
    candidates: tuple[CandidateMarketScenario, ...] = Field(min_length=5, max_length=10)
    created_at: datetime
    engine_version: str
    configuration_version: str

    @field_validator("market_cutoff", "created_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("simulation timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def complete(self) -> MarketSimulationCycle:
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate count is inconsistent")
        keys = [item.diversity_key for item in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("simulation candidates must be materially distinct")
        if any(item.market_cutoff != self.market_cutoff for item in self.candidates):
            raise ValueError("candidate cutoff differs from simulation cutoff")
        return self


class PrimaryScenarioSelection(SimulationModel):
    schema_version: str = "1.0"
    selection_id: UUID
    simulation_cycle_id: UUID
    cycle_id: UUID
    market_state_id: UUID
    instrument: str
    market_cutoff: datetime
    selected_at: datetime
    status: SelectionStatus
    authoritative_action: ScenarioSignalAction
    primary_candidate_id: UUID | None = None
    alternative_candidate_id: UUID | None = None
    primary: CandidateMarketScenario | None = None
    alternative: CandidateMarketScenario | None = None
    minimum_score: float = Field(ge=0, le=100)
    signal_eligible: bool
    rejection_reason: str | None = None
    ranking_explanation: str
    lifecycle_status: str = "GENERATED"

    @field_validator("market_cutoff", "selected_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("selection timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def authority_contract(self) -> PrimaryScenarioSelection:
        if self.status == SelectionStatus.SELECTED:
            if self.primary is None or self.primary_candidate_id != self.primary.candidate_id:
                raise ValueError("selected authority requires its Primary Scenario")
            expected_action = {
                CandidateDirection.BULLISH: ScenarioSignalAction.BUY,
                CandidateDirection.BEARISH: ScenarioSignalAction.SELL,
            }.get(self.primary.direction, ScenarioSignalAction.HOLD)
            if self.authoritative_action != expected_action:
                raise ValueError("authoritative action must match Primary Scenario")
            if self.alternative is None or self.alternative_candidate_id != self.alternative.candidate_id:
                raise ValueError("selected authority requires a distinct Alternative Scenario")
        elif self.authoritative_action != ScenarioSignalAction.HOLD or self.signal_eligible:
            raise ValueError("unselected simulation must fail closed to HOLD")
        if self.signal_eligible:
            if (
                self.primary is None
                or self.primary.final_scenario_score < self.minimum_score
                or self.primary.scenario_validity != ScenarioValidity.VALID
                or self.primary.geometry_validity != GeometryValidity.VALID
                or self.primary.geometry is None
            ):
                raise ValueError("eligible Primary Scenario must pass every deterministic gate")
        return self


class CandidateScenarioOutcome(SimulationModel):
    outcome_id: UUID
    candidate_id: UUID
    selection_id: UUID
    instrument: str
    status: str
    actual_high: float = Field(gt=0)
    actual_low: float = Field(gt=0)
    actual_close: float = Field(gt=0)
    target_reached: bool
    invalidation_occurred: bool
    directional_accuracy: float = Field(ge=0, le=1)
    completed_at: datetime

    @field_validator("completed_at")
    @classmethod
    def outcome_time_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("outcome timestamp must be timezone-aware")
        return value.astimezone(UTC)
