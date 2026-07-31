from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
import math
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.ai_reasoning.analysis import (
    AIAnalysisSignal,
    AIMarketAnalysis,
    AIAnalysisTemporalContext,
    TemporalAnalysisMetrics,
)
from backend.app.engines.ai_scoring_engine import AIScoreSnapshot
if TYPE_CHECKING:
    from backend.app.scenario_forecasting.simulation_models import PrimaryScenarioSelection
else:
    PrimaryScenarioSelection = Any

NAMESPACE = UUID("81f5a757-4e58-5d38-b01d-ea7ec9f9e11d")
type JSONScalar = str | int | float | bool | None


class DecisionModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DecisionState(StrEnum):
    ELIGIBLE = "eligible"
    OBSERVE_ONLY = "observe_only"
    BLOCKED = "blocked"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EXPIRED = "expired"
    INVALID = "invalid"


class DecisionDirection(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class FinalSignalAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalReadiness(StrEnum):
    READY = "ready"
    CONDITIONAL = "conditional"
    WAITING = "waiting"
    REJECTED = "rejected"


class SignalEvidence(DecisionModel):
    evidence_id: UUID
    category: str
    side: str
    claim: str
    source_type: str
    source_reference: str
    observed_value: JSONScalar = None
    timestamp: datetime
    timeframe: str | None = None
    reliability: float = Field(ge=0, le=1)

    @field_validator("timestamp")
    @classmethod
    def evidence_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("evidence timestamps must be timezone-aware")
        return value.astimezone(UTC)


class SignalSourceLineage(DecisionModel):
    market_snapshot_id: UUID | None = None
    feature_snapshot_id: UUID | None = None
    current_ai_analysis_id: UUID | None = None
    current_ai_signal_id: UUID | None = None
    historical_ai_analysis_ids: tuple[UUID, ...] = ()
    quantitative_forecast_id: UUID | None = None
    primary_scenario_selection_id: UUID | None = None
    strategy_evaluation_ids: tuple[str, ...] = ()
    temporal_context_version: str | None = None
    signal_engine_version: str
    strategy_configuration_version: str
    risk_policy_version: str


class DecisionMode(StrEnum):
    LIVE = "live"
    REPLAY = "replay"


class DecisionLifecycleStatus(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class RuleOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    NOT_EVALUATED = "not_evaluated"


class RuleSeverity(StrEnum):
    HARD_BLOCK = "hard_block"
    INSUFFICIENT = "insufficient"
    SOFT_GATE = "soft_gate"
    WARNING = "warning"
    INFORMATIONAL = "informational"


class RuleCategory(StrEnum):
    SOURCE_INTEGRITY = "source_integrity"
    FRESHNESS = "freshness"
    DIRECTIONAL_STRENGTH = "directional_strength"
    CONFIDENCE = "confidence"
    RISK = "risk"
    DATA_QUALITY = "data_quality"
    ALIGNMENT = "alignment"
    CONFLICT = "conflict"
    ECONOMIC_EVENT = "economic_event"
    MARKET_REGIME = "market_regime"
    DEPENDENCY_HEALTH = "dependency_health"
    DUPLICATE = "duplicate"
    COOLDOWN = "cooldown"
    REVERSAL = "reversal"
    TEMPORAL_VALIDITY = "temporal_validity"
    POLICY_INTEGRITY = "policy_integrity"


class DependencyState(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class DependencyCriticality(StrEnum):
    CRITICAL = "critical"
    REQUIRED_FOR_ELIGIBILITY = "required_for_eligibility"
    OPTIONAL = "optional"
    INFORMATIONAL = "informational"


class DependencyHealth(DecisionModel):
    name: str = Field(min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_]+$")
    state: DependencyState
    criticality: DependencyCriticality
    checked_at: datetime
    reason_code: str = "dependency_status"

    @field_validator("checked_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("dependency timestamps must be timezone-aware")
        return value.astimezone(UTC)


class EconomicRiskReference(DecisionModel):
    context_id: UUID
    phase: str
    risk_score: float = Field(ge=0, le=100)
    as_of: datetime
    degraded: bool = False
    # The economic_calendar engine's own categorical state (see `CalendarContextState`), e.g.
    # "provider_unreachable", "no_relevant_events", "outside_risk_window" — mirrored verbatim so
    # this rule can distinguish a genuine provider failure from "nothing relevant is happening,"
    # instead of collapsing both into the same `degraded` boolean.
    context_state: str = "outside_risk_window"
    event_ids: tuple[UUID, ...] = ()

    @field_validator("as_of")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("economic context timestamp must be timezone-aware")
        return value.astimezone(UTC)


class MarketRegimeReference(DecisionModel):
    snapshot_id: UUID
    regime: str
    as_of: datetime
    degraded: bool = False

    @field_validator("as_of")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("regime timestamp must be timezone-aware")
        return value.astimezone(UTC)


class DecisionHistoryReference(DecisionModel):
    decision_id: UUID
    direction: DecisionDirection
    state: DecisionState
    decided_at: datetime
    valid_until: datetime
    directional_strength: float = Field(ge=0, le=100)
    confidence_score: float = Field(ge=0, le=100)

    @field_validator("decided_at", "valid_until")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("history timestamps must be timezone-aware")
        return value.astimezone(UTC)


class DecisionHistory(DecisionModel):
    latest: DecisionHistoryReference | None = None
    active: DecisionHistoryReference | None = None
    recent_same_direction: DecisionHistoryReference | None = None
    recent_opposite_eligible: DecisionHistoryReference | None = None


class SignalDecisionInput(DecisionModel):
    instrument: str = Field(min_length=2, max_length=32, pattern=r"^[A-Z0-9._-]+$")
    timeframe: str = Field(pattern=r"^(M1|M5|M15|M30|H1|H4|D1|W1|MN1)$")
    as_of: datetime
    requested_at: datetime
    ai_score: AIScoreSnapshot
    economic_risk: EconomicRiskReference | None = None
    market_regime: MarketRegimeReference | None = None
    dependency_health: tuple[DependencyHealth, ...] = ()
    history: DecisionHistory = DecisionHistory()
    current_ai_analysis: AIMarketAnalysis | None = None
    current_ai_signal: AIAnalysisSignal | None = None
    temporal_context: AIAnalysisTemporalContext | None = None
    temporal_metrics: TemporalAnalysisMetrics | None = None
    market_snapshot_id: UUID | None = None
    quantitative_forecast_id: UUID | None = None
    current_price: float | None = Field(default=None, gt=0)
    expected_move: float | None = Field(default=None, gt=0)
    current_primary_scenario: PrimaryScenarioSelection | None = None
    mode: DecisionMode = DecisionMode.LIVE
    policy_name: str
    policy_version: str

    @field_validator("as_of", "requested_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("decision timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def source_identity(self) -> SignalDecisionInput:
        if self.ai_score.instrument != self.instrument or self.ai_score.timeframe != self.timeframe:
            raise ValueError("AI score identity does not match decision request")
        if self.ai_score.as_of > self.as_of or self.ai_score.calculated_at > self.requested_at:
            raise ValueError("future AI score evidence is prohibited")
        if self.economic_risk and self.economic_risk.as_of > self.as_of:
            raise ValueError("future economic context is prohibited")
        if self.market_regime and self.market_regime.as_of > self.as_of:
            raise ValueError("future market regime is prohibited")
        if self.current_ai_analysis is not None:
            if self.current_ai_analysis.analysis_timestamp > self.as_of:
                raise ValueError("future AI analysis is prohibited")
            if not self.current_ai_analysis.validation_passed:
                raise ValueError("invalid AI analysis is prohibited")
        if self.current_ai_signal is not None:
            if self.current_ai_analysis is None:
                raise ValueError("AI signal requires its current AI analysis")
            if self.current_ai_signal.analysis_id != self.current_ai_analysis.analysis_id:
                raise ValueError("AI signal does not belong to current AI analysis")
            if self.current_ai_signal.snapshot_id != self.market_snapshot_id:
                raise ValueError("AI signal does not belong to current snapshot")
        if self.temporal_context is not None and self.temporal_context.as_of > self.as_of:
            raise ValueError("future temporal context is prohibited")
        if self.current_primary_scenario is not None:
            primary = self.current_primary_scenario
            if primary.market_cutoff > self.as_of:
                raise ValueError("future Primary Scenario is prohibited")
            if primary.market_state_id != self.market_snapshot_id:
                raise ValueError("Primary Scenario does not belong to current snapshot")
            if primary.instrument != self.instrument:
                raise ValueError("Primary Scenario instrument mismatch")
        return self

    def fingerprint(self, configuration_hash: str) -> str:
        payload = self.model_dump(mode="json", exclude={"requested_at", "dependency_health", "history"})
        payload["configuration_hash"] = configuration_hash
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class RuleEvaluation(DecisionModel):
    rule_id: str
    rule_version: str = "1.0.0"
    category: RuleCategory
    severity: RuleSeverity
    outcome: RuleOutcome
    observed_value: JSONScalar = None
    threshold: JSONScalar = None
    reason_code: str
    contribution: float | None = Field(default=None, ge=-100, le=100)
    evaluated_at: datetime
    metadata: dict[str, JSONScalar] = Field(default_factory=dict)

    @field_validator("evaluated_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("rule timestamp must be timezone-aware")
        return value.astimezone(UTC)


class DecisionReason(DecisionModel):
    reason_code: str
    severity: RuleSeverity
    source: str
    message_key: str
    rule_id: str
    metadata: dict[str, JSONScalar] = Field(default_factory=dict)


class DecisionExplanation(DecisionModel):
    summary_code: str
    decision_state: DecisionState
    direction: DecisionDirection
    passed_rules: tuple[str, ...] = ()
    failed_rules: tuple[str, ...] = ()
    warning_rules: tuple[str, ...] = ()
    evidence_limitations: tuple[str, ...] = ()
    policy_name: str
    policy_version: str
    financial_safety_notice: str = "analytical_decision_only"


class DecisionMetadata(DecisionModel):
    engine_version: str
    schema_version: str
    configuration_version: str
    configuration_hash: str = Field(min_length=64, max_length=64)
    ai_score_configuration_hash: str = Field(min_length=64, max_length=64)
    analytical_decision: bool = True
    deterministic: bool = True
    point_in_time_safe: bool = True
    replay_safe: bool = True
    explainable: bool = True
    trade_execution: bool = False
    order_generation: bool = False
    position_sizing: bool = False
    broker_integration: bool = False
    publication_degraded: bool = False


class SignalDecision(DecisionModel):
    decision_id: UUID
    decision_key: str
    input_fingerprint: str = Field(min_length=64, max_length=64)
    instrument: str
    timeframe: str
    direction: DecisionDirection
    state: DecisionState
    as_of: datetime
    decided_at: datetime
    valid_from: datetime
    valid_until: datetime
    ai_score_snapshot_id: UUID
    ai_score_policy_name: str
    ai_score_policy_version: str
    decision_policy_name: str
    decision_policy_version: str
    eligibility_score: float = Field(ge=0, le=100)
    directional_strength: float = Field(ge=0, le=100)
    confidence_score: float = Field(ge=0, le=100)
    guardrail_confidence: float = Field(default=0, ge=0, le=100)
    overall_confidence: float = Field(default=0, ge=0, le=100)
    market_risk_score: float = Field(ge=0, le=100)
    data_quality_score: float = Field(ge=0, le=100)
    evidence_alignment_score: float = Field(ge=0, le=100)
    rules: tuple[RuleEvaluation, ...]
    blockers: tuple[DecisionReason, ...]
    warnings: tuple[DecisionReason, ...]
    supporting_reasons: tuple[DecisionReason, ...]
    previous_decision_id: UUID | None = None
    supersedes_decision_id: UUID | None = None
    status: DecisionLifecycleStatus = DecisionLifecycleStatus.ACTIVE
    mode: DecisionMode
    explanation: DecisionExplanation
    metadata: DecisionMetadata
    final_action: FinalSignalAction = FinalSignalAction.HOLD
    setup_family: str | None = None
    readiness: SignalReadiness = SignalReadiness.WAITING
    entry_low: float | None = Field(default=None, gt=0)
    entry_high: float | None = Field(default=None, gt=0)
    invalidation: str | None = None
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit_targets: tuple[float, ...] = ()
    risk_reward: float | None = Field(default=None, gt=0)
    decision_reason: str = ""
    opposite_direction_rejection: str = ""
    supporting_evidence: tuple[SignalEvidence, ...] = ()
    contradicting_evidence: tuple[SignalEvidence, ...] = ()
    temporal_evidence: tuple[SignalEvidence, ...] = ()
    risk_evidence: tuple[SignalEvidence, ...] = ()
    source_lineage: SignalSourceLineage | None = None
    publication_eligible: bool = False
    notification_context: dict[str, Any] | None = None

    @field_validator("as_of", "decided_at", "valid_from", "valid_until")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("decision timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("final_action", mode="before")
    @classmethod
    def normalize_historical_wait_action(cls, value: object) -> object:
        """Keep historical JSON readable without allowing WAIT into new decisions."""

        return FinalSignalAction.HOLD if value == "WAIT" else value

    @field_validator("readiness", mode="before")
    @classmethod
    def normalize_historical_waiting_readiness(cls, value: object) -> object:
        return SignalReadiness.REJECTED if value == "waiting" else value

    @model_validator(mode="after")
    def invariants(self) -> SignalDecision:
        if self.valid_until < self.valid_from:
            raise ValueError("valid_until cannot precede valid_from")
        if self.state == DecisionState.ELIGIBLE and (self.blockers or self.direction == DecisionDirection.NEUTRAL):
            raise ValueError("eligible decisions require non-neutral direction and no blockers")
        if self.state == DecisionState.BLOCKED and not self.blockers:
            raise ValueError("blocked decisions require a blocker")
        if self.state == DecisionState.OBSERVE_ONLY and (self.blockers or not self.warnings):
            raise ValueError("observe-only decisions require warnings and no blockers")
        if self.state in {DecisionState.INSUFFICIENT_EVIDENCE, DecisionState.INVALID} and not self.blockers:
            raise ValueError("non-actionable decisions require explicit limitations")
        if self.final_action == FinalSignalAction.HOLD:
            if any(
                value is not None
                for value in (
                    self.entry_low,
                    self.entry_high,
                    self.stop_loss,
                    self.risk_reward,
                )
            ) or self.take_profit_targets:
                raise ValueError("HOLD cannot contain execution geometry")
            if self.publication_eligible:
                raise ValueError("HOLD cannot be publication eligible")
        else:
            geometry = (
                self.entry_low,
                self.entry_high,
                self.stop_loss,
                self.risk_reward,
            )
            if self.publication_eligible or self.readiness == SignalReadiness.READY:
                if (
                    any(value is None for value in geometry)
                    or not self.take_profit_targets
                    or not self.invalidation
                ):
                    raise ValueError(
                        "execution-ready signal requires entry, invalidation, stop, "
                        "targets, and risk/reward"
                    )
            elif any(value is not None for value in geometry) or self.take_profit_targets:
                raise ValueError("blocked analytical direction cannot contain partial geometry")
        for value in (self.eligibility_score, self.directional_strength, self.confidence_score, self.market_risk_score, self.data_quality_score, self.evidence_alignment_score):
            if not math.isfinite(value):
                raise ValueError("decision scores must be finite")
        return self

    def history_reference(self) -> DecisionHistoryReference:
        return DecisionHistoryReference(
            decision_id=self.decision_id,
            direction=self.direction,
            state=self.state,
            decided_at=self.decided_at,
            valid_until=self.valid_until,
            directional_strength=self.directional_strength,
            confidence_score=self.confidence_score,
        )


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instrument: str = Field(default="XAUUSD", min_length=2, max_length=32, pattern=r"^[A-Za-z0-9._-]+$")
    timeframe: str = Field(default="M15", pattern=r"^(M1|M5|M15|M30|H1|H4|D1|W1|MN1)$")
    ai_score_snapshot_id: UUID
    as_of: datetime | None = None
    mode: DecisionMode = DecisionMode.LIVE
    decision_policy_name: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{2,63}$")
    decision_policy_version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    persist: bool = True
    publish_events: bool = True
    current_ai_analysis: AIMarketAnalysis | None = None
    current_ai_signal: AIAnalysisSignal | None = None
    temporal_context: AIAnalysisTemporalContext | None = None
    temporal_metrics: TemporalAnalysisMetrics | None = None
    market_snapshot_id: UUID | None = None
    quantitative_forecast_id: UUID | None = None
    current_price: float | None = Field(default=None, gt=0)
    expected_move: float | None = Field(default=None, gt=0)
    current_primary_scenario: PrimaryScenarioSelection | None = None

    @field_validator("instrument")
    @classmethod
    def canonical_instrument(cls, value: str) -> str:
        return value.upper()


def stable_id(*parts: object) -> UUID:
    return uuid5(NAMESPACE, "|".join(str(part) for part in parts))
