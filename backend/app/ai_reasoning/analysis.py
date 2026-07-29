"""Analysis-only AI contracts and deterministic temporal intelligence.

The provider is allowed to interpret market evidence, but never to recommend a
trade.  Trading decisions remain the responsibility of the deterministic
Signal Decision Engine.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from statistics import fmean, pstdev
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictAnalysisModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AnalysisStatus(StrEnum):
    AVAILABLE = "available"
    INVALID = "invalid"
    FAILED = "failed"


class AnalysisSignalAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AnalysisSignalStrength(StrEnum):
    VERY_WEAK = "VERY_WEAK"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


class AnalysisSignalLifecycle(StrEnum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    STALE = "STALE"
    EXPIRED = "EXPIRED"
    STOPPED = "STOPPED"
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    SUPERSEDED = "SUPERSEDED"


class AnalysisExecutionEligibility(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"


class AnalysisExecutionStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class QuantAIAlignment(StrEnum):
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    QUANT_UNAVAILABLE = "quant_unavailable"
    NEUTRAL = "neutral"


class EvidenceKind(StrEnum):
    OBSERVED_FACT = "observed_fact"
    CALCULATED_FEATURE = "calculated_feature"
    AI_INTERPRETATION = "ai_interpretation"
    UNCERTAINTY = "uncertainty"


class AnalysisBias(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class RegimeClassification(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    TRANSITIONAL = "transitional"
    UNCERTAIN = "uncertain"


class AnalysisEvidence(StrictAnalysisModel):
    claim: str = Field(min_length=1, max_length=500)
    kind: EvidenceKind
    source_type: str = Field(min_length=1, max_length=64)
    source_reference: str = Field(min_length=1, max_length=256)
    timeframe: str | None = Field(max_length=16)
    observed_value: str | float | int | bool | None


class MarketRegimeAnalysis(StrictAnalysisModel):
    classification: RegimeClassification
    strength: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[AnalysisEvidence, ...]


class HigherTimeframeAnalysis(StrictAnalysisModel):
    bias: AnalysisBias
    description: str = Field(min_length=1, max_length=1000)
    evidence: tuple[AnalysisEvidence, ...]


class MarketStructureAnalysis(StrictAnalysisModel):
    short_term: str = Field(min_length=1, max_length=500)
    medium_term: str = Field(min_length=1, max_length=500)
    higher_timeframe: str = Field(min_length=1, max_length=500)
    recent_change: str = Field(min_length=1, max_length=500)
    evidence: tuple[AnalysisEvidence, ...]


class LiquidityAnalysis(StrictAnalysisModel):
    summary: str = Field(min_length=1, max_length=1000)
    events: tuple[str, ...]
    unresolved_liquidity: tuple[str, ...]
    evidence: tuple[AnalysisEvidence, ...]


class SupplyDemandAnalysis(StrictAnalysisModel):
    summary: str = Field(min_length=1, max_length=1000)
    nearest_supply: float | None = Field(gt=0)
    nearest_demand: float | None = Field(gt=0)
    evidence: tuple[AnalysisEvidence, ...]


class MomentumTrend(StrEnum):
    STRENGTHENING = "strengthening"
    WEAKENING = "weakening"
    STABLE = "stable"
    UNCERTAIN = "uncertain"


class MomentumAnalysis(StrictAnalysisModel):
    direction: AnalysisBias
    strength: float = Field(ge=0, le=100)
    trend: MomentumTrend
    evidence: tuple[AnalysisEvidence, ...]


class VolatilityState(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"
    UNCERTAIN = "uncertain"


class VolatilityTrend(StrEnum):
    EXPANDING = "expanding"
    CONTRACTING = "contracting"
    STABLE = "stable"
    UNCERTAIN = "uncertain"


class VolatilityAnalysis(StrictAnalysisModel):
    state: VolatilityState
    trend: VolatilityTrend
    evidence: tuple[AnalysisEvidence, ...]


class AlternativeAnalysisScenario(StrictAnalysisModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=1000)
    probability: float = Field(ge=0, le=1)
    confirmation_evidence: tuple[str, ...]


class AIAnalysisOutput(StrictAnalysisModel):
    """Exact provider-wire contract.

    Deliberately contains no signal, proposal, setup, price geometry, readiness,
    publication, or execution fields.
    """

    market_regime: MarketRegimeAnalysis
    higher_timeframe_context: HigherTimeframeAnalysis
    market_structure: MarketStructureAnalysis
    liquidity_analysis: LiquidityAnalysis
    supply_demand_analysis: SupplyDemandAnalysis
    momentum_analysis: MomentumAnalysis
    volatility_analysis: VolatilityAnalysis
    bullish_evidence: tuple[AnalysisEvidence, ...]
    bearish_evidence: tuple[AnalysisEvidence, ...]
    contradictions: tuple[AnalysisEvidence, ...]
    key_risks: tuple[AnalysisEvidence, ...]
    alternative_scenarios: tuple[AlternativeAnalysisScenario, ...]
    analysis_confidence: float = Field(ge=0, le=1)
    executive_summary: str = Field(min_length=1, max_length=1500)
    invalidation_conditions: tuple[str, ...] = Field(default=(), max_length=2)
    data_quality_warnings: tuple[str, ...] = Field(default=(), max_length=3)

    @field_validator("invalidation_conditions")
    @classmethod
    def bounded_invalidations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or len(value) > 160 for value in values):
            raise ValueError("invalidation conditions must contain 1-160 characters")
        return values

    @field_validator("data_quality_warnings")
    @classmethod
    def bounded_quality_warnings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or len(value) > 120 for value in values):
            raise ValueError("data-quality warnings must contain 1-120 characters")
        return values


class AIProviderMetadata(StrictAnalysisModel):
    provider: str
    model: str
    prompt_version: str
    provider_adapter_version: str
    fallback_used: bool = False
    fallback_reason: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    token_usage: dict[str, int] | None = None


class AIMarketAnalysis(StrictAnalysisModel):
    schema_version: str = "2.0"
    analysis_id: UUID
    request_id: UUID
    cycle_id: UUID
    symbol: str
    timeframe: str
    market_snapshot_id: UUID
    quantitative_forecast_id: UUID
    analysis_timestamp: datetime
    knowledge_cutoff: datetime
    status: AnalysisStatus
    output: AIAnalysisOutput | None
    provider_metadata: AIProviderMetadata
    validation_passed: bool
    validation_errors: tuple[str, ...] = ()
    created_at: datetime

    @field_validator("analysis_timestamp", "knowledge_cutoff", "created_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("analysis timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def invariants(self) -> AIMarketAnalysis:
        if self.analysis_timestamp > self.knowledge_cutoff:
            raise ValueError("analysis exceeds its point-in-time knowledge cutoff")
        if self.status == AnalysisStatus.AVAILABLE:
            if self.output is None or not self.validation_passed:
                raise ValueError("available analysis requires a validated output")
        elif self.output is not None:
            raise ValueError("invalid analysis cannot retain provider interpretation")
        return self


class AIAnalysisSignal(StrictAnalysisModel):
    """Deterministic signal derived from a validated analysis, never from LLM fields."""

    schema_version: str = "2.0"
    signal_id: UUID
    cycle_id: UUID
    snapshot_id: UUID
    analysis_id: UUID
    instrument: str
    timeframe: str
    signal: AnalysisSignalAction
    confidence: int = Field(ge=0, le=100)
    strength: AnalysisSignalStrength
    entry: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    risk_reward_ratio: float | None = Field(default=None, gt=0)
    evidence_refs: tuple[str, ...] = ()
    reasoning_summary: str
    risk_flags: tuple[str, ...] = ()
    scoring_components: dict[str, float]
    analysis_confidence: float = Field(default=0, ge=0, le=100)
    signal_confidence: float = Field(default=0, ge=0, le=100)
    quant_confidence: float | None = Field(default=None, ge=0, le=100)
    overall_confidence: float = Field(default=0, ge=0, le=100)
    quant_ai_alignment: QuantAIAlignment = QuantAIAlignment.QUANT_UNAVAILABLE
    quant_ai_explanation: str = "Quantitative alignment was not evaluated."
    quality_threshold: float = Field(default=0, ge=0, le=100)
    geometry_basis: tuple[str, ...] = ()
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    expected_holding_seconds: int | None = Field(default=None, gt=0)
    lifecycle_status: AnalysisSignalLifecycle = AnalysisSignalLifecycle.ACTIVE
    execution_eligibility: AnalysisExecutionEligibility = (
        AnalysisExecutionEligibility.INELIGIBLE
    )
    execution_status: AnalysisExecutionStatus = AnalysisExecutionStatus.BLOCKED
    blocking_reasons: tuple[str, ...] = ()
    fallback: bool = False
    source: str = "deterministic_analysis_signal"
    generated_at: datetime

    @field_validator("generated_at", "valid_from", "valid_until")
    @classmethod
    def signal_time_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("analysis-signal timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="before")
    @classmethod
    def historical_execution_projection(cls, value: object) -> object:
        if not isinstance(value, dict) or "execution_eligibility" in value:
            return value
        projected = dict(value)
        geometry = tuple(
            projected.get(field)
            for field in ("entry", "stop_loss", "take_profit", "risk_reward_ratio")
        )
        if projected.get("signal") in {"BUY", "SELL"} and all(
            item is not None for item in geometry
        ):
            projected["execution_eligibility"] = "ELIGIBLE"
            projected["execution_status"] = "READY"
        else:
            projected["execution_eligibility"] = "INELIGIBLE"
            projected["execution_status"] = "BLOCKED"
            projected["blocking_reasons"] = tuple(
                projected.get("risk_flags") or ("historical_non_actionable_signal",)
            )
        return projected

    @model_validator(mode="after")
    def geometry_and_strength_are_consistent(self) -> AIAnalysisSignal:
        expected_strength = signal_strength(self.confidence)
        if self.strength != expected_strength:
            raise ValueError("analysis-signal strength does not match confidence")
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError("analysis-signal validity must have positive duration")
        geometry = (self.entry, self.stop_loss, self.take_profit, self.risk_reward_ratio)
        if self.signal == AnalysisSignalAction.HOLD:
            if any(value is not None for value in geometry):
                raise ValueError("HOLD analysis signal cannot contain execution geometry")
            return self
        if self.execution_eligibility == AnalysisExecutionEligibility.ELIGIBLE:
            if self.execution_status != AnalysisExecutionStatus.READY:
                raise ValueError("eligible analytical signal must be execution-ready")
            if self.blocking_reasons:
                raise ValueError("eligible analytical signal cannot retain blockers")
            if any(value is None for value in geometry):
                raise ValueError("eligible analytical signal requires complete geometry")
        else:
            if self.execution_status != AnalysisExecutionStatus.BLOCKED:
                raise ValueError("ineligible analytical signal must be blocked")
            if not self.blocking_reasons:
                raise ValueError("ineligible analytical signal requires explicit blockers")
            if all(value is None for value in geometry):
                return self
            if any(value is None for value in geometry):
                raise ValueError("partial analytical geometry is prohibited")
        assert self.entry is not None
        assert self.stop_loss is not None
        assert self.take_profit is not None
        if self.signal == AnalysisSignalAction.BUY and not (
            self.stop_loss < self.entry < self.take_profit
        ):
            raise ValueError("BUY analysis-signal geometry is invalid")
        if self.signal == AnalysisSignalAction.SELL and not (
            self.take_profit < self.entry < self.stop_loss
        ):
            raise ValueError("SELL analysis-signal geometry is invalid")
        return self


class AIAnalysisSignalOutcome(StrictAnalysisModel):
    """Latest immutable evaluation snapshot for one deterministic analysis signal."""

    schema_version: str = "1.0"
    outcome_id: UUID
    signal_id: UUID
    status: AnalysisSignalLifecycle
    entry_reached: bool
    entry_reached_at: datetime | None = None
    stop_hit: bool
    target_hit: bool
    expired: bool
    maximum_favorable_excursion: float = Field(default=0, ge=0)
    maximum_adverse_excursion: float = Field(default=0, ge=0)
    holding_time_seconds: float | None = Field(default=None, ge=0)
    actual_risk_reward: float | None = None
    profit_loss: float | None = None
    evaluated_at: datetime
    completed_at: datetime | None = None
    reason_codes: tuple[str, ...] = ()

    @field_validator("entry_reached_at", "evaluated_at", "completed_at")
    @classmethod
    def outcome_time_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("analysis-signal outcome timestamp must be timezone-aware")
        return value.astimezone(UTC)


def signal_strength(confidence: int) -> AnalysisSignalStrength:
    if confidence < 40:
        return AnalysisSignalStrength.VERY_WEAK
    if confidence < 55:
        return AnalysisSignalStrength.WEAK
    if confidence < 70:
        return AnalysisSignalStrength.MODERATE
    if confidence < 85:
        return AnalysisSignalStrength.STRONG
    return AnalysisSignalStrength.VERY_STRONG


class AnalysisReference(StrictAnalysisModel):
    analysis_id: UUID
    analysis_timestamp: datetime
    regime: RegimeClassification
    regime_strength: float
    confidence: float
    bullish_strength: float
    bearish_strength: float
    structure_summary: str
    liquidity_summary: str
    contradiction_count: int
    risk_count: int
    provider: str

    @field_validator("analysis_timestamp")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("analysis reference timestamp must be timezone-aware")
        return value.astimezone(UTC)


class TemporalDataQuality(StrEnum):
    SUFFICIENT = "sufficient"
    LIMITED = "limited"
    INSUFFICIENT_HISTORY = "insufficient_history"


class AIAnalysisTemporalContext(StrictAnalysisModel):
    version: str = "1.0"
    current_analysis_id: UUID
    as_of: datetime
    previous_analysis_id: UUID | None = None
    lookbacks: dict[str, AnalysisReference | None]
    rolling_window: tuple[AnalysisReference, ...]
    data_quality: TemporalDataQuality

    @field_validator("as_of")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("temporal boundary must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def point_in_time_safe(self) -> AIAnalysisTemporalContext:
        references = tuple(item for item in self.lookbacks.values() if item is not None) + self.rolling_window
        if any(item.analysis_timestamp >= self.as_of for item in references):
            raise ValueError("temporal context contains current or future analysis")
        return self


class ConsistencyClassification(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNSTABLE = "unstable"
    INSUFFICIENT_HISTORY = "insufficient_history"


class HistoricalConsistencyScore(StrictAnalysisModel):
    score: float = Field(ge=0, le=100)
    classification: ConsistencyClassification
    sample_size: int = Field(ge=0)
    time_range: str
    reason: str
    supporting_analysis_ids: tuple[UUID, ...]
    contradicting_analysis_ids: tuple[UUID, ...]
    data_quality: TemporalDataQuality


class TrendClassification(StrEnum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    HIGHLY_UNSTABLE = "highly_unstable"
    INSUFFICIENTLY_SAMPLED = "insufficiently_sampled"


class AnalysisMomentum(StrictAnalysisModel):
    direction: AnalysisBias
    strength: float = Field(ge=0, le=100)
    acceleration: float = Field(ge=-100, le=100)
    stability: TrendClassification
    time_range: str
    sample_count: int = Field(ge=0)
    explanation: str
    analysis_ids: tuple[UUID, ...]


class TemporalAnalysisMetrics(StrictAnalysisModel):
    version: str = "1.0"
    historical_consistency: HistoricalConsistencyScore
    analysis_momentum: AnalysisMomentum
    confidence_trend: TrendClassification
    regime_transition: str
    contradiction_trend: TrendClassification
    risk_trend: TrendClassification


class ValidatedAIAnalysis(StrictAnalysisModel):
    analysis: AIMarketAnalysis
    temporal_context: AIAnalysisTemporalContext
    temporal_metrics: TemporalAnalysisMetrics
    signal: AIAnalysisSignal | None = None


def analysis_reference(value: AIMarketAnalysis) -> AnalysisReference:
    if value.output is None:
        raise ValueError("only validated analyses may be referenced")
    bullish = _evidence_strength(value.output.bullish_evidence)
    bearish = _evidence_strength(value.output.bearish_evidence)
    return AnalysisReference(
        analysis_id=value.analysis_id,
        analysis_timestamp=value.analysis_timestamp,
        regime=value.output.market_regime.classification,
        regime_strength=value.output.market_regime.strength,
        confidence=value.output.analysis_confidence,
        bullish_strength=bullish,
        bearish_strength=bearish,
        structure_summary=value.output.market_structure.recent_change,
        liquidity_summary=value.output.liquidity_analysis.summary,
        contradiction_count=len(value.output.contradictions),
        risk_count=len(value.output.key_risks),
        provider=value.provider_metadata.provider,
    )


def _evidence_strength(values: tuple[AnalysisEvidence, ...]) -> float:
    # Count is a bounded interpretation-density measure, not a market indicator.
    return min(100.0, len(values) * 20.0)


class TemporalContextAnalyzer:
    """Deterministic analysis-history interpretation with no market forecasting."""

    def analyze(self, context: AIAnalysisTemporalContext, current: AIMarketAnalysis) -> TemporalAnalysisMetrics:
        current_ref = analysis_reference(current)
        samples = (*context.rolling_window, current_ref)
        if len(samples) < 3:
            insufficient = HistoricalConsistencyScore(
                score=50.0,
                classification=ConsistencyClassification.INSUFFICIENT_HISTORY,
                sample_size=max(0, len(samples) - 1),
                time_range=self._range(samples),
                reason="Building analysis history; temporal evidence has no blocking authority.",
                supporting_analysis_ids=(),
                contradicting_analysis_ids=(),
                data_quality=TemporalDataQuality.INSUFFICIENT_HISTORY,
            )
            return TemporalAnalysisMetrics(
                historical_consistency=insufficient,
                analysis_momentum=AnalysisMomentum(
                    direction=AnalysisBias.UNCERTAIN,
                    strength=0,
                    acceleration=0,
                    stability=TrendClassification.INSUFFICIENTLY_SAMPLED,
                    time_range=self._range(samples),
                    sample_count=len(samples),
                    explanation="Building analysis history.",
                    analysis_ids=tuple(item.analysis_id for item in samples),
                ),
                confidence_trend=TrendClassification.INSUFFICIENTLY_SAMPLED,
                regime_transition="insufficient_history",
                contradiction_trend=TrendClassification.INSUFFICIENTLY_SAMPLED,
                risk_trend=TrendClassification.INSUFFICIENTLY_SAMPLED,
            )

        recent = samples[-20:]
        current_regime = current_ref.regime
        supporting = tuple(item.analysis_id for item in recent[:-1] if item.regime == current_regime)
        contradicting = tuple(
            item.analysis_id
            for item in recent[:-1]
            if {item.regime, current_regime} == {RegimeClassification.BULLISH, RegimeClassification.BEARISH}
        )
        agreement = len(supporting) / max(1, len(recent) - 1)
        regime_changes = sum(
            left.regime != right.regime
            for left, right in zip(recent, recent[1:], strict=False)
        )
        oscillation_penalty = min(40.0, regime_changes * 8.0)
        confidence_volatility = pstdev(item.confidence for item in recent) if len(recent) > 1 else 0.0
        score = max(0.0, min(100.0, 45.0 + agreement * 55.0 - oscillation_penalty - confidence_volatility * 25.0))
        if score >= 75:
            classification = ConsistencyClassification.HIGH
        elif score >= 55:
            classification = ConsistencyClassification.MODERATE
        elif regime_changes >= max(2, len(recent) // 2):
            classification = ConsistencyClassification.UNSTABLE
        else:
            classification = ConsistencyClassification.LOW

        directional = [item.bullish_strength - item.bearish_strength for item in recent]
        first_half = fmean(directional[: max(1, len(directional) // 2)])
        second_half = fmean(directional[max(1, len(directional) // 2) :])
        delta = second_half - first_half
        recent_delta = directional[-1] - directional[-2]
        prior_delta = directional[-2] - directional[-3]
        acceleration = max(-100.0, min(100.0, recent_delta - prior_delta))
        if abs(delta) < 5:
            momentum_direction = AnalysisBias.NEUTRAL
        else:
            momentum_direction = AnalysisBias.BULLISH if delta > 0 else AnalysisBias.BEARISH
        momentum_stability = (
            TrendClassification.HIGHLY_UNSTABLE
            if pstdev(directional) > 35
            else TrendClassification.STABLE
        )

        return TemporalAnalysisMetrics(
            historical_consistency=HistoricalConsistencyScore(
                score=round(score, 2),
                classification=classification,
                sample_size=len(recent) - 1,
                time_range=self._range(recent),
                reason=(
                    f"{len(supporting)} of {len(recent) - 1} prior analyses agree with the "
                    f"current {current_regime.value} regime; {regime_changes} regime transitions observed."
                ),
                supporting_analysis_ids=supporting,
                contradicting_analysis_ids=contradicting,
                data_quality=(
                    TemporalDataQuality.SUFFICIENT
                    if len(recent) >= 5
                    else TemporalDataQuality.LIMITED
                ),
            ),
            analysis_momentum=AnalysisMomentum(
                direction=momentum_direction,
                strength=round(min(100.0, abs(delta)), 2),
                acceleration=round(acceleration, 2),
                stability=momentum_stability,
                time_range=self._range(recent),
                sample_count=len(recent),
                explanation=f"AI interpretation balance changed by {delta:.1f} points across the bounded window.",
                analysis_ids=tuple(item.analysis_id for item in recent),
            ),
            confidence_trend=self._trend([item.confidence for item in recent], scale=100),
            regime_transition=f"{recent[-2].regime.value}_to_{recent[-1].regime.value}",
            contradiction_trend=self._trend([float(item.contradiction_count) for item in recent]),
            risk_trend=self._trend([float(item.risk_count) for item in recent]),
        )

    @staticmethod
    def _trend(values: list[float], *, scale: float = 1.0) -> TrendClassification:
        if len(values) < 3:
            return TrendClassification.INSUFFICIENTLY_SAMPLED
        normalized = [item * scale for item in values]
        if pstdev(normalized) > 25:
            return TrendClassification.HIGHLY_UNSTABLE
        delta = fmean(normalized[-2:]) - fmean(normalized[:2])
        if delta > 3:
            return TrendClassification.INCREASING
        if delta < -3:
            return TrendClassification.DECREASING
        return TrendClassification.STABLE

    @staticmethod
    def _range(values: tuple[AnalysisReference, ...]) -> str:
        if len(values) < 2:
            return "0m"
        minutes = int((values[-1].analysis_timestamp - values[0].analysis_timestamp).total_seconds() / 60)
        return f"{max(0, minutes)}m"


DEFAULT_TEMPORAL_ANCHORS: dict[str, timedelta] = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}
