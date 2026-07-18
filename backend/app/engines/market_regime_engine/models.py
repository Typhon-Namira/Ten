"""Public, replay-stable Market Regime value objects."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.engines.market_data_engine import Timeframe


def stable_id(*parts: object) -> UUID:
    return uuid5(NAMESPACE_URL, "ten:market-regime:" + ":".join(str(part) for part in parts))


def utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DominantRegime(StrEnum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    BALANCED = "balanced"
    IMBALANCED_BULL = "imbalanced_bull"
    IMBALANCED_BEAR = "imbalanced_bear"
    ACCUMULATION_LIKE = "accumulation_like"
    DISTRIBUTION_LIKE = "distribution_like"
    MARKUP_LIKE = "markup_like"
    MARKDOWN_LIKE = "markdown_like"
    REACCUMULATION_LIKE = "reaccumulation_like"
    REDISTRIBUTION_LIKE = "redistribution_like"
    COMPRESSION = "compression"
    EXPANSION_BULL = "expansion_bull"
    EXPANSION_BEAR = "expansion_bear"
    TRANSITION = "transition"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_DATA = "insufficient_data"


class TrendRegime(StrEnum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGE = "range"
    NEUTRAL = "neutral"
    TRANSITION = "transition"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_DATA = "insufficient_data"


class VolatilityRegime(StrEnum):
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXPANDING = "expanding"
    CONTRACTING = "contracting"
    UNSTABLE = "unstable"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_DATA = "insufficient_data"


class AuctionRegime(StrEnum):
    BALANCED_AUCTION = "balanced_auction"
    BULLISH_IMBALANCE = "bullish_imbalance"
    BEARISH_IMBALANCE = "bearish_imbalance"
    MIXED_AUCTION = "mixed_auction"
    TRANSITION = "transition"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_DATA = "insufficient_data"


class ExpansionRegime(StrEnum):
    COMPRESSION = "compression"
    EARLY_EXPANSION = "early_expansion"
    EXPANSION = "expansion"
    LATE_EXPANSION = "late_expansion"
    DECELERATION = "deceleration"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_DATA = "insufficient_data"


class StructuralRegime(StrEnum):
    BULLISH_CONTINUATION = "bullish_continuation"
    BEARISH_CONTINUATION = "bearish_continuation"
    BULLISH_REVERSAL_ATTEMPT = "bullish_reversal_attempt"
    BEARISH_REVERSAL_ATTEMPT = "bearish_reversal_attempt"
    RANGE_STRUCTURE = "range_structure"
    STRUCTURAL_TRANSITION = "structural_transition"
    MIXED_STRUCTURE = "mixed_structure"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_DATA = "insufficient_data"


class ParticipationRegime(StrEnum):
    STRONG_BULLISH_PARTICIPATION = "strong_bullish_participation"
    MODERATE_BULLISH_PARTICIPATION = "moderate_bullish_participation"
    NEUTRAL_PARTICIPATION = "neutral_participation"
    MODERATE_BEARISH_PARTICIPATION = "moderate_bearish_participation"
    STRONG_BEARISH_PARTICIPATION = "strong_bearish_participation"
    DECLINING_PARTICIPATION = "declining_participation"
    CONFLICTED_PARTICIPATION = "conflicted_participation"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_DATA = "insufficient_data"


class InventoryRegime(StrEnum):
    PREPARATION_LIKE = "preparation_like"
    ACCUMULATION_LIKE = "accumulation_like"
    MARKUP_LIKE = "markup_like"
    REACCUMULATION_LIKE = "reaccumulation_like"
    DISTRIBUTION_LIKE = "distribution_like"
    MARKDOWN_LIKE = "markdown_like"
    REDISTRIBUTION_LIKE = "redistribution_like"
    TRANSITION_LIKE = "transition_like"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_DATA = "insufficient_data"


class RegimeLifecycle(StrEnum):
    INITIAL = "initial"
    DEVELOPING = "developing"
    CONFIRMED = "confirmed"
    MATURE = "mature"
    WEAKENING = "weakening"
    TRANSITIONING = "transitioning"
    INVALIDATED = "invalidated"
    RECOVERING = "recovering"
    INSUFFICIENT_DATA = "insufficient_data"


class RegimePersistence(StrEnum):
    TRANSIENT = "transient"
    DEVELOPING = "developing"
    PERSISTENT = "persistent"
    STRENGTHENING = "strengthening"
    STABLE = "stable"
    WEAKENING = "weakening"
    REVERSING = "reversing"
    UNSTABLE = "unstable"
    INSUFFICIENT_DATA = "insufficient_data"


class TrendMaturity(StrEnum):
    EARLY = "early"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    MATURE = "mature"
    LATE = "late"
    WEAKENING = "weakening"
    EXHAUSTION_RISK = "exhaustion_risk"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_DATA = "insufficient_data"


class TransitionState(StrEnum):
    NONE = "none"
    WATCH = "watch"
    DEVELOPING = "developing"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class EvidenceDirection(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class EvidenceRole(StrEnum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    CONTEXT = "context"


class EvidenceFamily(StrEnum):
    MARKET_DATA = "market_data"
    STRUCTURE = "structure"
    LIQUIDITY = "liquidity"
    VOLUME_PROFILE = "volume_profile"
    INSTITUTIONAL_FLOW = "institutional_flow"
    VOLATILITY = "volatility"
    SESSION = "session"
    MULTI_TIMEFRAME = "multi_timeframe"
    PERSISTENCE = "persistence"
    TRANSITION = "transition"


class ProcessingMode(StrEnum):
    SNAPSHOT = "snapshot"
    INCREMENTAL = "incremental"
    REPLAY = "replay"
    RECOVERY = "recovery"


class MarketRegimeEvidence(ImmutableModel):
    evidence_id: UUID
    source_engine: str
    source_engine_version: str
    source_object_type: str
    source_object_id: str
    source_snapshot_id: str | None = None
    symbol: str
    timeframe: Timeframe
    session: str = "unknown"
    event_timestamp: datetime
    available_at: datetime
    analysis_boundary: datetime
    direction: EvidenceDirection
    role: EvidenceRole = EvidenceRole.SUPPORTING
    family: EvidenceFamily
    subfamily: str
    raw_strength: float = Field(ge=0)
    normalized_strength: float = Field(ge=0, le=1)
    source_confidence: float = Field(ge=0, le=1)
    source_quality: float = Field(ge=0, le=1)
    effective_weight: float = Field(ge=0, le=1)
    correlation_group: str
    correlation_discount: float = Field(ge=0, le=1)
    decay_factor: float = Field(ge=0, le=1)
    accepted: bool = True
    rejected: bool = False
    discounted: bool = False
    contradicting: bool = False
    unavailable: bool = False
    rejection_reason: str | None = None
    payload_summary: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_timestamp", "available_at", "analysis_boundary")
    @classmethod
    def timezone(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def temporal_state(self) -> "MarketRegimeEvidence":
        if self.available_at < self.event_timestamp:
            raise ValueError("evidence availability cannot precede event time")
        if self.accepted and self.available_at > self.analysis_boundary:
            raise ValueError("future evidence cannot be accepted")
        return self


class MultiTimeframeRegimeState(ImmutableModel):
    requested_timeframe: str
    included_timeframes: tuple[str, ...] = ()
    excluded_timeframes: tuple[str, ...] = ()
    unavailable_timeframes: tuple[str, ...] = ()
    dominant_timeframe: str | None = None
    higher_timeframe_regime: DominantRegime | None = None
    lower_timeframe_regime: DominantRegime | None = None
    alignment_score: float = Field(ge=0, le=1)
    conflict_score: float = Field(ge=0, le=1)
    directional_alignment: float = Field(ge=0, le=1)
    volatility_alignment: float = Field(ge=0, le=1)
    auction_alignment: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    ambiguity: float = Field(ge=0, le=1)
    explanation: str


class CrossSessionRegimeState(ImmutableModel):
    current_session: str
    previous_session: str | None = None
    continuation_score: float = Field(ge=0, le=1)
    handoff_score: float = Field(ge=0, le=1)
    reversal_score: float = Field(ge=0, le=1)
    session_alignment: str
    dominant_session: str | None = None
    confidence: float = Field(ge=0, le=1)
    ambiguity: float = Field(ge=0, le=1)
    explanation: str


class RegimeExplanation(ImmutableModel):
    headline: str
    primary_interpretation: str
    alternative_interpretation: str
    accepted_evidence: tuple[UUID, ...] = ()
    rejected_evidence: tuple[UUID, ...] = ()
    discounted_evidence: tuple[UUID, ...] = ()
    contradicting_evidence: tuple[UUID, ...] = ()
    unavailable_evidence: tuple[UUID, ...] = ()
    confidence_components: dict[str, float] = Field(default_factory=dict)
    ambiguity_components: dict[str, float] = Field(default_factory=dict)
    quality_components: dict[str, float] = Field(default_factory=dict)
    limitations: tuple[str, ...] = ()


class RegimeTransition(ImmutableModel):
    transition_id: UUID
    symbol: str
    timeframe: Timeframe
    from_regime: DominantRegime
    to_regime: DominantRegime
    started_at: datetime
    confirmed_at: datetime | None = None
    invalidated_at: datetime | None = None
    state: TransitionState
    confidence: float = Field(ge=0, le=1)
    ambiguity: float = Field(ge=0, le=1)
    supporting_evidence_ids: tuple[UUID, ...] = ()
    contradicting_evidence_ids: tuple[UUID, ...] = ()
    reasoning_summary: str

    @field_validator("started_at", "confirmed_at", "invalidated_at")
    @classmethod
    def timezone(cls, value: datetime | None) -> datetime | None:
        return utc(value) if value else None


class DegradationState(ImmutableModel):
    is_degraded: bool = False
    missing_dependencies: tuple[str, ...] = ()
    failed_dependencies: tuple[str, ...] = ()
    stale_dependencies: tuple[str, ...] = ()
    disabled_dependencies: tuple[str, ...] = ()
    degradation_reasons: tuple[str, ...] = ()
    confidence_penalty: float = Field(0, ge=0, le=1)


class MarketRegimeSnapshot(ImmutableModel):
    snapshot_id: UUID
    engine_name: str = "market_regime"
    engine_version: str
    schema_version: str
    configuration_version: str
    algorithm_version: str
    symbol: str
    timeframe: Timeframe
    analysis_timestamp: datetime
    historical_boundary: datetime
    created_at: datetime
    dominant_regime: DominantRegime
    trend_regime: TrendRegime
    volatility_regime: VolatilityRegime
    auction_regime: AuctionRegime
    expansion_regime: ExpansionRegime
    structural_regime: StructuralRegime
    participation_regime: ParticipationRegime
    inventory_regime: InventoryRegime
    lifecycle: RegimeLifecycle
    persistence: RegimePersistence
    trend_maturity: TrendMaturity
    directional_bias: EvidenceDirection
    bullish_score: float = Field(ge=0, le=1)
    bearish_score: float = Field(ge=0, le=1)
    neutral_score: float = Field(ge=0, le=1)
    net_directional_score: float = Field(ge=-1, le=1)
    balance_score: float = Field(ge=0, le=1)
    imbalance_score: float = Field(ge=0, le=1)
    compression_score: float = Field(ge=0, le=1)
    expansion_score: float = Field(ge=0, le=1)
    trend_strength: float = Field(ge=0, le=1)
    volatility_score: float = Field(ge=0, le=1)
    volatility_percentile: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=1)
    ambiguity: float = Field(ge=0, le=1)
    conflict_score: float = Field(ge=0, le=1)
    evidence_diversity: float = Field(ge=0, le=1)
    source_diversity: float = Field(ge=0, le=1)
    primary_interpretation: str
    alternative_interpretation: str
    reasoning_summary: str
    transition_state: TransitionState
    previous_dominant_regime: DominantRegime | None = None
    transition_score: float = Field(ge=0, le=1)
    transition_started_at: datetime | None = None
    transition_confirmed_at: datetime | None = None
    multi_timeframe: MultiTimeframeRegimeState
    cross_session: CrossSessionRegimeState
    evidence: tuple[MarketRegimeEvidence, ...] = ()
    confidence_components: dict[str, float] = Field(default_factory=dict)
    ambiguity_components: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    degradation: DegradationState = Field(default_factory=DegradationState)
    repository_mode: str
    recovery_state: str
    processing_mode: ProcessingMode
    probabilistic_inference: bool = True
    trading_instruction: bool = False

    @field_validator("analysis_timestamp", "historical_boundary", "created_at", "transition_started_at", "transition_confirmed_at")
    @classmethod
    def timezone(cls, value: datetime | None) -> datetime | None:
        return utc(value) if value else None

    @model_validator(mode="after")
    def safeguards(self) -> "MarketRegimeSnapshot":
        if not self.probabilistic_inference or self.trading_instruction:
            raise ValueError("Market Regime outputs must remain probabilistic and non-instructional")
        if self.analysis_timestamp != self.historical_boundary:
            raise ValueError("analysis timestamp must equal historical boundary")
        return self


class MarketRegimeCheckpoint(ImmutableModel):
    checkpoint_id: UUID
    engine_name: str
    engine_version: str
    schema_version: str
    configuration_version: str
    algorithm_version: str
    symbol: str
    timeframe: Timeframe
    analysis_boundary: datetime
    state_payload: dict[str, object]
    payload_hash: str = Field(min_length=64, max_length=64)
    created_at: datetime

    @field_validator("analysis_boundary", "created_at")
    @classmethod
    def timezone(cls, value: datetime) -> datetime:
        return utc(value)


# Compatibility aliases retained for existing imports.
MarketRegime = DominantRegime
RegimeEvidence = MarketRegimeEvidence
MarketRegimeResult = MarketRegimeSnapshot
