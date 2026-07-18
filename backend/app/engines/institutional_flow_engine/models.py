from datetime import UTC, datetime
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.engines.market_data_engine import Timeframe


def stable_id(*parts: object) -> UUID:
    return uuid5(NAMESPACE_URL, "ten:institutional-flow:" + ":".join(str(part) for part in parts))


def utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


class FlowDirection(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    INDETERMINATE = "indeterminate"


class FlowState(StrEnum):
    STRONG_BULLISH = "strong_bullish_pressure"
    MODERATE_BULLISH = "moderate_bullish_pressure"
    BALANCED = "balanced"
    MODERATE_BEARISH = "moderate_bearish_pressure"
    STRONG_BEARISH = "strong_bearish_pressure"
    INDETERMINATE = "indeterminate"


class ParticipationLevel(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    INDETERMINATE = "indeterminate"


class ActivityType(StrEnum):
    INITIATIVE = "initiative"
    RESPONSIVE = "responsive"
    MIXED = "mixed"
    NONE = "none"


class InitiativeResponsiveType(StrEnum):
    INITIATIVE = "initiative"
    RESPONSIVE = "responsive"
    AMBIGUOUS = "ambiguous"


class AbsorptionType(StrEnum):
    BULLISH = "bullish_absorption_like"
    BEARISH = "bearish_absorption_like"
    MIXED = "mixed_absorption_like"
    NONE = "none"


class ExhaustionType(StrEnum):
    BULLISH = "bullish_exhaustion_like"
    BEARISH = "bearish_exhaustion_like"
    MIXED = "mixed_exhaustion_like"
    NONE = "none"


class InventoryBehaviorType(StrEnum):
    ACCUMULATION = "accumulation_like"
    DISTRIBUTION = "distribution_like"
    REACCUMULATION = "reaccumulation_like"
    REDISTRIBUTION = "redistribution_like"
    BALANCE = "ordinary_balance"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT = "insufficient_evidence"


class CampaignPhase(StrEnum):
    PREPARATION = "preparation"
    ACCUMULATION = "accumulation_like"
    MARKUP = "markup_like"
    REACCUMULATION = "reaccumulation_like"
    DISTRIBUTION = "distribution_like"
    MARKDOWN = "markdown_like"
    REDISTRIBUTION = "redistribution_like"
    TRANSITION = "transition"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT = "insufficient_evidence"


class FlowPersistenceState(StrEnum):
    TRANSIENT = "transient"
    DEVELOPING = "developing"
    PERSISTENT = "persistent"
    WEAKENING = "weakening"
    STRENGTHENING = "strengthening"
    REVERSING = "reversing"
    INVALIDATED = "invalidated"


class EvidenceSourceEngine(StrEnum):
    MARKET_DATA = "market_data"
    SMC = "smc"
    LIQUIDITY = "liquidity"
    VOLUME_PROFILE = "volume_profile"


class EvidenceType(StrEnum):
    RANGE_EXPANSION = "range_expansion"
    VOLUME_EXPANSION = "volume_expansion"
    DIRECTIONAL_PERSISTENCE = "directional_persistence"
    DISPLACEMENT = "displacement"
    STRUCTURAL_BREAK = "structural_break"
    STRUCTURAL_FAILURE = "structural_failure"
    VALUE_REJECTION = "value_rejection"
    VALUE_ACCEPTANCE = "value_acceptance"
    PROFILE_MIGRATION = "profile_migration"
    NODE_INTERACTION = "node_interaction"
    LIQUIDITY_EVENT = "liquidity_event"
    LIMITED_PROGRESS = "limited_progress"
    REPEATED_TEST = "repeated_test"
    EFFICIENCY_DECLINE = "efficiency_decline"
    SESSION_CONTINUITY = "session_continuity"
    CUSTOM = "custom"


class EvidenceRole(StrEnum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    INVALIDATING = "invalidating"
    CONTEXT = "context"


class CorrelationGroup(StrEnum):
    PRICE_ACTION = "price_action"
    STRUCTURE = "structure"
    LIQUIDITY = "liquidity"
    PROFILE = "profile"
    VOLUME = "volume"
    SESSION = "session"
    INDEPENDENT = "independent"


class ConflictType(StrEnum):
    DIRECTIONAL = "directional"
    INITIATIVE_RESPONSIVE = "initiative_responsive"
    INVENTORY = "inventory"
    TIMEFRAME = "timeframe"
    QUALITY = "quality"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DataQualityLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNUSABLE = "unusable"


class AnalysisStatus(StrEnum):
    COMPLETE = "complete"
    DEGRADED = "degraded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ProcessingMode(StrEnum):
    HISTORICAL = "historical"
    REALTIME = "realtime"
    INCREMENTAL = "incremental"
    REPLAY = "replay"
    RECOVERY = "recovery"


class LifecycleState(StrEnum):
    INITIALIZED = "initialized"
    ACTIVE = "active"
    PERSISTENT = "persistent"
    WEAKENING = "weakening"
    INVALIDATED = "invalidated"
    COMPLETED = "completed"
    DEGRADED = "degraded"


class SessionType(StrEnum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "london_new_york_overlap"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class InvalidationReason(StrEnum):
    STRUCTURAL = "structural_invalidation"
    CONTRADICTORY = "contradictory_evidence"
    STALE = "stale_evidence"
    LOW_QUALITY = "low_quality"
    SOURCE_INVALIDATED = "source_invalidated"


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class InstitutionalFlowEvidence(ImmutableModel):
    id: UUID
    source_engine: EvidenceSourceEngine
    evidence_type: EvidenceType
    source_object_id: str
    source_timestamp: datetime
    availability_timestamp: datetime
    timeframe: Timeframe
    session: SessionType = SessionType.UNKNOWN
    direction: FlowDirection
    strength: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=1)
    role: EvidenceRole = EvidenceRole.SUPPORTING
    correlation_group: CorrelationGroup = CorrelationGroup.INDEPENDENT
    invalidated: bool = False
    explanation: str
    configuration_version: str
    engine_version: str

    @field_validator("source_timestamp", "availability_timestamp")
    @classmethod
    def timezone(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def available_after_source(self) -> "InstitutionalFlowEvidence":
        if self.availability_timestamp < self.source_timestamp:
            raise ValueError("evidence availability cannot precede its source timestamp")
        return self


class InstitutionalFlowEvidenceBundle(ImmutableModel):
    accepted: tuple[InstitutionalFlowEvidence, ...] = ()
    rejected_future_ids: tuple[UUID, ...] = ()
    rejected_invalid_ids: tuple[UUID, ...] = ()
    deduplicated_ids: tuple[UUID, ...] = ()
    discounted_ids: tuple[UUID, ...] = ()


class InstitutionalFlowQuality(ImmutableModel):
    level: DataQualityLevel
    score: float = Field(ge=0, le=1)
    source_diversity: int = Field(ge=0)
    limitations: tuple[str, ...] = ()


class InstitutionalFlowExplanation(ImmutableModel):
    summary: str
    supporting_evidence_ids: tuple[UUID, ...] = ()
    contradicting_evidence_ids: tuple[UUID, ...] = ()
    alternative_interpretation: str | None = None


class ParticipationIntensity(ImmutableModel):
    score: float = Field(ge=0, le=1)
    level: ParticipationLevel
    direction: FlowDirection
    persistence: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=1)
    evidence_ids: tuple[UUID, ...]
    ambiguity: float = Field(ge=0, le=1)


class InitiativeActivity(ImmutableModel):
    direction: FlowDirection
    initiation_timestamp: datetime
    strength: float = Field(ge=0, le=1)
    continuation: float = Field(ge=0, le=1)
    structural_consequence: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    invalidated: bool = False
    evidence_ids: tuple[UUID, ...] = ()


class ResponsiveActivity(ImmutableModel):
    defended_reference: str | None
    direction: FlowDirection
    strength: float = Field(ge=0, le=1)
    persistence: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    invalidated: bool = False
    evidence_ids: tuple[UUID, ...] = ()


class AbsorptionInference(ImmutableModel):
    absorption_type: AbsorptionType
    absorbed_pressure: FlowDirection
    defending_side: FlowDirection
    price_zone: tuple[float, float] | None = None
    test_count: int = Field(ge=0)
    estimated_intensity: float = Field(ge=0, le=1)
    efficiency_change: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    ambiguity: float = Field(ge=0, le=1)
    invalidated: bool = False
    evidence_ids: tuple[UUID, ...] = ()


class ExhaustionInference(ImmutableModel):
    exhaustion_type: ExhaustionType
    exhausted_direction: FlowDirection
    onset: datetime
    strength: float = Field(ge=0, le=1)
    persistence: float = Field(ge=0, le=1)
    reversal_evidence: float = Field(ge=0, le=1)
    ambiguity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: tuple[UUID, ...] = ()


class AccumulationDistributionInference(ImmutableModel):
    behavior: InventoryBehaviorType
    direction: FlowDirection
    strength: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    ambiguity: float = Field(ge=0, le=1)
    evidence_family_count: int = Field(ge=0)
    alternative_interpretation: str | None = None
    evidence_ids: tuple[UUID, ...] = ()


class CampaignPhaseInference(ImmutableModel):
    phase: CampaignPhase
    previous_phase: CampaignPhase | None = None
    confidence: float = Field(ge=0, le=1)
    ambiguity: float = Field(ge=0, le=1)
    explanation: str
    evidence_ids: tuple[UUID, ...] = ()


class DirectionalFlowPressure(ImmutableModel):
    bullish_weight: float = Field(ge=0)
    bearish_weight: float = Field(ge=0)
    neutral_weight: float = Field(ge=0)
    net_pressure: float = Field(ge=-1, le=1)
    state: FlowState
    evidence_diversity: int = Field(ge=0)
    conflict: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=1)
    persistence: float = Field(ge=0, le=1)


class FlowPersistence(ImmutableModel):
    state: FlowPersistenceState
    score: float = Field(ge=0, le=1)
    window_observations: int = Field(ge=0)
    decay_factor: float = Field(ge=0, le=1)
    direction_changes: int = Field(ge=0)


class CrossSessionFlow(ImmutableModel):
    previous_session: SessionType
    current_session: SessionType
    relationship: str
    direction: FlowDirection
    strength: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    completed: bool
    evidence_ids: tuple[UUID, ...] = ()


class MultiTimeframeInstitutionalFlow(ImmutableModel):
    requested_timeframe: Timeframe
    direction_by_timeframe: dict[str, FlowDirection]
    aligned: bool
    conflict: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    analyzed_through: datetime
    maximum_depth: int = Field(ge=1, le=9)


class InstitutionalFlowConfluence(ImmutableModel):
    id: UUID
    source_evidence_ids: tuple[UUID, ...]
    source_engines: tuple[EvidenceSourceEngine, ...]
    direction: FlowDirection
    raw_score: float = Field(ge=0)
    correlation_discount: float = Field(ge=0, le=1)
    adjusted_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class InstitutionalFlowTransition(ImmutableModel):
    id: UUID
    previous_state: FlowState
    current_state: FlowState
    available_at: datetime
    reason: str


class InstitutionalFlowState(ImmutableModel):
    id: UUID
    lifecycle_state: LifecycleState
    participation: ParticipationIntensity
    initiative: InitiativeActivity | None = None
    responsive: ResponsiveActivity | None = None
    absorption: AbsorptionInference | None = None
    exhaustion: ExhaustionInference | None = None
    inventory: AccumulationDistributionInference
    campaign: CampaignPhaseInference
    pressure: DirectionalFlowPressure
    persistence: FlowPersistence
    cross_session: tuple[CrossSessionFlow, ...] = ()
    confluences: tuple[InstitutionalFlowConfluence, ...] = ()
    explanation: InstitutionalFlowExplanation
    version: int = Field(ge=1)


class InstitutionalFlowCheckpoint(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    snapshot_id: UUID
    last_processed_candle: datetime
    configuration_version: str
    engine_version: str
    state_hash: str = Field(min_length=64, max_length=64)


class InstitutionalFlowAnalysisSnapshot(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    session: SessionType
    analysis_timestamp: datetime
    availability_timestamp: datetime
    processing_mode: ProcessingMode
    status: AnalysisStatus
    state: InstitutionalFlowState
    evidence: InstitutionalFlowEvidenceBundle
    quality: InstitutionalFlowQuality
    multi_timeframe: MultiTimeframeInstitutionalFlow | None = None
    transitions: tuple[InstitutionalFlowTransition, ...] = ()
    configuration_version: str
    engine_version: str
    market_data_boundary: str
    upstream_versions: dict[str, str]
    created_at: datetime

    @field_validator("analysis_timestamp", "availability_timestamp", "created_at")
    @classmethod
    def timezone(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def temporal_boundary(self) -> "InstitutionalFlowAnalysisSnapshot":
        if self.availability_timestamp > self.analysis_timestamp:
            raise ValueError("snapshot availability cannot exceed analysis boundary")
        return self


# Backward-compatible pipeline result.
class FlowBias(StrEnum):
    BUYING = "buying"
    SELLING = "selling"
    BALANCED = "balanced"


class FlowScore(BaseModel):
    score: float = Field(ge=-1, le=1)
    bias: FlowBias
    volume_pressure: float = Field(ge=-1, le=1)
    price_acceleration: float = Field(ge=-1, le=1)
    delta_estimate: float = Field(ge=-1, le=1)
    absorption_probability: float = Field(ge=0, le=1)
    methodology: str = "OHLCV inference; not exchange order flow and not verified institutional identity"
    observations: list[str] = Field(default_factory=list)
