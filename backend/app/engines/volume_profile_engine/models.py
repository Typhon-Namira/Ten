from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid5, NAMESPACE_URL

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.engines.market_data_engine import Timeframe


def stable_id(*parts: object) -> UUID:
    return uuid5(NAMESPACE_URL, "ten:volume-profile:" + ":".join(str(x) for x in parts))


def utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


class ProfileType(StrEnum):
    DEVELOPING = "developing"
    FIXED_RANGE = "fixed_range"
    SESSION = "session"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    COMPOSITE = "composite"
    ANCHORED = "anchored"


class ProfileStatus(StrEnum):
    DEVELOPING = "developing"
    COMPLETED = "completed"
    DEGRADED = "degraded"


class ProfileSkipReason(StrEnum):
    INSUFFICIENT_VOLUME_PROFILE_DATA = "insufficient_volume_profile_data"
    EMPTY_PROFILE_PERIOD = "empty_profile_period"


class ProfileLifecycleState(StrEnum):
    INITIALIZED = "initialized"
    DEVELOPING = "developing"
    ACTIVE = "active"
    COMPLETED = "completed"
    PUBLISHED = "published"
    TESTED = "tested"
    PARTIALLY_INVALIDATED = "partially_invalidated"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    ARCHIVED = "archived"
    DEGRADED = "degraded"
    FAILED = "failed"


_TRANSITIONS = {
    ProfileLifecycleState.INITIALIZED: {
        ProfileLifecycleState.DEVELOPING,
        ProfileLifecycleState.COMPLETED,
        ProfileLifecycleState.DEGRADED,
        ProfileLifecycleState.FAILED,
    },
    ProfileLifecycleState.DEVELOPING: {
        ProfileLifecycleState.ACTIVE,
        ProfileLifecycleState.COMPLETED,
        ProfileLifecycleState.DEGRADED,
        ProfileLifecycleState.FAILED,
    },
    ProfileLifecycleState.ACTIVE: {
        ProfileLifecycleState.COMPLETED,
        ProfileLifecycleState.TESTED,
        ProfileLifecycleState.PARTIALLY_INVALIDATED,
        ProfileLifecycleState.INVALIDATED,
        ProfileLifecycleState.EXPIRED,
    },
    ProfileLifecycleState.COMPLETED: {ProfileLifecycleState.PUBLISHED, ProfileLifecycleState.TESTED, ProfileLifecycleState.ARCHIVED},
    ProfileLifecycleState.PUBLISHED: {
        ProfileLifecycleState.TESTED,
        ProfileLifecycleState.PARTIALLY_INVALIDATED,
        ProfileLifecycleState.INVALIDATED,
        ProfileLifecycleState.EXPIRED,
        ProfileLifecycleState.ARCHIVED,
    },
    ProfileLifecycleState.TESTED: {
        ProfileLifecycleState.PARTIALLY_INVALIDATED,
        ProfileLifecycleState.INVALIDATED,
        ProfileLifecycleState.EXPIRED,
        ProfileLifecycleState.ARCHIVED,
    },
    ProfileLifecycleState.PARTIALLY_INVALIDATED: {ProfileLifecycleState.INVALIDATED, ProfileLifecycleState.EXPIRED, ProfileLifecycleState.ARCHIVED},
    ProfileLifecycleState.INVALIDATED: {ProfileLifecycleState.ARCHIVED},
    ProfileLifecycleState.EXPIRED: {ProfileLifecycleState.ARCHIVED},
    ProfileLifecycleState.ARCHIVED: set(),
    ProfileLifecycleState.DEGRADED: {ProfileLifecycleState.FAILED, ProfileLifecycleState.ARCHIVED},
    ProfileLifecycleState.FAILED: {ProfileLifecycleState.ARCHIVED},
}


def validate_transition(previous: ProfileLifecycleState, current: ProfileLifecycleState) -> None:
    if previous != current and current not in _TRANSITIONS[previous]:
        raise ValueError(f"impossible profile lifecycle transition: {previous} -> {current}")


class VolumeSourceType(StrEnum):
    EXCHANGE = "exchange"
    BROKER = "broker"
    TICK = "tick"
    SYNTHETIC = "synthetic"
    MISSING = "missing"
    UNKNOWN = "unknown"


class VolumeAllocationMethod(StrEnum):
    CLOSE = "close"
    TYPICAL_PRICE = "typical_price"
    UNIFORM_RANGE = "uniform_range"
    BODY_WICK = "body_wick"


class PriceGridMethod(StrEnum):
    TICK = "tick"
    FIXED = "fixed"
    ROWS = "rows"
    PERCENTAGE = "percentage"
    ATR = "atr"
    AUTO = "auto"


class ValueAreaMethod(StrEnum):
    POC_EXPANSION = "poc_expansion"


class NodeType(StrEnum):
    HVN = "hvn"
    LVN = "lvn"


class ProfileShapeType(StrEnum):
    D_SHAPED = "d_shaped"
    P_SHAPED = "p_shaped"
    B_SHAPED = "b_shaped"
    DOUBLE_DISTRIBUTION = "double_distribution"
    TREND = "trend"
    THIN = "thin"
    MULTIMODAL = "multimodal"
    UNDEFINED = "undefined"


class ProfileMigrationType(StrEnum):
    UPWARD = "upward"
    DOWNWARD = "downward"
    STABLE = "stable"
    EXPANSION = "expansion"
    CONTRACTION = "contraction"
    ABRUPT = "abrupt"
    GRADUAL = "gradual"


class AnchorType(StrEnum):
    EXPLICIT = "explicit_timestamp"
    SMC_SWING = "smc_swing"
    BOS = "bos"
    CHOCH = "choch"
    MSS = "mss"
    DISPLACEMENT = "displacement"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    LIQUIDITY_RAID = "liquidity_raid"
    SESSION_START = "session_start"
    PERIOD_START = "period_start"


class ProcessingMode(StrEnum):
    HISTORICAL = "historical"
    REALTIME = "realtime"
    INCREMENTAL = "incremental"
    REPLAY = "replay"
    RECOVERY = "recovery"
    FIXED_RANGE = "fixed_range"
    COMPOSITE = "composite"
    ANCHORED = "anchored"


class AnalysisStatus(StrEnum):
    COMPLETE = "complete"
    DEGRADED = "degraded"
    INSUFFICIENT_HISTORY = "insufficient_history"


class SessionType(StrEnum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "london_new_york_overlap"
    CUSTOM = "custom"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DataQualityLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNUSABLE = "unusable"


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SkippedProfilePeriod(ImmutableModel):
    symbol: str
    timeframe: Timeframe
    profile_type: ProfileType
    period_key: str
    reason: ProfileSkipReason
    input_count: int = Field(ge=0)
    usable_count: int = Field(ge=0)
    analysis_boundary: datetime

    @field_validator("analysis_boundary")
    @classmethod
    def timezone(cls, value: datetime) -> datetime:
        return utc(value)


class VolumeDataQuality(ImmutableModel):
    source_type: VolumeSourceType
    quality_level: DataQualityLevel
    score: float = Field(ge=0, le=100)
    usable_volume_ratio: float = Field(ge=0, le=1)
    missing_observations: int = Field(ge=0)
    invalid_observations: int = Field(ge=0)
    limitations: tuple[str, ...] = ()


class VolumeProfileEvidence(ImmutableModel):
    code: str
    passed: bool
    value: float | str
    threshold: float | str | None = None
    explanation: str


class VolumeProfileBucket(ImmutableModel):
    id: UUID
    index: int = Field(ge=0)
    lower: float
    upper: float
    midpoint: float
    volume: float = Field(ge=0)
    estimated_buy_volume: float = Field(ge=0)
    estimated_sell_volume: float = Field(ge=0)
    source_count: int = Field(ge=0)
    upper_inclusive: bool = False

    @model_validator(mode="after")
    def bounds_and_volume(self) -> "VolumeProfileBucket":
        if self.upper <= self.lower or not self.lower <= self.midpoint <= self.upper:
            raise ValueError("bucket bounds are invalid")
        if abs(self.estimated_buy_volume + self.estimated_sell_volume - self.volume) > max(1e-8, self.volume * 1e-9):
            raise ValueError("directional allocation must conserve bucket volume")
        return self


class PointOfControl(ImmutableModel):
    id: UUID
    bucket_id: UUID
    price: float
    volume: float = Field(ge=0)
    volume_percent: float = Field(ge=0, le=1)
    confidence_score: float = Field(ge=0, le=100)
    tested: bool = False
    first_test_at: datetime | None = None
    test_count: int = Field(0, ge=0)


class ValueArea(ImmutableModel):
    id: UUID
    method: ValueAreaMethod
    target_percent: float = Field(gt=0, lt=1)
    achieved_percent: float = Field(ge=0, le=1)
    val: float
    vah: float
    included_bucket_ids: tuple[UUID, ...]
    included_volume: float = Field(ge=0)
    overshoot_percent: float = Field(ge=0)
    confidence_score: float = Field(ge=0, le=100)
    vah_tested: bool = False
    val_tested: bool = False
    first_test_at: datetime | None = None
    test_count: int = Field(0, ge=0)

    @model_validator(mode="after")
    def ordered(self) -> "ValueArea":
        if self.val > self.vah:
            raise ValueError("VAL cannot exceed VAH")
        return self


class VolumeNode(ImmutableModel):
    id: UUID
    node_type: NodeType
    lower: float
    upper: float
    peak_price: float
    total_volume: float = Field(ge=0)
    mean_volume: float = Field(ge=0)
    prominence: float = Field(ge=0)
    bucket_ids: tuple[UUID, ...]
    lifecycle_state: ProfileLifecycleState = ProfileLifecycleState.ACTIVE
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    tested: bool = False
    first_test_at: datetime | None = None
    test_count: int = Field(0, ge=0)


class HighVolumeNode(VolumeNode):
    node_type: NodeType = NodeType.HVN


class LowVolumeNode(VolumeNode):
    node_type: NodeType = NodeType.LVN


class VolumeShelf(ImmutableModel):
    id: UUID
    lower: float
    upper: float
    peak_price: float
    mean_bucket_volume: float = Field(ge=0)
    total_volume: float = Field(ge=0)
    width_bins: int = Field(ge=2)
    prominence: float = Field(ge=0)
    contains_poc: bool
    overlaps_value_area: bool
    lifecycle_state: ProfileLifecycleState = ProfileLifecycleState.ACTIVE
    confidence_score: float = Field(ge=0, le=100)


class VolumeGap(ImmutableModel):
    id: UUID
    lower: float
    upper: float
    width_bins: int = Field(ge=1)
    surrounding_strength: float = Field(ge=0)
    caused_by_missing_data: bool
    lifecycle_state: ProfileLifecycleState = ProfileLifecycleState.ACTIVE
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)


class ProfileShape(ImmutableModel):
    id: UUID
    shape_type: ProfileShapeType
    alternative: ProfileShapeType | None = None
    features: dict[str, float]
    conflicting_evidence: tuple[str, ...] = ()
    confidence_score: float = Field(ge=0, le=100)
    configuration_version: str


class ProfileAnchor(ImmutableModel):
    id: UUID
    anchor_type: AnchorType
    source_object_id: str | None = None
    anchor_timestamp: datetime
    availability_timestamp: datetime
    anchor_price: float | None = None
    source_engine: str
    confidence_score: float = Field(ge=0, le=100)
    validated: bool = True

    @model_validator(mode="after")
    def available_after_anchor(self) -> "ProfileAnchor":
        if self.availability_timestamp < self.anchor_timestamp:
            raise ValueError("anchor availability cannot precede anchor timestamp")
        return self


class ProfileLifecycleTransition(ImmutableModel):
    id: UUID
    profile_id: UUID
    previous: ProfileLifecycleState
    current: ProfileLifecycleState
    available_at: datetime
    reason: str

    @model_validator(mode="after")
    def transition(self) -> "ProfileLifecycleTransition":
        validate_transition(self.previous, self.current)
        return self


class ProfileMigration(ImmutableModel):
    id: UUID
    previous_profile_id: UUID
    current_profile_id: UUID
    migration_type: ProfileMigrationType
    poc_change: float
    vah_change: float
    val_change: float
    normalized_change: float
    bucket_change: int
    elapsed_seconds: float = Field(ge=0)
    available_at: datetime
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)


class ProfileConfluence(ImmutableModel):
    id: UUID
    price: float
    lower: float
    upper: float
    source_object_ids: tuple[str, ...]
    source_types: tuple[str, ...]
    timeframe_count: int = Field(ge=1)
    source_diversity: int = Field(ge=1)
    correlation_adjustment: float = Field(ge=0, le=1)
    lifecycle_state: ProfileLifecycleState = ProfileLifecycleState.ACTIVE
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)


class VolumeProfile(ImmutableModel):
    id: UUID
    logical_id: UUID
    symbol: str
    timeframe: Timeframe
    instrument: str
    profile_type: ProfileType
    session: SessionType | None = None
    status: ProfileStatus
    lifecycle_state: ProfileLifecycleState
    start_timestamp: datetime
    end_timestamp: datetime
    availability_timestamp: datetime
    completion_timestamp: datetime | None = None
    source_candle_count: int = Field(ge=0)
    source_first_timestamp: datetime
    source_last_timestamp: datetime
    volume_source_type: VolumeSourceType
    allocation_method: VolumeAllocationMethod
    price_grid_method: PriceGridMethod
    row_size: float = Field(gt=0)
    bucket_count: int = Field(ge=0)
    total_volume: float = Field(ge=0)
    included_volume: float = Field(ge=0)
    excluded_volume: float = Field(ge=0)
    buckets: tuple[VolumeProfileBucket, ...] = ()
    poc: PointOfControl | None = None
    value_area: ValueArea | None = None
    hvns: tuple[HighVolumeNode, ...] = ()
    lvns: tuple[LowVolumeNode, ...] = ()
    shelves: tuple[VolumeShelf, ...] = ()
    gaps: tuple[VolumeGap, ...] = ()
    shape: ProfileShape | None = None
    anchor: ProfileAnchor | None = None
    constituent_profile_ids: tuple[UUID, ...] = ()
    evidence: tuple[VolumeProfileEvidence, ...] = ()
    volume_data_quality: VolumeDataQuality
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    configuration_version: str
    engine_version: str
    analysis_boundary: datetime
    version: int = Field(ge=1)
    degradation_reason: str | None = None

    @field_validator(
        "start_timestamp",
        "end_timestamp",
        "availability_timestamp",
        "source_first_timestamp",
        "source_last_timestamp",
        "analysis_boundary",
        "completion_timestamp",
    )
    @classmethod
    def timezone(cls, value: datetime | None) -> datetime | None:
        return utc(value) if value else None

    @model_validator(mode="after")
    def consistency(self) -> "VolumeProfile":
        if self.start_timestamp > self.end_timestamp or self.source_first_timestamp > self.source_last_timestamp:
            raise ValueError("profile timestamp range is invalid")
        if self.availability_timestamp > self.analysis_boundary:
            raise ValueError("profile cannot be available after analysis boundary")
        if self.bucket_count != len(self.buckets):
            raise ValueError("bucket_count must match buckets")
        if abs(sum(x.volume for x in self.buckets) - self.included_volume) > max(1e-7, self.included_volume * 1e-8):
            raise ValueError("profile volume must be conserved")
        if self.total_volume + 1e-8 < self.included_volume + self.excluded_volume:
            raise ValueError("included and excluded volume exceed source volume")
        return self


class DevelopingVolumeProfile(VolumeProfile):
    profile_type: ProfileType = ProfileType.DEVELOPING


class CompletedVolumeProfile(VolumeProfile):
    status: ProfileStatus = ProfileStatus.COMPLETED


class CompositeVolumeProfile(VolumeProfile):
    profile_type: ProfileType = ProfileType.COMPOSITE


class AnchoredVolumeProfile(VolumeProfile):
    profile_type: ProfileType = ProfileType.ANCHORED


class SessionVolumeProfile(VolumeProfile):
    profile_type: ProfileType = ProfileType.SESSION
    session: SessionType = SessionType.CUSTOM


class FixedRangeVolumeProfile(VolumeProfile):
    profile_type: ProfileType = ProfileType.FIXED_RANGE


class VolumeProfileCheckpoint(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    snapshot_id: UUID
    last_processed_candle: datetime
    configuration_version: str
    engine_version: str
    state_hash: str = Field(min_length=64, max_length=64)


class MultiTimeframeVolumeProfileContext(ImmutableModel):
    symbol: str
    requested_timeframe: Timeframe
    profile_ids_by_timeframe: dict[str, tuple[UUID, ...]]
    overlapping_poc_ids: tuple[UUID, ...] = ()
    confluences: tuple[ProfileConfluence, ...] = ()
    analyzed_through: datetime
    maximum_depth: int = Field(ge=1, le=9)


class VolumeProfileAnalysisSnapshot(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    analysis_timestamp: datetime
    processing_mode: ProcessingMode
    status: AnalysisStatus
    profiles: tuple[VolumeProfile, ...] = ()
    developing: tuple[VolumeProfile, ...] = ()
    completed: tuple[VolumeProfile, ...] = ()
    migrations: tuple[ProfileMigration, ...] = ()
    confluences: tuple[ProfileConfluence, ...] = ()
    lifecycle_transitions: tuple[ProfileLifecycleTransition, ...] = ()
    multi_timeframe: MultiTimeframeVolumeProfileContext | None = None
    volume_data_quality: VolumeDataQuality
    confidence_summary: dict[str, float]
    quality_summary: dict[str, float]
    degraded_reasons: tuple[ProfileSkipReason, ...] = ()
    skipped_periods: tuple[SkippedProfilePeriod, ...] = ()
    configuration_version: str
    engine_version: str
    market_data_boundary: str
    created_at: datetime

    @field_validator("analysis_timestamp", "created_at")
    @classmethod
    def timezone(cls, value: datetime) -> datetime:
        return utc(value)


class PriceNode(BaseModel):
    price: float
    volume: float = Field(ge=0)
    kind: str


class VolumeProfileResult(BaseModel):
    poc: float | None = None
    vah: float | None = None
    val: float | None = None
    total_volume: float = Field(default=0, ge=0)
    nodes: list[PriceNode] = Field(default_factory=list)
    profile_type: str = "composite"
    observations: list[str] = Field(default_factory=list)
    snapshot: VolumeProfileAnalysisSnapshot | None = None
