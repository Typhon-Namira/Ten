"""Immutable, replay-safe Liquidity Engine Production 1.0 contracts."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.engines.market_data_engine import Timeframe

ENGINE_VERSION = "1.0.0"


class LiquiditySide(StrEnum):
    BUY_SIDE = "buy_side"
    SELL_SIDE = "sell_side"


class LiquidityLevelType(StrEnum):
    EQUAL_HIGH = "equal_high"
    EQUAL_LOW = "equal_low"
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    SESSION_HIGH = "session_high"
    SESSION_LOW = "session_low"
    PREVIOUS_DAY_HIGH = "previous_day_high"
    PREVIOUS_DAY_LOW = "previous_day_low"
    PREVIOUS_WEEK_HIGH = "previous_week_high"
    PREVIOUS_WEEK_LOW = "previous_week_low"
    PREVIOUS_MONTH_HIGH = "previous_month_high"
    PREVIOUS_MONTH_LOW = "previous_month_low"
    CURRENT_PERIOD_HIGH = "current_period_high"
    CURRENT_PERIOD_LOW = "current_period_low"
    ROUND_NUMBER = "round_number"
    PROTECTED_LEVEL = "protected_level"
    INDUCEMENT = "inducement"


class LiquidityPoolType(StrEnum):
    EQUAL_LEVEL = "equal_level"
    STRUCTURAL = "structural"
    SESSION = "session"
    REFERENCE = "reference"
    PSYCHOLOGICAL = "psychological"
    COMPOSITE = "composite"


class LiquidityEventType(StrEnum):
    APPROACH = "approach"
    TOUCH = "touch"
    SWEEP = "sweep"
    GRAB = "grab"
    RAID = "raid"
    STOP_HUNT = "stop_hunt"
    FALSE_BREAK = "false_break"
    RECLAIM = "reclaim"
    CONSUMPTION = "consumption"
    INVALIDATION = "invalidation"


class LiquidityLifecycleState(StrEnum):
    DETECTED = "detected"
    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    APPROACHED = "approached"
    TOUCHED = "touched"
    PARTIALLY_SWEPT = "partially_swept"
    SWEPT = "swept"
    RAIDED = "raided"
    CONSUMED = "consumed"
    RECLAIMED = "reclaimed"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    ARCHIVED = "archived"


class LiquidityStrength(StrEnum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


class LiquidityScope(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    SESSION = "session"
    PERIOD = "period"
    MULTI_TIMEFRAME = "multi_timeframe"


class LiquiditySource(StrEnum):
    SMC_SWING = "smc_swing"
    MICRO_CANDLE = "micro_candle"
    SESSION = "session"
    PERIOD = "period"
    ROUND_NUMBER = "round_number"
    SMC_PROTECTED = "smc_protected"
    EXTERNAL_CONTRACT = "external_contract"


class ConfirmationState(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ProcessingMode(StrEnum):
    HISTORICAL = "historical"
    REALTIME = "realtime"
    INCREMENTAL = "incremental"
    REPLAY = "replay"
    TIME_TRAVEL = "time_travel"
    RECOVERY = "recovery"


class AnalysisStatus(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT_HISTORY = "insufficient_history"
    DEGRADED = "degraded"
    FAILED = "failed"


class SessionType(StrEnum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NEW_YORK_OVERLAP = "london_new_york_overlap"
    CLOSED = "closed"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


class SweepClassification(StrEnum):
    WICK_ONLY = "wick_only"
    CLOSE_THROUGH = "close_through"
    SHALLOW = "shallow"
    DEEP = "deep"
    PARTIAL = "partial"
    FULL = "full"
    CONTINUATION = "continuation"
    AMBIGUOUS = "ambiguous"


class TargetStatus(StrEnum):
    ACTIVE = "active"
    REACHED = "reached"
    INVALIDATED = "invalidated"


def stable_id(kind: str, symbol: str, timeframe: Timeframe, *parts: object) -> UUID:
    return uuid5(NAMESPACE_URL, ":".join((kind, symbol.replace("/", "").upper(), timeframe.value, *(str(part) for part in parts))))


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class LiquidityEvidence(ImmutableModel):
    code: str
    source_ids: tuple[str, ...] = ()
    observed_value: float | str | bool | None = None
    threshold: float | str | bool | None = None
    passed: bool = True
    weight: float = Field(default=1, ge=0, le=1)


class LiquidityLevel(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    level_type: LiquidityLevelType
    scope: LiquidityScope
    side: LiquiditySide
    price: float = Field(gt=0)
    lower_bound: float = Field(gt=0)
    upper_bound: float = Field(gt=0)
    source_timestamps: tuple[datetime, ...]
    created_at: datetime
    available_at: datetime
    confirmation_at: datetime | None = None
    invalidation_at: datetime | None = None
    lifecycle_state: LiquidityLifecycleState = LiquidityLifecycleState.ACTIVE
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    strength_score: float = Field(ge=0, le=100)
    freshness_score: float = Field(ge=0, le=100)
    touch_count: int = Field(default=1, ge=1)
    source: LiquiditySource
    evidence: tuple[LiquidityEvidence, ...] = ()
    source_object_ids: tuple[str, ...] = ()
    configuration_version: str
    engine_version: str = ENGINE_VERSION
    analysis_boundary: datetime
    replay_metadata: dict[str, Any] = Field(default_factory=dict)
    invalidation_reason: str | None = None
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def valid_level(self) -> "LiquidityLevel":
        if not self.lower_bound <= self.price <= self.upper_bound:
            raise ValueError("price must be within level bounds")
        if self.available_at > self.analysis_boundary:
            raise ValueError("level cannot be available after analysis boundary")
        if self.source_timestamps and max(self.source_timestamps) > self.available_at:
            raise ValueError("availability cannot precede source observations")
        return self

    @property
    def touches(self) -> int:
        return self.touch_count

    @property
    def swept(self) -> bool:
        return self.lifecycle_state in {
            LiquidityLifecycleState.SWEPT,
            LiquidityLifecycleState.RAIDED,
            LiquidityLifecycleState.CONSUMED,
            LiquidityLifecycleState.RECLAIMED,
        }

    @property
    def last_seen(self) -> datetime:
        return max(self.source_timestamps) if self.source_timestamps else self.available_at


class EqualLevelCluster(LiquidityLevel):
    member_prices: tuple[float, ...]
    tolerance_used: float = Field(ge=0)
    temporal_separation_seconds: float = Field(ge=0)
    outliers_rejected: int = Field(default=0, ge=0)
    parent_cluster_id: UUID | None = None


class LiquidityPool(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    pool_type: LiquidityPoolType
    scope: LiquidityScope
    side: LiquiditySide
    lower_bound: float = Field(gt=0)
    upper_bound: float = Field(gt=0)
    constituent_level_ids: tuple[UUID, ...]
    touch_count: int = Field(ge=1)
    created_at: datetime
    available_at: datetime
    lifecycle_state: LiquidityLifecycleState
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    strength_score: float = Field(ge=0, le=100)
    freshness_score: float = Field(ge=0, le=100)
    distance_from_price: float
    sweep_percentage: float = Field(default=0, ge=0, le=100)
    evidence: tuple[LiquidityEvidence, ...] = ()
    target_rank: int | None = Field(default=None, ge=1)
    configuration_version: str
    engine_version: str = ENGINE_VERSION
    analysis_boundary: datetime
    invalidation_at: datetime | None = None
    invalidation_reason: str | None = None
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def valid_pool(self) -> "LiquidityPool":
        if self.upper_bound < self.lower_bound:
            raise ValueError("pool upper bound cannot be below lower bound")
        if self.available_at > self.analysis_boundary:
            raise ValueError("pool cannot leak into an earlier boundary")
        return self


class LiquidityEvent(ImmutableModel):
    id: UUID
    event_type: LiquidityEventType
    symbol: str
    timeframe: Timeframe
    pool_id: UUID
    side: LiquiditySide
    occurred_at: datetime
    available_at: datetime
    price: float = Field(gt=0)
    lifecycle_from: LiquidityLifecycleState
    lifecycle_to: LiquidityLifecycleState
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    evidence: tuple[LiquidityEvidence, ...] = ()
    configuration_version: str
    engine_version: str = ENGINE_VERSION
    version: int = Field(default=1, ge=1)


class LiquiditySweep(LiquidityEvent):
    classification: SweepClassification
    penetration_distance: float = Field(ge=0)
    penetration_percentage: float = Field(ge=0)
    wick_penetration: float = Field(ge=0)
    close_penetration: float = Field(ge=0)
    time_outside_seconds: float = Field(ge=0)
    reclaim_timestamp: datetime | None = None
    reclaim_strength: float = Field(default=0, ge=0, le=100)
    displacement_after_reclaim: float = Field(default=0, ge=0)
    structure_shift_after_reclaim: bool = False
    maximum_favorable_response: float = Field(default=0, ge=0)
    maximum_adverse_continuation: float = Field(default=0, ge=0)
    time_to_response_seconds: float | None = Field(default=None, ge=0)


class LiquidityGrab(LiquiditySweep):
    rejection_candles: int = Field(ge=1)


class LiquidityRaid(LiquiditySweep):
    consumed_pool_ids: tuple[UUID, ...] = ()


class StopHunt(LiquiditySweep):
    price_action_classification_only: bool = True


class FalseBreak(LiquiditySweep):
    held_outside_candles: int = Field(ge=1)


class SessionLiquidityRange(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    session: SessionType
    opened_at: datetime
    available_at: datetime
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    midpoint: float = Field(gt=0)
    completed: bool
    source_candle_ids: tuple[str, ...]
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    configuration_version: str
    engine_version: str = ENGINE_VERSION


class ReferenceLiquidityLevel(LiquidityLevel):
    source_period_start: datetime
    source_period_end: datetime


class LiquidityConfluence(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    lower_bound: float = Field(gt=0)
    upper_bound: float = Field(gt=0)
    contributing_source_ids: tuple[str, ...]
    source_diversity: int = Field(ge=1)
    timeframe_diversity: int = Field(ge=1)
    agreement_score: float = Field(ge=0, le=100)
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    lifecycle_state: LiquidityLifecycleState
    available_at: datetime
    configuration_version: str
    engine_version: str = ENGINE_VERSION


class LiquidityTarget(ImmutableModel):
    id: UUID
    pool_id: UUID
    symbol: str
    timeframe: Timeframe
    side: LiquiditySide
    rank: int = Field(ge=1)
    relative_distance: float = Field(ge=0)
    strength_score: float = Field(ge=0, le=100)
    accessibility_score: float = Field(ge=0, le=100)
    path_obstruction_score: float = Field(ge=0, le=100)
    intermediate_pool_count: int = Field(ge=0)
    confidence_score: float = Field(ge=0, le=100)
    status: TargetStatus
    evidence: tuple[LiquidityEvidence, ...] = ()
    invalidation_condition: str
    available_at: datetime
    configuration_version: str
    engine_version: str = ENGINE_VERSION


class LiquidityMapBand(ImmutableModel):
    lower_bound: float = Field(gt=0)
    upper_bound: float = Field(gt=0)
    side: LiquiditySide
    inferred_density: float = Field(ge=0, le=100)
    source_count: int = Field(ge=1)
    weighted_strength: float = Field(ge=0, le=100)
    timeframe_composition: tuple[str, ...]
    active: bool
    distance_from_price: float
    confidence_score: float = Field(ge=0, le=100)
    age_seconds: float = Field(ge=0)


class MultiTimeframeLiquidityContext(ImmutableModel):
    symbol: str
    requested_timeframe: Timeframe
    pools_by_timeframe: dict[str, tuple[UUID, ...]] = Field(default_factory=dict)
    nested_pool_ids: tuple[UUID, ...] = ()
    confluence_score: float = Field(ge=0, le=100)
    conflict_count: int = Field(ge=0)
    analyzed_through: datetime
    maximum_depth: int = Field(ge=1)


class LiquidityState(ImmutableModel):
    symbol: str
    timeframe: Timeframe
    latest_price: float = Field(gt=0)
    active_pool_ids: tuple[UUID, ...] = ()
    consumed_pool_ids: tuple[UUID, ...] = ()
    last_event_id: UUID | None = None
    updated_at: datetime
    version: int = Field(default=1, ge=1)


class LiquidityCheckpoint(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    snapshot_id: UUID
    last_processed_candle: datetime
    configuration_version: str
    engine_version: str
    state_hash: str
    recovered_at: datetime | None = None


class LiquidityAnalysisSnapshot(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    analysis_timestamp: datetime
    market_data_boundary: str
    processing_mode: ProcessingMode
    status: AnalysisStatus
    state: LiquidityState
    levels: tuple[LiquidityLevel, ...] = ()
    equal_levels: tuple[EqualLevelCluster, ...] = ()
    pools: tuple[LiquidityPool, ...] = ()
    events: tuple[LiquidityEvent, ...] = ()
    sweeps: tuple[LiquiditySweep, ...] = ()
    grabs: tuple[LiquidityGrab, ...] = ()
    raids: tuple[LiquidityRaid, ...] = ()
    stop_hunts: tuple[StopHunt, ...] = ()
    false_breaks: tuple[FalseBreak, ...] = ()
    sessions: tuple[SessionLiquidityRange, ...] = ()
    reference_levels: tuple[ReferenceLiquidityLevel, ...] = ()
    inducements: tuple[LiquidityLevel, ...] = ()
    confluences: tuple[LiquidityConfluence, ...] = ()
    targets: tuple[LiquidityTarget, ...] = ()
    map_bands: tuple[LiquidityMapBand, ...] = ()
    multi_timeframe: MultiTimeframeLiquidityContext | None = None
    confidence_summary: dict[str, float] = Field(default_factory=dict)
    quality_summary: dict[str, float] = Field(default_factory=dict)
    degraded_reasons: tuple[str, ...] = ()
    configuration_version: str
    engine_version: str = ENGINE_VERSION
    created_at: datetime


class LiquidityResult(BaseModel):
    """Backward-compatible pipeline wrapper."""

    levels: list[LiquidityLevel] = Field(default_factory=list)
    nearest_buy_side: float | None = None
    nearest_sell_side: float | None = None
    active_session: str = "unknown"
    observations: list[str] = Field(default_factory=list)
    snapshot: LiquidityAnalysisSnapshot | None = None


_TRANSITIONS: dict[LiquidityLifecycleState, set[LiquidityLifecycleState]] = {
    LiquidityLifecycleState.DETECTED: {LiquidityLifecycleState.PENDING_CONFIRMATION, LiquidityLifecycleState.CONFIRMED, LiquidityLifecycleState.INVALIDATED},
    LiquidityLifecycleState.PENDING_CONFIRMATION: {LiquidityLifecycleState.CONFIRMED, LiquidityLifecycleState.INVALIDATED, LiquidityLifecycleState.EXPIRED},
    LiquidityLifecycleState.CONFIRMED: {LiquidityLifecycleState.ACTIVE, LiquidityLifecycleState.INVALIDATED, LiquidityLifecycleState.EXPIRED},
    LiquidityLifecycleState.ACTIVE: {
        LiquidityLifecycleState.APPROACHED,
        LiquidityLifecycleState.TOUCHED,
        LiquidityLifecycleState.PARTIALLY_SWEPT,
        LiquidityLifecycleState.SWEPT,
        LiquidityLifecycleState.INVALIDATED,
        LiquidityLifecycleState.EXPIRED,
    },
    LiquidityLifecycleState.APPROACHED: {
        LiquidityLifecycleState.TOUCHED,
        LiquidityLifecycleState.ACTIVE,
        LiquidityLifecycleState.SWEPT,
        LiquidityLifecycleState.EXPIRED,
    },
    LiquidityLifecycleState.TOUCHED: {
        LiquidityLifecycleState.ACTIVE,
        LiquidityLifecycleState.PARTIALLY_SWEPT,
        LiquidityLifecycleState.SWEPT,
        LiquidityLifecycleState.INVALIDATED,
    },
    LiquidityLifecycleState.PARTIALLY_SWEPT: {
        LiquidityLifecycleState.SWEPT,
        LiquidityLifecycleState.RAIDED,
        LiquidityLifecycleState.RECLAIMED,
        LiquidityLifecycleState.CONSUMED,
    },
    LiquidityLifecycleState.SWEPT: {
        LiquidityLifecycleState.RAIDED,
        LiquidityLifecycleState.RECLAIMED,
        LiquidityLifecycleState.CONSUMED,
        LiquidityLifecycleState.ARCHIVED,
    },
    LiquidityLifecycleState.RAIDED: {LiquidityLifecycleState.RECLAIMED, LiquidityLifecycleState.CONSUMED, LiquidityLifecycleState.ARCHIVED},
    LiquidityLifecycleState.RECLAIMED: {LiquidityLifecycleState.CONSUMED, LiquidityLifecycleState.ARCHIVED},
    LiquidityLifecycleState.CONSUMED: {LiquidityLifecycleState.ARCHIVED},
    LiquidityLifecycleState.INVALIDATED: {LiquidityLifecycleState.ARCHIVED},
    LiquidityLifecycleState.EXPIRED: {LiquidityLifecycleState.ARCHIVED},
    LiquidityLifecycleState.ARCHIVED: set(),
}


def validate_transition(previous: LiquidityLifecycleState, current: LiquidityLifecycleState) -> None:
    if previous != current and current not in _TRANSITIONS[previous]:
        raise ValueError(f"impossible liquidity lifecycle transition: {previous} -> {current}")


def utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("liquidity timestamps must be timezone-aware")
    return value.astimezone(UTC)
