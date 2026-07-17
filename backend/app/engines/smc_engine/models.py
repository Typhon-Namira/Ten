"""Immutable, replay-safe Smart Money Concepts domain contracts."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid5, NAMESPACE_URL

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from backend.app.engines.market_data_engine import Timeframe

ENGINE_VERSION = "2.0.0"


class StructureDirection(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    TRANSITIONAL = "transitional"


Bias = StructureDirection


class StructureScope(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    SWING = "swing"
    NESTED = "nested"


class StructureEventType(StrEnum):
    BOS = "bos"
    CHOCH = "choch"
    MSS = "mss"
    CONTINUATION = "structure_continuation"
    INVALIDATED = "structure_invalidated"


class SwingType(StrEnum):
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    INTERNAL_HIGH = "internal_high"
    INTERNAL_LOW = "internal_low"
    EXTERNAL_HIGH = "external_high"
    EXTERNAL_LOW = "external_low"


class ConfirmationState(StrEnum):
    UNCONFIRMED = "unconfirmed"
    PROVISIONAL = "provisionally_confirmed"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"


class LifecycleState(StrEnum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    PARTIALLY_MITIGATED = "partially_mitigated"
    MITIGATED = "mitigated"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class AnalysisStatus(StrEnum):
    READY = "ready"
    INSUFFICIENT_HISTORY = "insufficient_history"
    DEGRADED_INPUT = "degraded_input"
    INVALID_INPUT = "invalid_input"
    PARTIAL = "partial_analysis"
    COMPLETE = "complete"


class ProcessingMode(StrEnum):
    REALTIME = "realtime"
    HISTORICAL = "historical"
    REPLAY = "replay"
    REBUILD = "rebuild"
    VALIDATION = "validation"


class ConfirmationMethod(StrEnum):
    CLOSE = "close"
    WICK = "wick"
    HYBRID = "hybrid"


def stable_id(kind: str, symbol: str, timeframe: Timeframe, *parts: object) -> UUID:
    value = ":".join((kind, symbol, timeframe.value, *(str(part) for part in parts)))
    return uuid5(NAMESPACE_URL, value)


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Evidence(ImmutableModel):
    code: str
    description: str
    value: float | str | bool | None = None
    threshold: float | str | bool | None = None
    passed: bool = True


class SwingPoint(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    timestamp: datetime
    candle_index: int = Field(ge=0)
    price: float = Field(gt=0)
    swing_type: SwingType
    scope: StructureScope
    confirmation_state: ConfirmationState
    left_window: int = Field(ge=1)
    right_window: int = Field(ge=1)
    strength: float = Field(ge=0, le=100)
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    source_candle_ids: tuple[str, ...]
    detected_at: datetime
    confirmed_at: datetime | None = None
    invalidated_at: datetime | None = None
    algorithm_version: str = ENGINE_VERSION


class StructureLeg(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    start_swing_id: UUID
    end_swing_id: UUID
    direction: StructureDirection
    scope: StructureScope
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    magnitude: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    displacement_score: float = Field(ge=0, le=100)
    confirmation_state: ConfirmationState
    confidence_score: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def valid_range(self) -> "StructureLeg":
        if self.high < self.low:
            raise ValueError("structure leg high cannot be below low")
        return self


class StructureEvent(ImmutableModel):
    id: UUID
    event_type: StructureEventType
    symbol: str
    timeframe: Timeframe
    scope: StructureScope
    direction: StructureDirection
    timestamp: datetime
    broken_level: float = Field(gt=0)
    broken_swing_id: UUID
    confirmation_candle_id: str
    confirmation_method: ConfirmationMethod
    close_confirmed: bool
    wick_confirmed: bool
    previous_direction: StructureDirection
    resulting_direction: StructureDirection
    break_distance: float = Field(ge=0)
    displacement_score: float = Field(ge=0, le=100)
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    evidence: tuple[Evidence, ...]
    invalidation_metadata: dict[str, Any] = Field(default_factory=dict)
    algorithm_version: str = ENGINE_VERSION
    created_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def kind(self) -> str:
        return self.event_type.value.upper()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def price(self) -> float:
        return self.broken_level

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        return self.confidence_score / 100


class MarketStructureState(ImmutableModel):
    symbol: str
    timeframe: Timeframe
    current_direction: StructureDirection = StructureDirection.NEUTRAL
    previous_direction: StructureDirection = StructureDirection.NEUTRAL
    internal_direction: StructureDirection = StructureDirection.NEUTRAL
    external_direction: StructureDirection = StructureDirection.NEUTRAL
    active_swing_high_id: UUID | None = None
    active_swing_low_id: UUID | None = None
    protected_high_id: UUID | None = None
    protected_low_id: UUID | None = None
    last_bos_id: UUID | None = None
    last_choch_id: UUID | None = None
    last_mss_id: UUID | None = None
    active_dealing_range_id: UUID | None = None
    state_version: int = Field(default=0, ge=0)
    last_processed_candle: datetime | None = None
    updated_at: datetime


class SMCAnalysisSnapshot(ImmutableModel):
    id: UUID
    symbol: str
    timeframe: Timeframe
    analysis_timestamp: datetime
    market_data_boundary: str
    status: AnalysisStatus
    processing_mode: ProcessingMode
    structure_state: MarketStructureState
    swings: tuple[SwingPoint, ...] = ()
    structure_legs: tuple[StructureLeg, ...] = ()
    structure_events: tuple[StructureEvent, ...] = ()
    confidence_summary: dict[str, float] = Field(default_factory=dict)
    quality_summary: dict[str, float] = Field(default_factory=dict)
    reasoning_metadata: dict[str, Any] = Field(default_factory=dict)
    engine_version: str = ENGINE_VERSION
    configuration_version: str
    created_at: datetime


class SMCResult(BaseModel):
    """Backward-compatible pipeline result wrapping the institutional snapshot."""

    bias: StructureDirection = StructureDirection.NEUTRAL
    structure_events: list[StructureEvent] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    fair_value_gaps: list[object] = Field(default_factory=list)
    premium_discount_position: str = "equilibrium"
    snapshot: SMCAnalysisSnapshot | None = None


def utc_from(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        raise ValueError("SMC timestamps must be timezone-aware")
    return timestamp.astimezone(UTC)
