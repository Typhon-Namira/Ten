"""Immutable contracts for AI-centric, point-in-time market evidence.

These models are additive.  The production scoring and decision engines do not consume them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.engines.market_data_engine.models import MarketScheduleStatus


REQUIRED_TIMEFRAMES = ("M1", "M5", "M15")


class ImmutableMarketStateModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvidenceAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    STALE = "stale"


class EvidenceClassification(StrEnum):
    MARKET_EVIDENCE = "market_evidence"
    CONFIDENCE_MODIFIER = "confidence_modifier"
    RISK_EVIDENCE = "risk_evidence"
    EXECUTION_EVIDENCE = "execution_evidence"
    INFORMATIONAL_CONTEXT = "informational_context"
    TECHNICAL_HARD_GATE = "mandatory_technical_hard_gate"
    RISK_HARD_GATE = "mandatory_risk_hard_gate"


class MarketStateStatus(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"


class CapturedEngineEvidence(ImmutableMarketStateModel):
    """One complete engine result captured before legacy score normalization."""

    source_engine: str
    source_engine_version: str
    evidence_type: str = "engine_snapshot"
    classification: EvidenceClassification = EvidenceClassification.MARKET_EVIDENCE
    availability: EvidenceAvailability
    raw_value: Any | None = None
    normalized_value: Any | None = None
    confidence: float | None = Field(default=None, ge=0, le=100)
    quality: float | None = Field(default=None, ge=0, le=100)
    uncertainty: float | None = Field(default=None, ge=0, le=100)
    observed_at: datetime
    available_at: datetime
    provenance: dict[str, Any] = Field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    @field_validator("observed_at", "available_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("evidence timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def unavailable_has_no_synthetic_value(self) -> CapturedEngineEvidence:
        if self.availability == EvidenceAvailability.UNAVAILABLE:
            if self.raw_value is not None or self.normalized_value is not None:
                raise ValueError("unavailable evidence cannot carry a synthetic value")
            if any(value is not None for value in (self.confidence, self.quality, self.uncertainty)):
                raise ValueError("unavailable evidence cannot be represented by numeric zero")
        return self


class MarketEvidenceFrame(ImmutableMarketStateModel):
    """Complete evidence produced for one closed candle on one timeframe."""

    frame_id: UUID
    frame_hash: str = Field(min_length=64, max_length=64)
    cycle_id: UUID
    correlation_id: UUID
    instrument: str
    timeframe: str
    candle_open_at: datetime
    candle_close_at: datetime
    knowledge_cutoff: datetime
    mode: str
    market_event_id: str
    evidence: tuple[CapturedEngineEvidence, ...]
    created_at: datetime

    @field_validator("candle_open_at", "candle_close_at", "knowledge_cutoff", "created_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("frame timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def point_in_time_invariants(self) -> MarketEvidenceFrame:
        if self.timeframe not in REQUIRED_TIMEFRAMES:
            raise ValueError("Unified Market State Phase 1 supports M1, M5, and M15 frames")
        if self.candle_close_at <= self.candle_open_at:
            raise ValueError("frame candle close must follow open")
        if self.candle_close_at > self.knowledge_cutoff or self.created_at > self.knowledge_cutoff:
            raise ValueError("frame contains information beyond its knowledge cutoff")
        if any(item.observed_at > self.knowledge_cutoff or item.available_at > self.knowledge_cutoff for item in self.evidence):
            raise ValueError("frame evidence contains future data")
        engines = [item.source_engine for item in self.evidence]
        if len(engines) != len(set(engines)):
            raise ValueError("frame evidence engines must be unique")
        return self


class TimeframeState(ImmutableMarketStateModel):
    timeframe: str
    frame_id: UUID
    source_candle_open_at: datetime
    source_candle_close_at: datetime
    expected_candle_close_at: datetime
    freshness_seconds: float = Field(ge=0)
    stale: bool
    evidence_ids: tuple[UUID, ...]

    @field_validator("source_candle_open_at", "source_candle_close_at", "expected_candle_close_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timeframe-state timestamps must be timezone-aware")
        return value.astimezone(UTC)


class EvidenceItem(ImmutableMarketStateModel):
    """State-specific evidence with explicit availability and full raw provenance."""

    evidence_id: UUID
    market_state_id: UUID
    source_frame_id: UUID
    source_engine: str
    source_engine_version: str
    source_timeframe: str
    source_candle_timestamp: datetime
    source_candle_close_timestamp: datetime
    evidence_type: str
    classification: EvidenceClassification
    availability: EvidenceAvailability
    normalized_value: Any | None = None
    raw_value: Any | None = None
    confidence: float | None = Field(default=None, ge=0, le=100)
    quality: float | None = Field(default=None, ge=0, le=100)
    uncertainty: float | None = Field(default=None, ge=0, le=100)
    observed_at: datetime
    available_at: datetime
    freshness_seconds: float = Field(ge=0)
    provenance: dict[str, Any] = Field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    @field_validator(
        "source_candle_timestamp",
        "source_candle_close_timestamp",
        "observed_at",
        "available_at",
    )
    @classmethod
    def timestamps_are_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("evidence-item timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def unavailable_has_no_synthetic_value(self) -> EvidenceItem:
        if self.availability == EvidenceAvailability.UNAVAILABLE:
            if self.raw_value is not None or self.normalized_value is not None:
                raise ValueError("unavailable evidence cannot carry a synthetic value")
            if any(value is not None for value in (self.confidence, self.quality, self.uncertainty)):
                raise ValueError("unavailable evidence cannot be represented by numeric zero")
        return self


class UnifiedMarketState(ImmutableMarketStateModel):
    schema_version: str = "1.0"
    state_id: UUID
    state_hash: str = Field(min_length=64, max_length=64)
    cycle_id: UUID
    correlation_id: UUID
    instrument: str
    trigger_timeframe: str
    market_data_boundary: datetime
    knowledge_cutoff: datetime
    mode: str
    status: MarketStateStatus
    market_schedule: MarketScheduleStatus | None = None
    timeframes: tuple[TimeframeState, ...]
    evidence: tuple[EvidenceItem, ...]
    unavailable_evidence: tuple[UUID, ...] = ()
    degraded_evidence: tuple[UUID, ...] = ()
    stale_evidence: tuple[UUID, ...] = ()
    evidence_completeness: float = Field(ge=0, le=1)
    created_at: datetime

    @field_validator("market_data_boundary", "knowledge_cutoff", "created_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("market-state timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def synchronized_point_in_time_state(self) -> UnifiedMarketState:
        names = tuple(item.timeframe for item in self.timeframes)
        if set(names) != set(REQUIRED_TIMEFRAMES) or len(names) != len(REQUIRED_TIMEFRAMES):
            raise ValueError("Unified Market State requires exactly one M1, M5, and M15 frame")
        if self.market_data_boundary > self.knowledge_cutoff or self.created_at > self.knowledge_cutoff:
            raise ValueError("market state exceeds its knowledge cutoff")
        frames = {timeframe_state.frame_id: timeframe_state for timeframe_state in self.timeframes}
        for timeframe_state in self.timeframes:
            if timeframe_state.source_candle_close_at > timeframe_state.expected_candle_close_at:
                raise ValueError("timeframe state contains a future candle")
            if timeframe_state.expected_candle_close_at > self.market_data_boundary:
                raise ValueError("timeframe expectation exceeds the state boundary")
        for evidence_item in self.evidence:
            if evidence_item.market_state_id != self.state_id or evidence_item.source_frame_id not in frames:
                raise ValueError("evidence relationship does not belong to this market state")
            if evidence_item.source_candle_close_timestamp > self.market_data_boundary:
                raise ValueError("evidence contains a future candle")
            if evidence_item.observed_at > self.knowledge_cutoff or evidence_item.available_at > self.knowledge_cutoff:
                raise ValueError("evidence was unavailable at the knowledge cutoff")
        expected_unavailable = {item.evidence_id for item in self.evidence if item.availability == EvidenceAvailability.UNAVAILABLE}
        expected_degraded = {item.evidence_id for item in self.evidence if item.availability == EvidenceAvailability.DEGRADED}
        expected_stale = {item.evidence_id for item in self.evidence if item.availability == EvidenceAvailability.STALE}
        if set(self.unavailable_evidence) != expected_unavailable:
            raise ValueError("unavailable evidence index is inconsistent")
        if set(self.degraded_evidence) != expected_degraded:
            raise ValueError("degraded evidence index is inconsistent")
        if set(self.stale_evidence) != expected_stale:
            raise ValueError("stale evidence index is inconsistent")
        return self
