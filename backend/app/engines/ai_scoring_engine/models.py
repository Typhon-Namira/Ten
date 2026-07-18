from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NAMESPACE = UUID("c2ff62ba-3495-5ab2-b0db-3a63a66d72d4")


class ScoringModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ScoreMode(StrEnum):
    LIVE = "live"
    REPLAY = "replay"


class ScoreStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    STALE = "stale"
    INVALID = "invalid"
    REPLAY = "replay"


class DirectionalLabel(StrEnum):
    STRONG_BEARISH = "strong_bearish"
    BEARISH = "bearish"
    SLIGHTLY_BEARISH = "slightly_bearish"
    NEUTRAL = "neutral"
    SLIGHTLY_BULLISH = "slightly_bullish"
    BULLISH = "bullish"
    STRONG_BULLISH = "strong_bullish"


class SourceState(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    INVALID = "invalid"


class FreshnessState(StrEnum):
    FRESH = "fresh"
    AGING = "aging"
    STALE = "stale"
    EXPIRED = "expired"


class SourceEvidence(ScoringModel):
    source: str
    source_group: str
    source_version: str
    evidence_id: str
    source_timestamp: datetime
    observation_timestamp: datetime
    publication_timestamp: datetime
    direction: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    degraded: bool = False
    reason_codes: tuple[str, ...] = ()

    @field_validator("source_timestamp", "observation_timestamp", "publication_timestamp")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)


class SourceHealth(ScoringModel):
    source: str
    state: SourceState
    checked_at: datetime
    reason_codes: tuple[str, ...] = ()

    @field_validator("checked_at")
    @classmethod
    def checked_at_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("checked_at must be timezone-aware")
        return value.astimezone(UTC)


class ScoringInput(ScoringModel):
    instrument: str = Field(min_length=2, max_length=32, pattern=r"^[A-Z0-9._-]+$")
    timeframe: str = Field(min_length=1, max_length=16, pattern=r"^[A-Za-z0-9]+$")
    as_of: datetime
    requested_at: datetime
    mode: ScoreMode = ScoreMode.LIVE
    market_data: SourceEvidence | None = None
    market_regime: SourceEvidence | None = None
    smc: SourceEvidence | None = None
    liquidity: SourceEvidence | None = None
    volume_profile: SourceEvidence | None = None
    institutional_flow: SourceEvidence | None = None
    economic_calendar: SourceEvidence | None = None
    source_health: tuple[SourceHealth, ...] = ()

    @field_validator("as_of", "requested_at")
    @classmethod
    def input_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def no_future_evidence(self) -> ScoringInput:
        for item in self.sources():
            if item.publication_timestamp > self.as_of or item.observation_timestamp > self.as_of:
                raise ValueError(f"future evidence is prohibited: {item.source}")
        return self

    def sources(self) -> tuple[SourceEvidence, ...]:
        values = (self.market_data, self.market_regime, self.smc, self.liquidity, self.volume_profile, self.institutional_flow, self.economic_calendar)
        return tuple(item for item in values if item is not None)

    def fingerprint(self, policy_name: str, policy_version: str, configuration_hash: str) -> str:
        payload = self.model_dump(mode="json", exclude={"requested_at", "source_health"})
        payload.update(policy_name=policy_name, policy_version=policy_version, configuration_hash=configuration_hash)
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ScoreComponent(ScoringModel):
    component_name: str
    source_engine: str
    source_group: str
    evidence_id: str
    normalized_direction: float = Field(ge=-1, le=1)
    directional_contribution: float = Field(ge=-100, le=100)
    confidence_contribution: float = Field(ge=0, le=100)
    risk_contribution: float = Field(ge=0, le=100)
    quality_contribution: float = Field(ge=0, le=100)
    configured_weight: float = Field(ge=0, le=1)
    effective_weight: float = Field(ge=0, le=1)
    freshness_factor: float = Field(ge=0, le=1)
    availability_factor: float = Field(ge=0, le=1)
    quality_factor: float = Field(ge=0, le=1)
    freshness_state: FreshnessState
    reason_codes: tuple[str, ...]
    source_timestamp: datetime
    observation_timestamp: datetime


class EvidenceConflict(ScoringModel):
    conflict_id: UUID
    conflict_type: str
    severity: str
    sources: tuple[str, ...]
    directional_gap: float = Field(ge=0, le=2)
    description_code: str
    confidence_penalty: float = Field(ge=0, le=100)


class ExplanationContributor(ScoringModel):
    source: str
    reason_code: str
    contribution: float


class ScoreExplanation(ScoringModel):
    summary_code: str
    positive_contributors: tuple[ExplanationContributor, ...] = ()
    negative_contributors: tuple[ExplanationContributor, ...] = ()
    risk_contributors: tuple[ExplanationContributor, ...] = ()
    limitations: tuple[str, ...] = ()
    financial_safety_code: str = "analytical_intelligence_only"


class ScoreMetadata(ScoringModel):
    engine_version: str
    schema_version: str
    configuration_version: str
    configuration_hash: str
    input_fingerprint: str
    independent_group_count: int = Field(ge=0)
    available_source_count: int = Field(ge=0)
    probabilistic_context: bool = True
    deterministic_replay: bool = True
    point_in_time_safe: bool = True
    explainable_output: bool = True
    trading_instruction: bool = False
    order_execution: bool = False


class AIScoreSnapshot(ScoringModel):
    snapshot_id: UUID
    instrument: str
    timeframe: str
    as_of: datetime
    calculated_at: datetime
    mode: ScoreMode
    policy_name: str
    policy_version: str
    directional_score: float = Field(ge=-100, le=100)
    directional_label: DirectionalLabel
    confidence_score: float = Field(ge=0, le=100)
    market_risk_score: float = Field(ge=0, le=100)
    evidence_alignment_score: float = Field(ge=0, le=100)
    data_quality_score: float = Field(ge=0, le=100)
    composite_score: float = Field(ge=-100, le=100)
    components: tuple[ScoreComponent, ...]
    conflicts: tuple[EvidenceConflict, ...]
    missing_sources: tuple[str, ...]
    degraded_sources: tuple[str, ...]
    explanation: ScoreExplanation
    status: ScoreStatus
    metadata: ScoreMetadata


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instrument: str = Field(default="XAUUSD", min_length=2, max_length=32, pattern=r"^[A-Za-z0-9._-]+$")
    timeframe: str = Field(default="M15", pattern=r"^(M1|M5|M15|M30|H1|H4|D1|W1|MN1)$")
    as_of: datetime | None = None
    persist: bool = True
    publish_events: bool = True
    mode: ScoreMode = ScoreMode.LIVE

    @field_validator("instrument")
    @classmethod
    def canonical_instrument(cls, value: str) -> str:
        return value.upper()


def stable_id(*parts: object) -> UUID:
    return uuid5(NAMESPACE, "|".join(str(part) for part in parts))


class ScoredDirection(StrEnum):
    """Legacy signal-pipeline vocabulary; not emitted by the production scoring API."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class ScoringContext(BaseModel):
    """Deprecated pipeline compatibility model."""

    features: dict[str, dict[str, Any]] = Field(default_factory=dict)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    market_structure: dict[str, Any] = Field(default_factory=dict, deprecated=True)
    liquidity: dict[str, Any] = Field(default_factory=dict, deprecated=True)
    flow_score: dict[str, Any] = Field(default_factory=dict, deprecated=True)
    volume_profile: dict[str, Any] = Field(default_factory=dict, deprecated=True)
    news_risk: dict[str, Any] = Field(default_factory=dict, deprecated=True)

    @classmethod
    def from_features(cls, features: dict[str, dict[str, Any]], engine_versions: dict[str, str]) -> ScoringContext:
        return cls(features=features, engine_versions=engine_versions)


class SignalScore(BaseModel):
    """Deprecated signal-engine adapter; production APIs return AIScoreSnapshot."""

    confidence: float | None = Field(default=None, ge=0, le=1)
    direction: ScoredDirection
    quality_score: float = Field(ge=0, le=100)
    risk_notes: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    model: str
    prompt_version: str
