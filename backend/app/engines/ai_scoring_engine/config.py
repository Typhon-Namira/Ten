from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ComponentConfig(FrozenConfig):
    directional_weight: float = Field(ge=0, le=1)
    confidence_weight: float = Field(default=0.15, ge=0, le=1)
    risk_weight: float = Field(default=0.1, ge=0, le=1)
    source_group: str
    fresh_seconds: int = Field(default=300, ge=1, le=604800)
    stale_seconds: int = Field(default=1800, ge=2, le=2592000)

    @model_validator(mode="after")
    def validate_freshness(self) -> ComponentConfig:
        if self.fresh_seconds >= self.stale_seconds:
            raise ValueError("fresh_seconds must be below stale_seconds")
        if self.source_group not in {"data", "structure", "participation", "context", "event_risk"}:
            raise ValueError("unknown source group")
        return self


def _default_components() -> dict[str, ComponentConfig]:
    return {
        "market_data": ComponentConfig(directional_weight=0, confidence_weight=0.10, risk_weight=0.10, source_group="data", fresh_seconds=60, stale_seconds=300),
        "market_regime": ComponentConfig(directional_weight=0.20, confidence_weight=0.15, risk_weight=0.15, source_group="context"),
        "smc": ComponentConfig(directional_weight=0.25, confidence_weight=0.20, risk_weight=0.05, source_group="structure"),
        "liquidity": ComponentConfig(directional_weight=0.15, confidence_weight=0.15, risk_weight=0.15, source_group="structure"),
        "volume_profile": ComponentConfig(directional_weight=0.15, confidence_weight=0.15, risk_weight=0.10, source_group="participation"),
        "institutional_flow": ComponentConfig(directional_weight=0.25, confidence_weight=0.20, risk_weight=0.10, source_group="participation"),
        "economic_calendar": ComponentConfig(directional_weight=0, confidence_weight=0.05, risk_weight=0.35, source_group="event_risk", fresh_seconds=300, stale_seconds=1800),
    }


class EvidencePolicy(FrozenConfig):
    minimum_components: int = Field(default=2, ge=1, le=7)
    minimum_independent_groups: int = Field(default=2, ge=1, le=5)
    single_source_confidence_ceiling: float = Field(default=35, ge=0, le=100)
    degraded_confidence_ceiling: float = Field(default=65, ge=0, le=100)
    stale_confidence_ceiling: float = Field(default=45, ge=0, le=100)
    maximum_missing_directional_weight: float = Field(default=0.60, ge=0, le=1)


class ConflictPolicy(FrozenConfig):
    directional_gap_threshold: float = Field(default=1.20, gt=0, le=2)
    severe_gap_threshold: float = Field(default=1.60, gt=0, le=2)
    moderate_penalty: float = Field(default=8, ge=0, le=100)
    severe_penalty: float = Field(default=15, ge=0, le=100)
    maximum_total_penalty: float = Field(default=35, ge=0, le=100)

    @model_validator(mode="after")
    def validate_thresholds(self) -> ConflictPolicy:
        if self.directional_gap_threshold >= self.severe_gap_threshold:
            raise ValueError("conflict threshold must be below severe threshold")
        return self


class LabelThresholds(FrozenConfig):
    strong_bearish: float = Field(default=-70, ge=-100, le=100)
    bearish: float = Field(default=-35, ge=-100, le=100)
    slightly_bearish: float = Field(default=-10, ge=-100, le=100)
    slightly_bullish: float = Field(default=10, ge=-100, le=100)
    bullish: float = Field(default=35, ge=-100, le=100)
    strong_bullish: float = Field(default=70, ge=-100, le=100)

    @model_validator(mode="after")
    def ordered(self) -> LabelThresholds:
        values = (self.strong_bearish, self.bearish, self.slightly_bearish, self.slightly_bullish, self.bullish, self.strong_bullish)
        if list(values) != sorted(values) or len(set(values)) != len(values):
            raise ValueError("directional thresholds must be strictly increasing")
        return self


class APIConfig(FrozenConfig):
    maximum_page_size: int = Field(default=200, ge=1, le=1000)
    maximum_history_days: int = Field(default=365, ge=1, le=3650)


class RetentionConfig(FrozenConfig):
    live_days: int = Field(default=180, ge=1, le=3650)
    replay_days: int = Field(default=30, ge=1, le=3650)
    cleanup_batch_size: int = Field(default=500, ge=1, le=10000)


class AIScoringConfig(FrozenConfig):
    enabled: bool = True
    engine_version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    schema_version: str = "1.0"
    policy_name: str = Field(default="weighted_intelligence_v1", pattern=r"^[a-z][a-z0-9_]{2,63}$")
    policy_version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    configuration_version: str = Field(default="ai-scoring-1.0", pattern=r"^[a-z0-9][a-z0-9._-]+$")
    components: dict[str, ComponentConfig] = Field(default_factory=_default_components)
    evidence: EvidencePolicy = Field(default_factory=EvidencePolicy)
    conflicts: ConflictPolicy = Field(default_factory=ConflictPolicy)
    labels: LabelThresholds = Field(default_factory=LabelThresholds)
    api: APIConfig = Field(default_factory=APIConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    future_clock_skew_seconds: int = Field(default=5, ge=0, le=300)
    output_precision: int = Field(default=4, ge=0, le=8)
    persistence_required_in_production: bool = True
    publish_replay_events: bool = False

    @model_validator(mode="after")
    def validate_components(self) -> AIScoringConfig:
        required = {"market_data", "market_regime", "smc", "liquidity", "volume_profile", "institutional_flow", "economic_calendar"}
        if set(self.components) != required:
            raise ValueError("components must define every approved source exactly once")
        if sum(item.directional_weight for item in self.components.values()) <= 0:
            raise ValueError("at least one directional component weight is required")
        return self
