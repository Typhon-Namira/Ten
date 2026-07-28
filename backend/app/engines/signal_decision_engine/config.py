from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ThresholdPair(FrozenConfig):
    minimum_for_observation: float = Field(ge=0, le=100)
    minimum_for_eligibility: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def ordered(self) -> ThresholdPair:
        if self.minimum_for_observation > self.minimum_for_eligibility:
            raise ValueError("observation threshold cannot exceed eligibility threshold")
        return self


class QualityThresholds(ThresholdPair):
    invalid_below: float = Field(default=10, ge=0, le=100)

    @model_validator(mode="after")
    def quality_order(self) -> QualityThresholds:
        if self.invalid_below >= self.minimum_for_observation:
            raise ValueError("invalid quality threshold must be below observation threshold")
        return self


class RiskThresholds(FrozenConfig):
    preferred_maximum: float = Field(default=40, ge=0, le=100)
    hard_block_minimum: float = Field(default=65, ge=0, le=100)

    @model_validator(mode="after")
    def ordered(self) -> RiskThresholds:
        if self.preferred_maximum >= self.hard_block_minimum:
            raise ValueError("preferred risk must be below hard block risk")
        return self


class ConflictConfig(FrozenConfig):
    severe_penalty_threshold: float = Field(default=25, ge=0, le=100)


class FreshnessWindow(FrozenConfig):
    eligible_max_age_seconds: int = Field(ge=1, le=2_592_000)
    observe_max_age_seconds: int = Field(ge=2, le=5_184_000)

    @model_validator(mode="after")
    def ordered(self) -> FreshnessWindow:
        if self.eligible_max_age_seconds >= self.observe_max_age_seconds:
            raise ValueError("eligible freshness must be below observe freshness")
        return self


def default_freshness() -> dict[str, FreshnessWindow]:
    return {
        "M1": FreshnessWindow(eligible_max_age_seconds=90, observe_max_age_seconds=180),
        "M5": FreshnessWindow(eligible_max_age_seconds=300, observe_max_age_seconds=600),
        "M15": FreshnessWindow(eligible_max_age_seconds=900, observe_max_age_seconds=1800),
        "M30": FreshnessWindow(eligible_max_age_seconds=1800, observe_max_age_seconds=3600),
        "H1": FreshnessWindow(eligible_max_age_seconds=3600, observe_max_age_seconds=7200),
        "H4": FreshnessWindow(eligible_max_age_seconds=14400, observe_max_age_seconds=28800),
        "D1": FreshnessWindow(eligible_max_age_seconds=86400, observe_max_age_seconds=172800),
        "W1": FreshnessWindow(eligible_max_age_seconds=604800, observe_max_age_seconds=1209600),
        "MN1": FreshnessWindow(eligible_max_age_seconds=2592000, observe_max_age_seconds=5184000),
    }


class EconomicEventConfig(FrozenConfig):
    enabled: bool = True
    fail_closed_when_source_unavailable: bool = True
    hard_phases: frozenset[str] = frozenset({"imminent", "at_event", "overlapping"})
    caution_phases: frozenset[str] = frozenset({"pre_event", "post_event", "cooldown"})


def default_regimes() -> dict[str, str]:
    eligible = {
        "trending_bull",
        "trending_bear",
        "imbalanced_bull",
        "imbalanced_bear",
        "markup_like",
        "markdown_like",
        "expansion_bull",
        "expansion_bear",
    }
    observe = {
        "ranging",
        "balanced",
        "accumulation_like",
        "distribution_like",
        "reaccumulation_like",
        "redistribution_like",
        "compression",
        "transition",
        "high_volatility",
        "low_volatility",
    }
    blocked = {"uncertain", "insufficient_data"}
    return {**{item: "eligible" for item in eligible}, **{item: "observe_only" for item in observe}, **{item: "blocked" for item in blocked}}


class CooldownConfig(FrozenConfig):
    enabled: bool = True
    eligible_repeat_seconds: int = Field(default=900, ge=0, le=604800)
    observe_repeat_seconds: int = Field(default=300, ge=0, le=604800)
    blocked_repeat_seconds: int = Field(default=300, ge=0, le=604800)


class ReversalConfig(FrozenConfig):
    enabled: bool = True
    lock_seconds: int = Field(default=600, ge=0, le=604800)
    additional_strength_required: float = Field(default=10, ge=0, le=100)
    additional_confidence_required: float = Field(default=5, ge=0, le=100)


class HysteresisConfig(FrozenConfig):
    enabled: bool = True
    strength_relief: float = Field(default=5, ge=0, le=25)
    confidence_relief: float = Field(default=5, ge=0, le=25)


def default_validity() -> dict[str, dict[str, int]]:
    return {
        "eligible": {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800, "MN1": 2592000},
        "observe_only": {"M1": 30, "M5": 180, "M15": 600, "M30": 1200, "H1": 1800, "H4": 7200, "D1": 43200, "W1": 302400, "MN1": 1296000},
        "blocked": {"default": 300},
        "insufficient_evidence": {"default": 180},
        "invalid": {"default": 60},
    }


class APIConfig(FrozenConfig):
    maximum_page_size: int = Field(default=200, ge=1, le=1000)
    maximum_history_days: int = Field(default=365, ge=1, le=3650)


class RetentionConfig(FrozenConfig):
    live_days: int = Field(default=180, ge=1, le=3650)
    replay_days: int = Field(default=30, ge=1, le=3650)
    cleanup_batch_size: int = Field(default=500, ge=1, le=10000)


class SignalDecisionConfig(FrozenConfig):
    enabled: bool = True
    engine_version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    schema_version: str = "1.0"
    policy_name: str = Field(default="conservative_signal_policy", pattern=r"^[a-z][a-z0-9_]{2,63}$")
    policy_version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    configuration_version: str = Field(default="signal-decision-1.0", pattern=r"^[a-z0-9][a-z0-9._-]+$")
    direction_mapping: dict[str, str] = Field(default_factory=lambda: {
        "strong_bullish": "bullish", "bullish": "bullish", "slightly_bullish": "bullish", "neutral": "neutral",
        "slightly_bearish": "bearish", "bearish": "bearish", "strong_bearish": "bearish",
    })
    directional_strength: ThresholdPair = ThresholdPair(minimum_for_observation=20, minimum_for_eligibility=45)
    confidence: ThresholdPair = ThresholdPair(minimum_for_observation=45, minimum_for_eligibility=70)
    risk: RiskThresholds = RiskThresholds()
    data_quality: QualityThresholds = QualityThresholds(invalid_below=10, minimum_for_observation=45, minimum_for_eligibility=70)
    alignment: ThresholdPair = ThresholdPair(minimum_for_observation=35, minimum_for_eligibility=60)
    conflicts: ConflictConfig = ConflictConfig()
    freshness: dict[str, FreshnessWindow] = Field(default_factory=default_freshness)
    economic_event: EconomicEventConfig = EconomicEventConfig()
    market_regimes: dict[str, str] = Field(default_factory=default_regimes)
    cooldown: CooldownConfig = CooldownConfig()
    reversal: ReversalConfig = ReversalConfig()
    hysteresis: HysteresisConfig = HysteresisConfig()
    validity: dict[str, dict[str, int]] = Field(default_factory=default_validity)
    api: APIConfig = APIConfig()
    retention: RetentionConfig = RetentionConfig()
    future_clock_skew_seconds: int = Field(default=5, ge=0, le=300)
    output_precision: int = Field(default=4, ge=0, le=8)
    minimum_risk_reward: float = Field(default=2.0, ge=1, le=10)
    persistence_required_in_production: bool = True
    publish_replay_events: bool = False
    publish_replay_features: bool = False

    @model_validator(mode="after")
    def validate_policy(self) -> SignalDecisionConfig:
        expected = {"strong_bullish", "bullish", "slightly_bullish", "neutral", "slightly_bearish", "bearish", "strong_bearish"}
        if set(self.direction_mapping) != expected or set(self.direction_mapping.values()) != {"bullish", "bearish", "neutral"}:
            raise ValueError("direction mapping must exhaustively map approved AI labels")
        if set(self.freshness) != {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"}:
            raise ValueError("freshness must define every supported timeframe")
        if set(self.market_regimes.values()) - {"eligible", "observe_only", "blocked"}:
            raise ValueError("unknown market-regime policy outcome")
        for state, values in self.validity.items():
            if state not in {"eligible", "observe_only", "blocked", "insufficient_evidence", "invalid"} or not values or any(value <= 0 for value in values.values()):
                raise ValueError("invalid decision validity policy")
        return self

    def validity_seconds(self, state: str, timeframe: str) -> int:
        values = self.validity[state]
        return values.get(timeframe, values.get("default", 300))
