"""Configuration-driven platform feature flags."""

from enum import StrEnum

from pydantic import BaseModel, Field


class FeatureFlag(StrEnum):
    ENABLE_SMC = "EnableSMC"
    ENABLE_LIQUIDITY = "EnableLiquidity"
    ENABLE_FLOW = "EnableFlow"
    ENABLE_VOLUME_PROFILE = "EnableVolumeProfile"
    ENABLE_ECONOMIC_FILTER = "EnableEconomicFilter"
    ENABLE_AI = "EnableAI"
    ENABLE_REPLAY = "EnableReplay"
    ENABLE_MARKET_REGIME = "EnableMarketRegime"
    ENABLE_DASHBOARD_MODULES = "EnableDashboardModules"
    AI_CENTRIC_SHADOW_MODE = "ai_centric_shadow_mode"
    AI_SIGNAL_PROPOSALS = "ai_signal_proposals"
    AI_SIGNAL_MONITORING = "ai_signal_monitoring"
    AI_SIGNAL_PUBLICATION = "ai_signal_publication"
    AI_SIGNAL_ADJUSTMENTS = "ai_signal_adjustments"


class FeatureFlagSettings(BaseModel):
    """Validated feature flag values with unknown flags rejected."""

    flags: dict[FeatureFlag, bool] = Field(default_factory=dict)

    def enabled(self, flag: FeatureFlag, default: bool = False) -> bool:
        return self.flags.get(flag, default)
