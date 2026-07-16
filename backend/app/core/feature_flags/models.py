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


class FeatureFlagSettings(BaseModel):
    """Validated feature flag values with unknown flags rejected."""

    flags: dict[FeatureFlag, bool] = Field(default_factory=dict)

    def enabled(self, flag: FeatureFlag, default: bool = False) -> bool:
        return self.flags.get(flag, default)
