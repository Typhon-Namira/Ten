from backend.app.core.feature_flags import FeatureFlag, FeatureFlagService, FeatureFlagSettings
from backend.app.core.config import YamlConfigRepository


def test_feature_flags_are_typed_and_snapshot_safe() -> None:
    service = FeatureFlagService(FeatureFlagSettings(flags={FeatureFlag.ENABLE_SMC: True, FeatureFlag.ENABLE_REPLAY: False}))
    assert service.is_enabled(FeatureFlag.ENABLE_SMC)
    assert not service.is_enabled(FeatureFlag.ENABLE_REPLAY)
    assert service.snapshot() == {"EnableSMC": True, "EnableReplay": False}


def test_ai_centric_phase_one_flags_exist_and_are_disabled_by_default() -> None:
    service = FeatureFlagService.from_yaml(YamlConfigRepository())
    phase_one = (
        FeatureFlag.AI_CENTRIC_SHADOW_MODE,
        FeatureFlag.AI_SIGNAL_PROPOSALS,
        FeatureFlag.AI_SIGNAL_MONITORING,
        FeatureFlag.AI_SIGNAL_PUBLICATION,
        FeatureFlag.AI_SIGNAL_ADJUSTMENTS,
    )
    assert all(service.is_enabled(flag) is False for flag in phase_one)
