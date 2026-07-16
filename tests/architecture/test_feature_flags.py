from backend.app.core.feature_flags import FeatureFlag, FeatureFlagService, FeatureFlagSettings


def test_feature_flags_are_typed_and_snapshot_safe() -> None:
    service = FeatureFlagService(FeatureFlagSettings(flags={FeatureFlag.ENABLE_SMC: True, FeatureFlag.ENABLE_REPLAY: False}))
    assert service.is_enabled(FeatureFlag.ENABLE_SMC)
    assert not service.is_enabled(FeatureFlag.ENABLE_REPLAY)
    assert service.snapshot() == {"EnableSMC": True, "EnableReplay": False}
