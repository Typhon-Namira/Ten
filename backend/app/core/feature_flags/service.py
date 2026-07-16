from backend.app.core.config import YamlConfigRepository

from .models import FeatureFlag, FeatureFlagSettings


class FeatureFlagService:
    """Read-only flag service; configuration changes require an explicit reload."""

    def __init__(self, settings: FeatureFlagSettings) -> None:
        self._settings = settings

    @classmethod
    def from_yaml(cls, configs: YamlConfigRepository) -> "FeatureFlagService":
        return cls(configs.load_model("feature_flags", FeatureFlagSettings))

    def is_enabled(self, flag: FeatureFlag, default: bool = False) -> bool:
        return self._settings.enabled(flag, default)

    def snapshot(self) -> dict[str, bool]:
        return {flag.value: value for flag, value in self._settings.flags.items()}
