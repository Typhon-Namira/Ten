from collections.abc import Mapping

from backend.app.core.config import YamlConfigRepository

from .models import FeatureFlag, FeatureFlagSettings


class FeatureFlagService:
    """Read-only flag service; configuration changes require an explicit reload."""

    def __init__(
        self,
        settings: FeatureFlagSettings,
        overrides: Mapping[FeatureFlag, bool | None] | None = None,
    ) -> None:
        values = dict(settings.flags)
        for flag, value in (overrides or {}).items():
            if value is not None:
                values[flag] = value
        self._settings = settings.model_copy(update={"flags": values})

    @classmethod
    def from_yaml(
        cls,
        configs: YamlConfigRepository,
        overrides: Mapping[FeatureFlag, bool | None] | None = None,
    ) -> "FeatureFlagService":
        return cls(configs.load_model("feature_flags", FeatureFlagSettings), overrides)

    def is_enabled(self, flag: FeatureFlag, default: bool = False) -> bool:
        return self._settings.enabled(flag, default)

    def snapshot(self) -> dict[str, bool]:
        return {flag.value: value for flag, value in self._settings.flags.items()}
