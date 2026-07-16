"""Centralized, dynamically loaded engine registry."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from backend.app.core.config import Settings, YamlConfigRepository, get_settings
from backend.app.core.exceptions import ConfigurationError
from backend.app.core.feature_flags import FeatureFlag, FeatureFlagService
from backend.app.engines.common import EngineLifecycleStatus, EngineState, EngineStatus
from backend.app.plugins import PluginRegistry

from .engine_factory import EngineBuildContext, EngineDefinition, EngineFactory
from .engine_loader import EngineLoader
from .pipeline_contracts import EngineExecutionResult, PipelineExecutionContext


class EngineSelection(BaseModel):
    enabled: bool = True
    version: str | None = None


class EngineRegistryConfig(BaseModel):
    engines: dict[str, EngineSelection] = Field(default_factory=dict)


@dataclass(frozen=True)
class RegisteredEngine:
    definition: EngineDefinition
    instance: Any | None
    enabled: bool
    disabled_reason: str | None = None


class EngineRegistry:
    """Owns configured engine instances; callers never instantiate engines directly."""

    def __init__(self, factory: EngineFactory, context: EngineBuildContext, config: EngineRegistryConfig) -> None:
        self.factory = factory
        self.context = context
        self.config = config
        self._engines: dict[str, RegisteredEngine] = {}
        self._initialize()

    def _initialize(self) -> None:
        for name, selection in self.config.engines.items():
            definition = self.factory.definition(name, selection.version)
            metadata = definition.metadata
            flag_enabled = True
            if metadata.feature_flag:
                flag_enabled = self.context.feature_flags.is_enabled(FeatureFlag(metadata.feature_flag))
            enabled = selection.enabled and metadata.enabled and flag_enabled and metadata.status != EngineLifecycleStatus.DISABLED
            instance = None
            reason = None
            if enabled:
                engine_config = self.context.configs.load(metadata.config_key)
                instance = self.factory.create(definition, self.context, engine_config)
            else:
                reason = "Disabled by registry metadata, feature flag, or engine selection."
            self._engines[name] = RegisteredEngine(definition=definition, instance=instance, enabled=enabled, disabled_reason=reason)

    def get(self, name: str) -> Any:
        registration = self.registration(name)
        if not registration.enabled or registration.instance is None:
            raise ConfigurationError(f"Engine is disabled: {name}")
        return registration.instance

    def registration(self, name: str) -> RegisteredEngine:
        try:
            return self._engines[name]
        except KeyError as exc:
            raise ConfigurationError(f"Engine is not configured: {name}") from exc

    async def execute(self, name: str, context: PipelineExecutionContext) -> EngineExecutionResult:
        registration = self.registration(name)
        if not registration.enabled or registration.instance is None:
            raise ConfigurationError(f"Engine is disabled: {name}")
        return await registration.definition.executor(registration.instance, context)

    def statuses(self) -> list[EngineStatus]:
        statuses: list[EngineStatus] = []
        for registration in self._engines.values():
            metadata = registration.definition.metadata
            state = EngineState.READY if registration.enabled else EngineState.OFFLINE
            if metadata.name == "ai_scoring" and registration.enabled and not self.context.settings.openrouter_api_key:
                state = EngineState.DEGRADED
            statuses.append(
                EngineStatus(
                    name=metadata.name,
                    version=metadata.version,
                    state=state,
                    details=registration.disabled_reason or "Registered and enabled.",
                    compatibility_version=metadata.compatibility_version,
                    created_date=metadata.created_date,
                    lifecycle_status=metadata.status,
                    dependencies=metadata.dependencies,
                    description=metadata.description,
                    enabled=registration.enabled,
                )
            )
        return sorted(statuses, key=lambda item: item.name)

    def names(self, enabled_only: bool = False) -> tuple[str, ...]:
        return tuple(name for name, registration in self._engines.items() if not enabled_only or registration.enabled)

    def override(self, name: str, instance: Any) -> None:
        """Replace an enabled adapter for tests or application-level dependency injection."""

        registration = self.registration(name)
        self._engines[name] = RegisteredEngine(definition=registration.definition, instance=instance, enabled=True)


def build_engine_registry(
    *,
    settings: Settings | None = None,
    configs: YamlConfigRepository | None = None,
    plugins: PluginRegistry | None = None,
) -> EngineRegistry:
    """Composition root driven entirely by discovered registrations and YAML."""

    resolved_configs = configs or YamlConfigRepository()
    flags = FeatureFlagService.from_yaml(resolved_configs)
    factory = EngineFactory()
    EngineLoader(factory).discover()
    context = EngineBuildContext(settings=settings or get_settings(), configs=resolved_configs, feature_flags=flags, plugins=plugins or PluginRegistry())
    registry_config = resolved_configs.load_model("engine_registry", EngineRegistryConfig)
    return EngineRegistry(factory, context, registry_config)
