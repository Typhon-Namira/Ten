"""Version-aware engine definition catalog and construction service."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from backend.app.core.config import Settings, YamlConfigRepository
from backend.app.core.exceptions import ConfigurationError
from backend.app.core.feature_flags import FeatureFlagService
from backend.app.engines.common import EngineMetadata
from backend.app.plugins import PluginRegistry

from .pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

EngineBuilder = Callable[["EngineBuildContext", Mapping[str, Any]], Any]
EngineExecutor = Callable[[Any, PipelineExecutionContext], Awaitable[EngineExecutionResult]]


@dataclass(frozen=True)
class EngineBuildContext:
    settings: Settings
    configs: YamlConfigRepository
    feature_flags: FeatureFlagService
    plugins: PluginRegistry


@dataclass(frozen=True)
class EngineDefinition:
    metadata: EngineMetadata
    builder: EngineBuilder
    executor: EngineExecutor


class EngineFactory:
    """Stores coexisting engine versions and constructs selected implementations."""

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], EngineDefinition] = {}

    def register(self, metadata: EngineMetadata, builder: EngineBuilder, executor: EngineExecutor) -> None:
        key = (metadata.name, metadata.version)
        if key in self._definitions:
            raise ConfigurationError(f"Engine version already registered: {metadata.name}@{metadata.version}")
        self._definitions[key] = EngineDefinition(metadata=metadata, builder=builder, executor=executor)

    def definition(self, name: str, version: str | None = None) -> EngineDefinition:
        candidates = [definition for (engine_name, _), definition in self._definitions.items() if engine_name == name]
        if version is not None:
            candidates = [definition for definition in candidates if definition.metadata.version == version]
        if not candidates:
            suffix = f"@{version}" if version else ""
            raise ConfigurationError(f"Engine is not registered: {name}{suffix}")
        return max(candidates, key=lambda item: tuple(int(part) for part in item.metadata.version.split(".")))

    def definitions(self) -> tuple[EngineDefinition, ...]:
        return tuple(sorted(self._definitions.values(), key=lambda item: (item.metadata.name, item.metadata.version)))

    def create(self, definition: EngineDefinition, context: EngineBuildContext, config: Mapping[str, Any]) -> Any:
        return definition.builder(context, config)
