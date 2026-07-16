"""Entry-point based external plugin discovery."""

from importlib.metadata import entry_points

from backend.app.core.exceptions import ConfigurationError

from .base import Plugin
from .registry import PluginRegistry


class PluginLoader:
    """Loads `ten.plugins` package entry points without modifying TEN source."""

    entry_point_group = "ten.plugins"

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry

    def discover(self) -> tuple[str, ...]:
        loaded: list[str] = []
        for entry_point in entry_points(group=self.entry_point_group):
            plugin_factory = entry_point.load()
            plugin = plugin_factory()
            if not isinstance(plugin, Plugin):
                raise ConfigurationError(f"Plugin entry point did not return Plugin: {entry_point.name}")
            self.registry.register(plugin)
            loaded.append(entry_point.name)
        return tuple(sorted(loaded))
