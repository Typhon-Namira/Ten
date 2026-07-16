from backend.app.core.exceptions import ConfigurationError

from .base import Plugin, PluginType


class PluginRegistry:
    """Runtime plugin catalog keyed by extension type and stable name."""

    def __init__(self) -> None:
        self._plugins: dict[tuple[PluginType, str], Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        key = (plugin.metadata.plugin_type, plugin.metadata.name)
        if key in self._plugins:
            raise ConfigurationError(f"Plugin already registered: {plugin.metadata.name}")
        self._plugins[key] = plugin

    def get(self, plugin_type: PluginType, name: str) -> Plugin:
        try:
            return self._plugins[(plugin_type, name)]
        except KeyError as exc:
            raise ConfigurationError(f"Plugin not registered: {plugin_type.value}/{name}") from exc

    def list(self, plugin_type: PluginType | None = None) -> tuple[Plugin, ...]:
        return tuple(plugin for (kind, _), plugin in self._plugins.items() if plugin_type is None or kind == plugin_type)
